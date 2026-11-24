"""Perception, grounding, belief fusion, and grasp-planning mixin for SkillRuntime."""

from __future__ import annotations

import time

import numpy as np

from ..grasping import plan_grasps_from_fix
from ..grasping.camera_grasp import DEFAULT_INSERTION_DEPTH_M, plan_grasp_from_mask
from ..perception.colors import detection_color, parse_color_query
from ..perception.grounding import (
    Extrinsics,
    mask_to_points_cam,
    oriented_bbox,
)
from ..types import Detection, Frame, SkillError, transform_points
from .specs import _DESTINATION_FAMILY, _jpeg


def _rt():
    from . import runtime as _runtime_mod
    return _runtime_mod


class _PerceptionMixin:
    def observe(self) -> Frame:
        frame = self.camera.get_frame()
        frame = self.depth.ensure_depth(frame)
        self.last_frame = frame
        self.arm.harness.heartbeat()
        return frame


    def frame_jpeg(self) -> bytes | None:
        if self.last_frame is None:
            return None
        return _jpeg(self.last_frame.rgb)


    def _update_beliefs_from_frame(self, frame: Frame, dets, T=None) -> list[dict]:
        summaries = []
        if not frame.has_depth:
            return [
                {"label": d.label, "conf": round(d.conf, 2), "position": None}
                for d in dets
            ]
        if T is None:
            T = (frame.T_base_cam if frame.T_base_cam is not None
                 else self.extrinsics.cam_to_base())
        for d in dets:
            mask = d.mask
            if mask is None:
                h, w = frame.rgb.shape[:2]
                mask = np.zeros((h, w), dtype=bool)
                x0, y0, x1, y1 = d.bbox.astype(int)
                mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < 10:
                continue
            pts_base = transform_points(T, pts_cam)
            center, extents, _ = oriented_bbox(pts_base)
            self.beliefs.update(
                d.label, center, d.conf, extent=extents,
                top_z=float(pts_base[:, 2].max()), t=frame.t,
                points=pts_base if d.mask is not None else None,
            )
            summaries.append(
                {
                    "label": d.label,
                    "conf": round(d.conf, 2),
                    "position": [round(float(x), 3) for x in center],
                }
            )
        return summaries


    def _resolve_query(self, query: str) -> dict:
        """Turn a user phrase ("pink object", "red mug", "bottle") into
        detector inputs, using the live belief store when it already knows
        the answer (the WorldWatcher keeps it warm)."""
        color, noun = parse_color_query(query)
        belief = self.beliefs.find(query)
        near = belief.position.copy() if belief is not None else None
        prompts = vocab = prefer = None
        if noun:
            q = query.strip().lower()
            prompts = [noun] if noun == q else [q, noun]
            # v4: a destination-container query gets the whole synonym
            # family as prompts, so YOLOE can latch onto whichever noun
            # reads best in this view ("bin" often scores where "box"
            # doesn't, and vice versa). The fix's label is canonicalized
            # back to the queried word in _localize.
            fam = _DESTINATION_FAMILY.get(noun)
            if fam:
                for alt in sorted(fam - {noun}):
                    prompts.append(alt)
        elif belief is not None and (color is None or belief.color == color):
            # Warm path: the world model already knows the answer. When the
            # class is in the default vocabulary, detect with the WHOLE
            # vocabulary (no set_classes churn -- YOLOE re-embeds text on
            # every class change) and prefer that label among candidates.
            if belief.label in self._default_classes:
                vocab, prefer = self._default_classes, belief.label
            else:
                prompts = [belief.label]
        else:
            # No belief, or an UNTAGGED belief matched a color query only by
            # fallback -- don't trust it blindly: scan the vocabulary and let
            # the color filter decide (near still helps ranking).
            vocab = self._default_classes
        return {
            "prompts": prompts, "vocab": vocab, "color": color,
            "near_xyz": near, "belief": belief, "prefer_label": prefer,
        }


    def _mask_above_table(self, frame, mask, margin_m: float = 0.004):
        """Zero out mask pixels whose backprojected base-frame z sits at or
        below the table plane (plus a small margin).

        Detector masks almost always include some table surface around or
        under the object; feeding those pixels to the camera-frame grasp
        planner puts the grasp center on the tabletop instead of on the
        object, and the descent leg then rams the table and shoves the
        target out of the jaws' reach -- the classic "grasp missed and the
        toy slid away" failure. ``np.nonzero`` row-major order matches
        ``mask_to_points_cam``'s point order; on any inconsistency the
        ORIGINAL mask is returned (fail-open: a table-contaminated grasp
        attempt still beats no candidate at all).

        Returns ``(filtered_mask, dz_shaved)``: ``dz_shaved`` (>= 0) is how
        much the filter raised the mask's lowest 3D point. It is ~margin_m
        when the object sits on the table (its bottom ring pixels lie AT
        the table plane and get stripped too), which would shift the
        planned grasp UP by the same amount -- callers must lower the
        grasp z by ``dz_shaved`` to put it back (v3 fix for "grasps
        slightly too high").
        """
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        T = (frame.T_base_cam if frame.T_base_cam is not None
             else self.extrinsics.cam_to_base())
        try:
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < 30:
                return mask, 0.0
            pts_base = transform_points(T, pts_cam)
            keep = pts_base[:, 2] > table_z + margin_m
            if int(keep.sum()) < 30:
                return mask, 0.0  # near-flat object: don't gut the mask
            ys, xs = np.nonzero(mask)
            if ys.shape[0] != pts_cam.shape[0]:
                return mask, 0.0  # ordering assumption broken: fail open
            out = np.zeros_like(mask)
            out[ys[keep], xs[keep]] = True
            dz = float(pts_base[keep][:, 2].min() - pts_base[:, 2].min())
            return out, float(np.clip(dz, 0.0, 0.05))
        except Exception:
            return mask, 0.0


    def _clamp_grasp_z_floor(self, grasps: list) -> None:
        """Shallow any grasp point that dipped a few mm under the workspace
        floor (in place -- returns None).

        Mirrors the legacy path's z-floor clamp: a millimetre under the
        floor must shallow the grasp slightly, not abort it, but executing a
        grasp point below the table plane means the descent rams the
        tabletop.
        """
        harness = getattr(self.arm, "harness", None)
        lim = getattr(harness, "limits", None)
        try:
            ws_z = float(lim.workspace_min[2])
        except (AttributeError, TypeError, ValueError, IndexError):
            ws_z = float(self.cfg.safety.workspace.min[2])
        try:
            table_z = float(lim.table_z)
        except (AttributeError, TypeError, ValueError):
            table_z = float(self.cfg.safety.get("table_z", 0.0))
        z_floor = max(
            ws_z + 0.002,
            table_z + float(self.cfg.grasp.get("min_grasp_z_m", 0.008)),
        )
        for g in grasps:
            dz = z_floor - float(g.position[2])
            if 0.0 < dz <= 0.03:  # bigger misses are garbage; vetting drops them
                g.position = np.asarray(g.position, dtype=float).copy()
                g.position[2] = z_floor


    def _plan_grasps(self, fix, label: str | None = None) -> list:
        """Grasp candidates: camera-frame planner primary, OBB fallback.

        Pipeline order (each step is skipped if its preconditions fail,
        the next layer picks up):

        1. **camera-frame planner** (rebot_grasp-style) — runs when the
           detection has a real ``mask`` and the last frame has depth.
           2D OBB on the mask → backproject → approach toward the camera
           origin → ``insertion_depth_m`` shift into the object. Best
           when the camera is tilted: the gripper dives along the actual
           line of sight, not a hard-coded base-frame vector.

        2. **learned GraspGen-X** — only when ``backend: graspgenx`` and
           the server is up; concatenated before the analytic fallback.

        3. **analytic OBB planner** (``plan_grasps_from_fix``) — always
           available; serves as the IK fallback when the camera-frame
           planner produces nothing (mock stack, mono camera, etc.).

        Fake-RL layer: past-attempt memory re-ranks and applies a learned
        z-nudge, just like before.
        """
        gcfg = self.cfg.grasp

        # 1. camera-frame grasp planner (rebot_grasp-style, primary)
        cam_grasps: list = []
        det = getattr(fix, "detection", None)
        last_frame = getattr(self, "last_frame", None)
        if (
            det is not None
            and getattr(det, "mask", None) is not None
            and last_frame is not None
            and getattr(last_frame, "has_depth", False)
            and getattr(last_frame, "depth_m", None) is not None
        ):
            try:
                T_cam2base = self.extrinsics.cam_to_base()
                mask = det.mask
                dz_shaved = 0.0
                if getattr(last_frame, "has_depth", False):
                    # Strip table-plane pixels BEFORE backprojection, or the
                    # grasp center lands on the tabletop (see
                    # _mask_above_table). The filter also removes the
                    # object's own bottom ring (it sits AT the table
                    # plane), raising the apparent object by dz_shaved --
                    # compensate below or every grasp lands a few mm high.
                    mask, dz_shaved = self._mask_above_table(last_frame, mask)
                cam_grasps = plan_grasp_from_mask(
                    mask=mask,
                    depth_m=last_frame.depth_m,
                    K=last_frame.K,
                    T_cam2base=T_cam2base,
                    insertion_depth_m=float(
                        gcfg.get("insertion_depth_m", DEFAULT_INSERTION_DEPTH_M)
                    ),
                    max_width_m=self._max_width,
                    label=label or fix.label,
                    confidence=float(getattr(det, "conf", 1.0)),
                )
                if cam_grasps and dz_shaved > 1e-4:
                    # v3: the shave raised the object's apparent bottom by
                    # dz_shaved and with it every planned grasp height --
                    # lower them back by the same amount, or each grasp
                    # lands a few mm ABOVE the toy. ``mask_shave_compensate``
                    # (default 1.0) tunes the correction: set 0.5 if this
                    # rig's planner anchors to the OBB CENTER rather than
                    # the bottom (center only shifts by half the shave).
                    comp = float(gcfg.get("mask_shave_compensate", 1.0))
                    for g in cam_grasps:
                        p = np.asarray(g.position, dtype=float).copy()
                        p[2] -= dz_shaved * comp
                        g.position = p
                if cam_grasps:
                    self.memory.add(
                        "note",
                        f"camera_grasp: {len(cam_grasps)} candidate(s) "
                        f"(insertion_depth={float(gcfg.get('insertion_depth_m', DEFAULT_INSERTION_DEPTH_M))*1000:.0f}mm)",
                    )
            except Exception as e:
                self.memory.add(
                    "note", f"camera_grasp failed ({str(e)[:90]}); OBB fallback"
                )

        # 2. analytic OBB planner (fallback / supplement)
        obb = plan_grasps_from_fix(
            fix,
            table_z=float(self.cfg.safety.get("table_z", 0.0)),
            max_width_m=self._max_width,
            depth_fraction=float(gcfg.get("depth_fraction", 0.5)),
            approach_pitch_rad=float(gcfg.get("approach_pitch_rad", 1.2)),
        )

        # 3. learned GraspGen-X (optional, when configured)
        if str(gcfg.get("backend", "obb")) != "graspgenx":
            grasps = cam_grasps + obb
        else:
            try:
                if self._graspgenx is None:
                    from ..grasping.graspgenx_backend import GraspGenXPlanner

                    self._graspgenx = GraspGenXPlanner(gcfg)
                learned = self._graspgenx.plan(fix, max_width_m=self._max_width)
                self.memory.add(
                    "note",
                    f"graspgenx: {len(learned)} grasps in {self._graspgenx.last_latency_s}s "
                    f"(top {learned[0].quality:.2f})",
                )
                grasps = learned + cam_grasps + obb
            except Exception as e:
                self.memory.add(
                    "note", f"graspgenx unavailable ({str(e)[:90]}); OBB fallback"
                )
                grasps = cam_grasps + obb

        # ---- fake-RL memory prior: re-rank + z-nudge -----------------------
        lbl = label or getattr(fix, "label", None) or "object"
        try:
            prior = self.grasp_memory.prior(lbl, fix)
            if prior:
                grasps = self.grasp_memory.rerank(grasps, lbl, fix)
                dz = float(prior["nudges"].get("grasp_z_delta", 0.0))
                if abs(dz) > 1e-4:
                    for g in grasps:
                        p = np.asarray(g.position, dtype=float).copy()
                        p[2] = p[2] + dz
                        g.position = p
                self._last_grasp_z_nudge = dz
                self.memory.add(
                    "note",
                    f"grasp-memory prior for {prior['profile']}: "
                    f"{prior['wins']}W/{prior['losses']}L sr={prior['success_rate']:.0%}"
                    + (f", z_nudge={dz:+.3f}" if abs(dz) > 1e-4 else "")
                    + (f", avoid={prior['top_fail']}" if prior.get("top_fail") else ""),
                )
            else:
                self._last_grasp_z_nudge = 0.0
        except Exception as e:
            self._last_grasp_z_nudge = 0.0
            self.memory.add("note", f"grasp-memory prior skipped ({str(e)[:60]})")
        return grasps


    def _localize(self, query: str, spatial_hint: str | None = None):
        """Fresh frame + color/proximity-aware 3D fix for a user phrase.

        A single detector pass at low confidence flickers frame to frame, so
        a miss is retried on a couple of FRESH frames before giving up on
        detection. After that it falls back to a fresh belief: the
        WorldWatcher fuses at ~3 Hz with temporal stability, and an object
        the world model has seen recently is still perfectly actionable
        (that is the whole point of keeping the model warm). Staleness is
        measured against the moment the current motion skill STARTED, not
        "now" -- the watcher is paused while the arm moves, so beliefs age
        artificially during exactly the retries that need them."""
        r = self._resolve_query(query)
        # v4: canonicalize family labels (see _DESTINATION_FAMILY). A toy
        # box detected as "basket" is rewritten to the queried word BEFORE
        # it reaches the belief store, so later "box" lookups match.
        qnoun = parse_color_query(query)[1]

        def _canon(fix):
            fam = _DESTINATION_FAMILY.get((qnoun or "").lower())
            lbl = getattr(getattr(fix, "detection", None), "label", None)
            if (fam and lbl and lbl.lower() in fam
                    and lbl.lower() != qnoun.lower()):
                fix.detection.label = qnoun
            return fix

        frame = None
        last_err: SkillError | None = None
        tries = int(self.cfg.get("perception_loop", {}).get("localize_frames", 3))
        for _ in range(max(tries, 1)):
            frame = self.observe()
            try:
                fix = _rt().localize_object(
                    frame, query, self.detector, self.extrinsics,
                    prompts=r["prompts"], spatial_hint=spatial_hint,
                    color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                    prefer_label=r["prefer_label"],
                )
                return frame, _canon(fix)
            except SkillError as e:
                last_err = e
        # Second filter: THE OTHER CAMERAS. An object 40 px small (or
        # occluded) in the primary view may be plainly visible from the side
        # or wrist camera; each pass costs one detector call.
        from ..perception.grounding import Extrinsics as _Ext

        for cam in self._other_cams():
            try:
                cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                if not cframe.has_depth:
                    continue
                ext = (_Ext(T=cframe.T_base_cam)
                       if cframe.T_base_cam is not None else cam.extrinsics)
                fix = _rt().localize_object(
                    cframe, query, self.detector, ext,
                    prompts=r["prompts"], spatial_hint=spatial_hint,
                    color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                    prefer_label=r["prefer_label"],
                )
                self.memory.add(
                    "note",
                    f"localize {query!r}: primary camera missed it; found "
                    f"through {getattr(cam.stream, 'name', 'another camera')}",
                )
                return cframe, _canon(fix)
            except Exception:
                continue  # a miss or camera hiccup: try the next view

        belief = r["belief"] or self.beliefs.find(query)
        max_age = float(self.cfg.get("perception_loop", {}).get(
            "belief_fallback_age_s", 3.0))
        # Staleness reference: the last fresh re-scan during this motion (a
        # re-scan that FAILED to see the object must age the belief), else
        # the motion start (the watcher is paused while the arm moves), else
        # now.
        if self._last_reobserve_t is not None:
            ref_t = self._last_reobserve_t
        elif self._motion_t0 is not None:
            ref_t = self._motion_t0
        else:
            ref_t = time.monotonic()
        mem_ok = (
            belief is not None
            and (ref_t - belief.last_seen_t) <= max_age
            # Grasping from memory demands an EXACT color match: the belief
            # store's neighbor tolerance (pink~red) is for conversation, not
            # for choosing what the jaws close on.
            and (r["color"] is None or belief.color == r["color"])
        )
        if mem_ok:
            self.memory.add(
                "note",
                f"localize {query!r}: instant detection missed; using the "
                f"world-model fix ({belief.label}, "
                f"{time.monotonic() - belief.last_seen_t:.1f}s old)",
            )
            return frame, _canon(self._fix_from_belief(query, belief))
        # Last filter: VLM grounding. YOLOE's text embeddings miss what a
        # full VLM reads easily; one slow call only ever runs on this
        # failure path.
        fix = self._vlm_ground_fix(frame, query)
        if fix is not None:
            return frame, _canon(fix)
        raise last_err


    def _vlm_ground_fix(self, frame, query: str):
        """Ask the VLM (2nd perception filter) for the object's bbox and
        lift it to a 3D fix -- trying EVERY camera's view (the primary may
        see the object at 40 px while the side camera fills the frame with
        it). Fail-soft: any error returns None so the caller reports the
        original detector failure."""
        gcfg = self.cfg.get("grounder", None)
        if not gcfg:
            return None
        try:
            if self._grounder is None:
                from ..perception.vlm_ground import VLMGrounder

                self._grounder = VLMGrounder(
                    base_url=str(gcfg["base_url"]),
                    model=str(gcfg.get("model", "")),
                    timeout_s=float(gcfg.get("timeout_s", 45.0)),
                )
            # Views to try: (frame, cam->base T). Primary first, then the
            # other fusing cameras with their own extrinsics.
            views = []
            if frame is not None and frame.has_depth:
                views.append((frame, frame.T_base_cam if frame.T_base_cam
                              is not None else self.extrinsics.cam_to_base()))
            for cam in self._other_cams():
                try:
                    cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                except Exception:
                    continue
                if not cframe.has_depth:
                    continue
                views.append((cframe, cframe.T_base_cam if cframe.T_base_cam
                              is not None else cam.extrinsics.cam_to_base()))
            det = gframe = T = None
            for cframe, cT in views:
                det = self._grounder.ground(cframe, query)
                if det is not None:
                    gframe, T = cframe, cT
                    break
            if det is None:
                return None
            h, w = gframe.rgb.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = det.bbox.astype(int)
            mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(gframe, mask)
            if pts_cam.shape[0] < 30:
                return None
            pts = transform_points(T, pts_cam)
            # A bbox rectangle sweeps in table pixels; shave the table plane
            # or the OBB extent balloons (the GGX bbox-slab lesson).
            table_z = float(self.cfg.safety.get("table_z", 0.0))
            above = pts[pts[:, 2] > table_z + 0.005]
            if above.shape[0] >= 30:
                pts = above
            center, extents, axes = oriented_bbox(pts)
            from ..types import ObjectFix

            self.memory.add(
                "note",
                f"YOLOE missed {query!r}; the VLM (2nd filter) found it at "
                f"{[round(float(v), 2) for v in center]}",
            )
            return ObjectFix(
                label=query, position=center, points=pts,
                detection=det, extent=extents, axes=axes,
            )
        except Exception as e:
            self.memory.add("note", f"VLM grounding unavailable: {str(e)[:80]}")
            return None


    def _other_cams(self) -> list:
        """Non-primary fusing cameras from the watcher wiring: each entry
        has .stream, .depth and .extrinsics (an object may be visible from
        the side camera while the primary sees it at 5 px)."""
        cams = getattr(self.watcher, "_cams", None) if self.watcher else None
        if not cams:
            return []
        return [c for c in cams[1:] if getattr(c, "fuse", True)]


    def _reobserve(self, frames: int = 2) -> None:
        """Refresh beliefs with fresh detector passes over EVERY fusing
        camera while the WorldWatcher is paused (motion skills hold it):
        fusion by hand, exactly what the 3 Hz loop would do. The held
        object (if any) is excluded -- fusing it dangling mid-air would
        corrupt its belief."""
        held = self._held_det_label if self.held_object else None
        for _ in range(max(int(frames), 1)):
            try:
                frame = self.observe()
                dets = self.detector.detect(frame, classes=self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                self._show_detections(dets)
                self._update_beliefs_from_frame(frame, dets)
                self._last_reobserve_t = time.monotonic()
            except Exception:
                return  # best effort: a camera hiccup must not kill the retry
        for cam in self._other_cams():
            try:
                frame = cam.depth.ensure_depth(cam.stream.get_frame())
                if not frame.has_depth:
                    continue
                dets = self.detector.detect(frame, classes=self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                T = (frame.T_base_cam if frame.T_base_cam is not None
                     else cam.extrinsics.cam_to_base())
                self._update_beliefs_from_frame(frame, dets, T=T)
            except Exception:
                continue


    @staticmethod
    def _fix_from_belief(query: str, belief) -> "ObjectFix":
        """Synthesize an ObjectFix from a belief.

        Preferred: the belief's remembered REAL point cloud (true shape ->
        true jaw width; a banana reads 35mm, not the 78mm of its bbox slab),
        re-centered on the fused position. Fallback: an axis-aligned box at
        the remembered extent (dense enough for both the OBB planner and the
        GraspGen-X backend)."""
        from ..types import ObjectFix

        center = np.asarray(belief.position, dtype=float).reshape(3)
        pts_mem = getattr(belief, "points", None)
        if pts_mem is not None and len(pts_mem) >= 50:
            pts = np.asarray(pts_mem, dtype=float)
            obb_center, extents, axes = oriented_bbox(pts)
            pts = pts + (center - obb_center)
            det = Detection(
                label=belief.label, conf=float(belief.conf),
                bbox=np.zeros(4, dtype=np.float32),
            )
            return ObjectFix(
                label=query, position=center.copy(), points=pts,
                detection=det, extent=extents, axes=axes,
            )
        ext = (np.sort(np.abs(np.asarray(belief.extent, dtype=float)))[::-1]
               if belief.extent is not None else np.array([0.05, 0.05, 0.05]))
        half = np.clip(ext[:3] / 2.0, 0.01, 0.2)
        top_z = float(belief.top_z) if belief.top_z is not None else float(center[2] + half[2])
        bottom_z = top_z - 2 * half[2]
        rng = np.random.default_rng(0)
        pts = []
        for ax in range(3):  # sample the 6 box faces
            for sign in (-1.0, 1.0):
                p = (rng.random((60, 3)) - 0.5) * 2 * half
                p[:, ax] = sign * half[ax]
                pts.append(p)
        pts = np.concatenate(pts)
        pts[:, 2] = np.clip(pts[:, 2] + (top_z + bottom_z) / 2, bottom_z, top_z)
        pts[:, :2] += center[:2]
        det = Detection(
            label=belief.label, conf=float(belief.conf),
            bbox=np.zeros(4, dtype=np.float32),
        )
        return ObjectFix(
            label=query,
            position=np.array([center[0], center[1], (top_z + bottom_z) / 2]),
            points=pts,
            detection=det,
            extent=np.array([2 * half[0], 2 * half[1], 2 * half[2]]),
            axes=np.eye(3),
        )


    def skill_get_observation(self) -> dict:
        frame = self.observe()
        dets = self.detector.detect(frame, classes=self._default_classes)
        self._show_detections(dets)
        objects = self._update_beliefs_from_frame(frame, dets)
        self.memory.add(
            "observation",
            f"saw {[o['label'] for o in objects]} (depth: {frame.depth_source})",
            rgb=frame.rgb,
        )
        # A pure LOOK must not power the motors: a SafeArm that has not
        # been started yet reports standby instead of being poked awake.
        if getattr(self.arm, "connected", True):
            state = self.arm.get_state()
            tcp = self.kin.fk(state.q)[:3, 3]
            wf = self._gripper_width_frac()
            robot = {
                "q_deg": [round(float(np.degrees(x)), 1) for x in state.q],
                "tcp_xyz": [round(float(x), 3) for x in tcp],
                "gripper_open_frac": round(wf, 2) if wf is not None else "unknown",
                "holding": self.held_object,
            }
        else:
            robot = {"status": "standby (motors unpowered until the first motion)",
                     "holding": self.held_object}
        return {
            "objects_visible": objects,
            "objects_remembered": self.beliefs.summary(),
            "robot": robot,
            "depth_source": frame.depth_source,
        }


    def skill_list_objects(self) -> dict:
        return {
            "objects": self.beliefs.summary(),
            "holding": self.held_object,
            "note": "state=remembered means not currently visible; position is last known",
        }


    def skill_localize_object(self, label: str, spatial_hint: str | None = None) -> dict:
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        color = detection_color(frame.rgb, fix.detection)
        # Beliefs live under DETECTOR labels (what the watcher re-fuses);
        # recording the user's query words would create ghost duplicates.
        self.beliefs.update(
            fix.detection.label or fix.label, fix.position, fix.detection.conf,
            extent=fix.extent, top_z=float(fix.points[:, 2].max()), t=frame.t,
            color=color,
        )
        return {
            "label": label,
            "detected_as": fix.detection.label,
            "color": color,
            "position": [round(float(x), 3) for x in fix.position],
            "extent_m": [round(float(x), 3) for x in fix.extent],
            "n_points": int(fix.points.shape[0]),
        }


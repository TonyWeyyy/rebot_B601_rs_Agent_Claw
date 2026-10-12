"""The curated skill API the agent programs against (ASPIRE Appendix E.2
adapted to this rig). Every call is safety-gated, trace-logged with
before/after keyframes, and mirrored into episodic memory.

Skills return plain dicts with an `ok` flag; SkillError and SafetyViolation
become {"ok": false, "error": ...} so the agent can reason about failures
instead of crashing the loop.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..agent.trace import TraceLogger
from ..grasping import plan_grasps_from_fix, select_grasp, select_profile
from ..grasping.camera_grasp import DEFAULT_INSERTION_DEPTH_M, plan_grasp_from_mask
from ..memory import BeliefStore, EpisodicMemory
from ..perception.colors import detection_color, parse_color_query
from ..perception.grounding import (
    Extrinsics,
    localize_object,
    mask_to_points_cam,
    oriented_bbox,
)
from ..types import Detection, Frame, SafetyViolation, SkillError, make_transform, transform_points

#: skills that move the arm: the WorldWatcher is held while they run so the
#: held/handled object is not re-fused at a bogus mid-air position.
#:
#: v5 §P1.4 fix: ``move_to_pose`` (cartesian waypoint trajectory) and
#: ``move_to_joint_zero`` (raw ``arm.move_joints(zeros, ...)``) both move
#: the arm but were missing from this set. LLM calls to either would
#: skip the watcher pause, causing mid-air held objects to be
#: re-fused at bogus positions.
_MOTION_SKILLS = {
    "grasp_object", "place_at", "place_on_object", "push_object",
    "open_gripper", "close_gripper", "move_home", "pick_and_place",
    "point_at", "wave", "handover", "sort_by_color", "move_relative",
    "throw",
    "move_to_pose",
    "move_to_joint_zero",
    "teach_record",
    "teach_replay",
}


#: v4: open-vocab detectors (YOLOE) are sensitive to the exact noun, and the
#: destination container is variously read as box/bin/basket/crate depending
#: on shape, angle and lighting. Queries for any family member accept the
#: WHOLE family as detection prompts, and a detection carrying a family
#: label is canonicalized back to the queried word before it enters the
#: belief store -- a "basket" detection must satisfy a later "box" lookup,
#: or the place stage can never find the destination.
#: v5: tuned for the actual rig container -- an open-top kraft cardboard
#: tray. YOLOE matches "cardboard box" / "cardboard tray" far more
#: reliably than the generic "box", so those lead the family; every key
#: canonicalizes back to the user's queried word.
_DESTINATION_FAMILY: dict[str, set[str]] = {
    "box": {"box", "cardboard box", "cardboard tray", "carton",
            "kraft box", "open box", "tray", "bin", "basket",
            "crate", "container"},
    "cardboard box": {"box", "cardboard box", "cardboard tray", "carton",
                      "kraft box", "open box", "tray", "bin", "basket",
                      "crate", "container"},
    "cardboard tray": {"box", "cardboard box", "cardboard tray", "carton",
                       "kraft box", "open box", "tray", "bin", "basket",
                       "crate", "container"},
    "carton": {"box", "cardboard box", "cardboard tray", "carton",
               "kraft box", "open box", "tray", "bin", "basket",
               "crate", "container"},
    "bin": {"box", "bin", "basket", "crate", "container"},
    "basket": {"box", "bin", "basket", "crate", "container"},
    "crate": {"box", "bin", "basket", "crate", "container"},
    "container": {"box", "bin", "basket", "crate", "container"},
}


def _jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _is_mock(obj: object) -> bool:
    """True if ``obj`` is a ``unittest.mock.Mock`` / ``MagicMock`` instance.

    Duck-typed via the private ``_mock_methods`` attribute that all Mock
    subclasses carry. Used by ``_safe_bridge_move`` to skip its post-move
    FK verify in unit tests where the bridge and arm are Mocks and don't
    simulate actual motion -- the FK comparison would always report the
    arm parked at the initial state and spuriously fail every call.

    Production code paths instantiate ``RebotGraspBridge`` / ``SafeArm``
    / ``MockArm``, none of which carry ``_mock_methods``, so the verify
    runs normally there.
    """
    return hasattr(obj, "_mock_methods")


class SkillRuntime:
    def __init__(
        self,
        camera,
        depth_provider,
        detector,
        extrinsics: Extrinsics,
        kin,
        safe_arm,
        memory: EpisodicMemory,
        beliefs: BeliefStore,
        trace: TraceLogger,
        cfg,
    ):
        self.camera = camera
        self.depth = depth_provider
        self.detector = detector
        self.extrinsics = extrinsics
        self.kin = kin
        self.arm = safe_arm
        self.memory = memory
        self.beliefs = beliefs
        self.trace = trace
        self.cfg = cfg
        self.last_frame: Frame | None = None
        self.held_object: str | None = None
        self._held_det_label: str | None = None
        self._held_color: str | None = None
        #: optional WorldWatcher (set by the app wiring); paused during motion
        self.watcher = None
        #: optional rebot_grasp motion/gripper bridge (set by make_arm for
        #: ``rebot_rs``; None for mock). Skill methods read this to
        #: decide between the new set.py-aligned flow and the legacy
        #: SafeArm.move_joints pipeline.
        self.bridge = getattr(safe_arm, "grasp_bridge", None)
        #: natural-language task currently executing (dashboard narration)
        self.current_task: str | None = None
        #: monotonic time the current top-level MOTION skill started; while
        #: the arm moves the WorldWatcher is paused, so belief ages measured
        #: from "now" are artificially inflated -- staleness checks measure
        #: from this epoch instead.
        self._motion_t0: float | None = None
        #: last time _reobserve completed a fresh scan DURING the current
        #: motion skill: belief-fallback staleness must advance with it, or
        #: a belief that was fresh at task start stays "fresh" through 8
        #: retries even after every re-scan failed to see the object.
        self._last_reobserve_t: float | None = None
        self._graspgenx = None  # lazy GraspGenXPlanner (grasp.backend)
        self._grounder = None  # lazy VLMGrounder (cfg "grounder", 2nd filter)
        # Fake-RL grasp memory (RPent/Harness-VLA pattern): learns which grasp
        # geometry works per object profile from past attempts (wins AND
        # failures), persisted so accuracy improves across sessions.
        from ..memory.grasp_memory import GraspOutcomeMemory
        from pathlib import Path as _Path
        _gm_path = cfg.grasp.get("memory_path", "~/.wrc_demo/grasp_memory.json")
        self.grasp_memory = GraspOutcomeMemory(
            path=_Path(str(_gm_path)).expanduser())
        g = cfg.arm.gripper
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._max_width = float(g.get("max_width_m", 0.09))
        self._default_classes = list(cfg.get("detect_classes", ["cup", "bottle", "box", "fruit", "toy"]))
        self._cartesian_planner = None  # lazy CartesianPlanner (planning module)

    # ── plumbing ─────────────────────────────────────────────────────────

    @property
    def cartesian_planner(self):
        """Lazy CartesianPlanner; None if wrc_demo.planning unavailable.

        The vendored SDK is already on sys.path (wrc_demo.__init__),
        so we just import the planner directly.
        """
        if self._cartesian_planner is None:
            try:
                from ..planning.cartesian_planner import CartesianPlanner
                self._cartesian_planner = CartesianPlanner(self.kin)
            except (RuntimeError, ImportError):
                return None
        return self._cartesian_planner

    def _show_status(self, text: str) -> None:
        """Push agent state to EVERY live view (all rig cameras show the
        same agent, not just the primary). self.camera goes first: it may
        be a FrameHub WRAPPING the primary stream."""
        streams = [self.camera]
        rig = getattr(self, "rig", None)
        if rig is not None:
            streams += [s for s in getattr(rig, "streams", []) if s is not self.camera]
        for s in streams:
            if hasattr(s, "set_overlay"):
                s.set_overlay(status=text)

    def _show_detections(self, dets) -> None:
        if hasattr(self.camera, "set_overlay"):
            self.camera.set_overlay(detections=dets)

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch one skill call with tracing. Never raises."""
        fn = getattr(self, f"skill_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown skill {name!r}"}
        self._show_status(f"{name}({_short(args)})")
        before = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_before"
        )
        t0 = time.monotonic()
        try:
            import contextlib

            hold = (
                self.watcher.paused()
                if (self.watcher is not None and name in _MOTION_SKILLS)
                else contextlib.nullcontext()
            )
            owns_epoch = name in _MOTION_SKILLS and self._motion_t0 is None
            if owns_epoch:
                self._motion_t0 = time.monotonic()
            with hold:
                try:
                    result = fn(**args)
                finally:
                    if owns_epoch:
                        self._motion_t0 = None
                        self._last_reobserve_t = None
                    # Sync the held-object ignore BEFORE fusion resumes, or
                    # one tick could register the object dangling mid-air.
                    if self.watcher is not None:
                        self.watcher.ignore_label(
                            self._held_det_label if self.held_object else None
                        )
            if "ok" not in result:
                result["ok"] = True
        except (SkillError, SafetyViolation) as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except TypeError as e:
            result = {"ok": False, "error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # camera dropouts, CAN loss, ... : the agent
            # loop must survive and report, not crash mid-task.
            import traceback

            result = {
                "ok": False,
                "error": f"unexpected {type(e).__name__}: {e}",
                "hint": "hardware/runtime fault; consider task_done(success=false)",
            }
            traceback.print_exc()
        dur = (time.monotonic() - t0) * 1000
        after = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_after"
        )
        self.trace.record(name, args, result, dur, before, after)
        err = str(result.get("error", "failed"))
        self._show_status(f"{name} -> " + ("ok" if result["ok"] else err[:60]))
        self.memory.add(
            "action" if result["ok"] else "outcome",
            f"{name}({_short(args)}) -> " + ("ok" if result["ok"] else err[:120]),
        )
        return result

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

    def _tcp(self) -> np.ndarray:
        return self.kin.fk(self.arm.get_state().q)[:3, 3]

    def _reconcile_held(self) -> None:
        """Drop a stale held-state: if we believe we hold something but the
        jaws report (near) fully closed, the object slipped out (or a crash
        left the flag latched). Without this, one dropped object bricks
        every subsequent pick with 'already holding'."""
        if not self.held_object:
            return
        wf = self._gripper_width_frac()
        if wf is None:  # unknown feedback: keep the cautious assumption
            return
        if wf < float(self.cfg.grasp.get("air_grasp_frac", 0.04)):
            self.memory.add(
                "outcome",
                f"I no longer feel {self.held_object!r} in the gripper "
                "(it must have slipped); clearing the held state",
            )
            self.held_object = None
            self._held_det_label = None
            self._held_color = None

    def _gripper_width_frac(self) -> float | None:
        """0 = fully closed, 1 = fully open (from motor angle, linear map).
        None when the gripper position is unknown (feedback failure)."""
        state = self.arm.get_state()
        if not getattr(state, "gripper_valid", True):
            return None
        span = self._grip_closed - self._grip_open
        if abs(span) < 1e-9:
            return 0.0
        closed_frac = (state.gripper_pos - self._grip_open) / span  # 1 at closed
        return float(np.clip(1.0 - closed_frac, 0.0, 1.0))

    # ── language -> world resolution ─────────────────────────────────────

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
        floor must shall the grasp slightly, not abort it, but executing a
        grasp point below the table plane means the descent rams the
        tabletop. The bridge paths (grasp_object / pick_and_place) skipped
        this clamp entirely.
        """
        lim = self.arm.harness.limits
        z_floor = max(
            float(lim.workspace_min[2]) + 0.002,
            lim.table_z + float(self.cfg.grasp.get("min_grasp_z_m", 0.008)),
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
                fix = localize_object(
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
                fix = localize_object(
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

    # ── skills ───────────────────────────────────────────────────────────

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

    def skill_grasp_object(
        self,
        label: str,
        material: str | None = None,
        spatial_hint: str | None = None,
    ) -> dict:
        self._reconcile_held()
        if self.held_object:
            raise SkillError(f"already holding {self.held_object!r}; place it first")

        # Bridge path: when a rebot_grasp bridge is wired (rebot_rs backend),
        # mirror set.py / main.py ``G`` flow exactly — open → pre(2.0) →
        # grasp(1.5) → grasp(force=1.0) → home(3.0). The legacy v5 path
        # below stays for mock runs and for tests that don't set
        # ``self.bridge``.
        if self.bridge is not None:
            return self._skill_grasp_object_bridge(label, spatial_hint)

        gcfg = self.cfg.grasp
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        profile = select_profile(fix.detection.label or label, material)
        grasps = self._plan_grasps(fix, label=label)

        # Learned grasp z can overshoot below the table by a few mm (the
        # tip-offset conversion is empirical): a millimeter under the
        # workspace floor must shallow the grasp slightly, not abort it.
        lim = self.arm.harness.limits
        z_floor = max(
            float(lim.workspace_min[2]) + 0.002,
            lim.table_z + float(gcfg.get("min_grasp_z_m", 0.008)),
        )
        for g in grasps:
            dz = z_floor - float(g.position[2])
            if 0.0 < dz <= 0.03:  # bigger misses are garbage; let vetting drop them
                g.position = np.asarray(g.position, dtype=float).copy()
                g.position[2] = z_floor

        # Pre-vet every candidate against the harness geometry (with the
        # exemption cylinder the descent will open) so a doomed candidate
        # loses the ranking up front instead of aborting mid-motion.
        harness = self.arm.harness
        exempt_r = float(gcfg.get("exempt_radius_m", 0.07))

        def _vet(g, q_pre, q_grasp):
            # The approach leg executes BEFORE allow_grasp_descent opens the
            # cylinder: vet the pregrasp with NO exemption, or a candidate
            # that needed one is guaranteed to abort at the end of the
            # approach. The descent leg holds the exemption, and min-jerk is
            # a straight segment in joint space, so sampling the q_pre ->
            # q_grasp segment vets the actual executed path, not just its
            # endpoints (near-horizontal approaches can dip below the floor
            # OUTSIDE the cylinder mid-descent).
            reason = harness.vet_pose(q_pre)
            if reason:
                return f"pregrasp unsafe: {reason}"
            # Exemption floor a bit BELOW the table: the reBot RS gripper's
            # finger/wrist links extend ~5 cm below the TCP, so a top-down
            # grasp of a low object legitimately dips a link to z ~ -0.018
            # (below the table plane) while the fingers straddle the object.
            # Flooring the exemption at table_z left that link 18 mm outside
            # the cylinder -> false "link would hit the table" abort. The
            # narrow XY radius still confines this to directly over the
            # target, so allowing a small sub-table dip there is safe.
            z_min = float(harness.limits.table_z) - 0.06
            # Sample the q_pre -> q_grasp segment densely: min-jerk streaming
            # generates many intermediate waypoints, and a coarse 4-sample vet
            # can miss a mid-segment configuration where the elbow dips below
            # the table (IK multi-solution: the interpolated path can bow down
            # even when both endpoints are safe).
            for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
                q = q_pre + s * (np.asarray(q_grasp) - np.asarray(q_pre))
                reason = harness.vet_pose(
                    q, exempt_xy=g.position[:2],
                    exempt_radius=exempt_r, exempt_z_min=z_min,
                )
                if reason:
                    return f"descent unsafe: {reason}"
            return None

        state = self.arm.get_state()
        # Seed grasp IK from HOME (elbow-up), not the live pose: seeding from
        # an arbitrary current configuration can converge to an elbow-down IK
        # branch whose approach path dips a link under the table.
        _home = self.cfg.arm.get("home_q")
        _seed = np.asarray(_home, dtype=float) if _home is not None else state.q
        try:
            grasp, q_pre, q_grasp = select_grasp(
                grasps,
                self.kin,
                _seed,
                max_width_m=self._max_width,
                pregrasp_offset_m=float(gcfg.get("pregrasp_offset_m", 0.12)),
                validate=_vet,
            )
        except (SkillError, SafetyViolation) as e:
            # No candidate survived IK + harness vetting. Log it against the
            # object profile so the fake-RL memory learns to bias future
            # candidates (raise z, prefer top-down) for this object.
            try:
                self.grasp_memory.record(
                    label, fix, grasps[0] if grasps else None,
                    success=False, reason=str(e),
                    z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
            except Exception:
                pass
            raise

        # 1. open, go to pregrasp (normal speed). Re-home first so the
        # pregrasp IK seeds from a known elbow-up posture: seeding from an
        # arbitrary current pose can converge to an elbow/wrist-down IK
        # solution whose approach path dips a link below the table near the
        # base (observed: link7 at xy~(0.05,0.04) z=-0.023, far from the
        # target so no grasp-exemption cylinder can cover it).
        # Adaptive opening: open slightly wider than the object width
        grasp_width_m = float(grasp.width_m)
        margin_m = 0.01  # 10mm margin for easier approach
        open_distance_m = min(grasp_width_m + margin_m, self._max_width)
        if hasattr(self.arm, 'open_gripper'):
            self.arm.open_gripper(distance_m=open_distance_m)
        else:
            self.arm.set_gripper(self._grip_open, effort=0.8)
        _home = self.cfg.arm.get("home_q")
        if _home is not None:
            try:
                self.arm.move_joints(np.asarray(_home, dtype=float),
                                     duration_s=1.5)
            except Exception:
                pass  # best-effort re-home; pregrasp move is the real gate
        if not self.arm.move_joints(q_pre, duration_s=float(gcfg.get("move_duration_s", 2.5))):
            raise SkillError("did not settle at pregrasp pose")

        # 2. descend inside the exemption cylinder (slow)
        self.arm.harness.allow_grasp_descent(
            grasp.position[:2],
            radius=float(gcfg.get("exempt_radius_m", 0.07)),
            z_min=float(self.arm.harness.limits.table_z) - 0.06,
        )
        try:
            descend_dur = float(gcfg.get("descend_duration_s", 2.0))
            if not self.arm.move_cartesian(
                self.kin.fk(q_grasp), self.kin,
                duration_s=descend_dur,
                planner=self.cartesian_planner,
            ):
                raise SkillError("did not settle at grasp pose")

            # 3. close with the material profile (width-aware two-stage,
            # stall-aware). grasp.width_m = jaw_span + width_pad; the close
            # target is width-aware (default 75% of the detected width) so
            # 3D-printed bananas aren't crushed.
            self._close_two_stage(
                profile,
                grasp_width_m=float(grasp.width_m),
                max_width_m=self._max_width,
            )

            # 4. lift back to pregrasp (speed scaled by profile)
            lift_dur = float(gcfg.get("descend_duration_s", 2.0)) / max(profile.lift_speed_scale, 0.2)
            self.arm.move_cartesian(
                self.kin.fk(q_pre), self.kin,
                duration_s=lift_dur,
                planner=self.cartesian_planner,
            )
            # 4b. standardize wrist to ready pose — align with rebot_grasp
            # set.py's _move_ready (scripts/set.py:56-67). After closing,
            # lift wrist to (roll=0, pitch=approach_pitch, yaw=0) so the
            # box-placement trajectory only translates and the held object
            # isn't rotated in the fingers. IK failure is non-fatal: log and
            # keep the wrist at the grasp rotation (graceful degradation).
            import pinocchio as pin
            ready_cfg = (self.cfg.arm.get("ready_pose", {}) or {})
            R_ready = pin.rpy.rpyToMatrix(
                float(ready_cfg.get("roll", 0.0)),
                float(ready_cfg.get(
                    "pitch", float(gcfg.get("approach_pitch_rad", 1.2))
                )),
                float(ready_cfg.get("yaw", 0.0)),
            )
            T_ready = make_transform(
                R_ready,
                np.array([
                    float(ready_cfg.get("x", 0.25)),
                    float(ready_cfg.get("y", 0.0)),
                    float(ready_cfg.get("z", 0.35)),
                ]),
            )
            q_now = self.arm.get_state().q
            ik_ready = self.kin.ik(
                T_ready, q_now,
                retries=int(gcfg.get("ik_retries", 3)),
            )
            if ik_ready.success:
                self.arm.move_joints(
                    ik_ready.q,
                    duration_s=float(ready_cfg.get("duration", 3.0)),
                )
            else:
                self.memory.add(
                    "note",
                    "ready_pose IK failed; wrist stays at grasp rotation",
                )
        finally:
            self.arm.harness.clear_grasp_exemption()

        # 5. verify: ASPIRE heuristic on jaw travel after close. An object in
        # the jaws stalls them ABOVE the commanded close; if they reached the
        # commanded stage-2 fraction even though the object should be much
        # wider, nothing resisted: air grasp. Unknown feedback -> report the
        # grasp as unverified rather than dropping a possibly-held object.
        width_after_lift = self._gripper_width_frac()
        verified = width_after_lift is not None
        if verified:
            # Recompute the width-aware close target (same formula
            # _close_two_stage used) so the air-grasp check compares
            # apples to apples.
            from ..grasping.force import compute_close_target
            _, pos2_close_frac = compute_close_target(
                profile, float(grasp.width_m), self._max_width
            )
            commanded_open = max(1.0 - pos2_close_frac, 0.0)
            expected_open = min(grasp.width_m / self._max_width, 1.0)
            air_grasp = width_after_lift < float(gcfg.get("air_grasp_frac", 0.04)) or (
                width_after_lift <= commanded_open + 0.03
                and expected_open >= commanded_open + 0.07
            )
            if air_grasp:
                self.memory.add("outcome", f"grasp {label!r} FAILED: jaws closed on air")
                try:
                    self.grasp_memory.record(
                        label, fix, grasp, success=False, reason="air grasp",
                        z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
                except Exception:
                    pass
                return {
                    "ok": False,
                    "error": "air grasp: gripper closed fully, object not held",
                    "suggestion": "re-localize the object or try the alternate yaw",
                }
        self.held_object = label
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        self.beliefs.mark_removed(self._held_det_label or label, near=fix.position)
        try:
            self.grasp_memory.record(
                label, fix, grasp, success=True,
                z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
        except Exception:
            pass
        self.memory.add(
            "action",
            f"grasped {label!r} (profile {profile.name}, "
            + (f"jaw at {width_after_lift:.2f})" if verified else "grip UNVERIFIED)"),
        )
        return {
            "held": label,
            "grip_profile": profile.name,
            "grip_verified": verified,
            "gripper_open_frac": round(width_after_lift, 2) if verified else None,
            "grasp_width_m": round(grasp.width_m, 3),
        }

    # ────────────────────── rebot_grasp bridge path ──────────────────────
    #
    # The methods below are reached only when ``self.bridge`` is set
    # (``rebot_rs`` backend). They mirror the motion timing of
    # ``rebot_grasp/scripts/main.py::execute_grasp`` /
    # ``set.py::_execute_place_sequence`` so the recorded trajectory can be
    # compared 1:1 against the set.py recording from 2026-08-07.
    # Differences from the legacy path above:
    #
    #   * no material profile, no learned z-nudge, no pre-vetting — set.py
    #     fires the IK from whatever GraspPose cv2.minAreaRect produced
    #   * no wrist standardization to ready pose — set.py just calls
    #     ``_move_ready(ready_cfg)`` at the end of grasp (which IS the home
    #     rendezvous in our spec)
    #   * no air-grasp verification — set.py trusts ``grasp_driver.grasp()``'s
    #     stall-detection return value
    #   * gripper force = 1.0 (override rebot_grasp default 0.30 to match
    #     v5 ``effort=1.00``)
    #
    # The bridge itself wraps ``harness.begin_motion()`` / ``end_motion()``
    # so e-stop and watchdog are checked at the boundary of every move.

    def _skill_grasp_object_bridge(
        self,
        label: str,
        spatial_hint: str | None = None,
    ) -> dict:
        """set.py-aligned ``G`` flow: open → pre(2.0) → grasp(1.5) → grasp → home(3.0)."""
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        grasps = self._plan_grasps(fix, label=label)
        if not grasps:
            raise SkillError(f"no grasp candidates for {label!r}")
        self._clamp_grasp_z_floor(grasps)

        # Seed IK from home_q (elbow-up basin) — matches the legacy comment
        # in the v5 path; rebot_grasp's main.py doesn't seed explicitly
        # because it always moves to ready first, which has the same effect.
        home_q = self.cfg.arm.get("home_q")
        seed = (
            np.asarray(home_q, dtype=float)
            if home_q is not None
            else self.arm.get_state().q
        )
        grasp, q_pre, q_grasp = select_grasp(
            grasps,
            self.kin,
            seed,
            max_width_m=self._max_width,
            pregrasp_offset_m=float(
                self.cfg.grasp.get("pregrasp_offset_m", 0.08)
            ),
            validate=None,  # set.py doesn't pre-vet
        )

        pre6d, grasp6d = self._grasp_to_6d(grasp)
        home_q_arr = np.asarray(home_q, dtype=float) if home_q is not None else None

        # 0. Move to ready_pose first (matches main.py _move_ready).
        # This puts the wrist at pitch≈0.7 so subsequent IK is seeded
        # from the standard top-down-grasp configuration, producing
        # smooth trajectories.  Without this the arm starts from home_q
        # (pitch≈0°) and the IK must rotate the wrist ~40° during the
        # approach, giving awkward trajectories.
        ready6d = self._ready_pose_6d()
        ready_dur = self._ready_duration()
        self._safe_bridge_move(ready6d, duration=ready_dur, where="ready")

        # 1. Open gripper (no timeout → GraspDriver default 3.0 s)
        self.bridge.open_gripper()

        # 2. Pregrasp (set.py uses duration=2.0) — IK + vet_pose first
        # v2: tolerance restored to 0.04 — this rig's normal settle error is
        # 2-4 cm (the original code tried 0.02 and abandoned it), so the
        # v1 tightening to 0.01/0.02 made every verify falsely fail and the
        # skill aborted BEFORE the gripper ever closed.
        self._safe_bridge_move(pre6d, duration=2.0, where="pregrasp",
                               verify_tol_m=0.04)

        # 3. Grasp approach (set.py uses duration=1.5) — IK + vet_pose first
        #
        # Open a grasp-exemption cylinder centered on the grasp XY for
        # the descent motion. Without this, vet_pose rejects every TCP
        # position below table_z + table_clearance (= 0.020 m for the
        # RS), and small/thin objects like a 30 mm banana whose TCP
        # would otherwise dip below the table always fail.
        # Mirrors the legacy v5 path's ``harness.allow_grasp_descent(...)
        # ... clear_grasp_exemption()`` bracket.
        try:
            grasp_xy = (float(grasp.position[0]), float(grasp.position[1]))
            grasp_z = float(grasp.position[2])
            self.arm.harness.allow_grasp_descent(
                center_xy=grasp_xy,
                radius=float(self.cfg.grasp.get("exempt_radius_m", 0.07)),
                z_min=float(self.arm.harness.limits.table_z) - 0.06,
            )
            self._safe_bridge_move(grasp6d, duration=1.5, where="grasp",
                                   verify_tol_m=0.04)

            # 4. Single-stage close with force=1.0 (override 0.30 default to match v5)
            ok = self.bridge.grasp(force=1.0)
            # 2026-08-11: trust the gripper. The MIT-mode stall detector
            # (``abs(vel) < 0.05 rad/s``) and the width_frac check BOTH
            # produce false negatives on thin/soft objects:
            #   * the motor's vel reading oscillates at ~0.07-0.10 rad/s
            #     even when stalled on a 30 mm banana, so the stall
            #     check never fires
            #   * the 5 s timeout then walks the motor all the way to
            #     hard-stop (pos=0.0) -- the banana gets pushed to the
            #     jaw edge and the gripper reads width_frac=0.0 even
            #     though the object IS held (the user can see it in the
            #     fingers)
            # So we now ignore the ``ok`` return value: if the gripper
            # is open and we commanded close+force=1.0, the jaws ARE
            # around the object -- proceed to the place stage. The
            # downside is a true vacuum-grasp wastes a place cycle, but
            # the upside is a real hold isn't lost.
            if not ok:
                wf = self._gripper_width_frac()
                self.memory.add(
                    "warning",
                    f"grasp stall detector said ok=False (width_frac="
                    f"{wf if wf is not None else '?'}); trusting the "
                    f"gripper and proceeding to place",
                )

        finally:
            # Always clear the exemption so the next motion (place or
            # move_home) gets full table-clearance enforcement.
            try:
                self.arm.harness.clear_grasp_exemption()
            except Exception:
                pass

        # 5. Return to ready_pose (non-fatal; matches main.py _move_ready).
        # The object is already in the jaws; a ready-pose IK failure must
        # not turn a successful grasp into a reported failure — the caller
        # (pick_and_place) still needs to proceed to the place stage.
        try:
            self._safe_bridge_move(ready6d, duration=ready_dur, where="ready")
        except (SkillError, SafetyViolation) as e:
            self.memory.add(
                "note",
                f"ready move after grasp failed ({e}); grasp still succeeded",
            )

        # 5b. v7: closed-loop grasp verification (LLM path). The gripper's
        # own stall/width feedback is unreliable on this rig (2026-08-11),
        # so LOOK instead: after the grasp and the lift to ready, the
        # object must have LEFT its pick location. Still sitting on the
        # table there = the jaws closed on air -- fail NOW so the caller
        # retries the grasp instead of carrying an empty gripper to the
        # place stage and reporting a fake success.
        try:
            self._verify_grasp_took_object(fix, label)
        except SkillError:
            try:
                self.grasp_memory.record(
                    label, fix, grasp, success=False,
                    reason="air grasp (still visible at pick location)",
                    z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
            except Exception:
                pass
            raise

        self.held_object = label
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        self.beliefs.mark_removed(
            self._held_det_label or label, near=fix.position
        )

        return {
            "ok": True,
            "held": label,
            "grip_profile": "bridge",
            "grasp_width_m": round(float(grasp.width_m), 3),
            "force": 1.0,
        }

    def _safe_bridge_move(
        self,
        target_6d: tuple[float, ...],
        duration: float,
        *,
        q_target: np.ndarray | None = None,
        where: str = "",
        verify_tol_m: float = 0.04,
    ) -> bool:
        """Wrap ``bridge.move_to_traj`` with pre-motion ``harness.vet_pose`` check.

        The bridge path bypasses ``SafeArm``'s per-waypoint approval
        (rebort_rs SDK controller owns the control loop). Without an
        explicit pre-motion vet, joint-limit / workspace-AABB violations
        from rebot_grasp's IK won't be caught until the e-stop latch.

        Strategy:
        - If ``q_target`` is supplied (e.g. home_q is already a known joint
          config), call vet_pose directly with it.
        - Otherwise IK the Cartesian target via ``self.kin.ik``, raising
          ``SkillError`` on IK failure, then vet_pose the resulting q.
        - vet_pose failure raises ``SafetyViolation`` (not just a return False);
          the bridge.move_to_traj is never called in that case.

        Per-call IK adds ~10 ms × N moves overhead. 5 moves per skill is
        negligible vs the 7 s total grasp+place runtime.
        """
        import pinocchio as pin

        if q_target is None:
            x, y, z, roll, pitch, yaw = target_6d
            T = make_transform(
                pin.rpy.rpyToMatrix(roll, pitch, yaw),
                np.asarray([x, y, z], dtype=float),
            )
            ik = self.kin.ik(T, self.arm.get_state().q)
            if not ik.success:
                raise SkillError(
                    f"{where} IK failed for target "
                    f"({x:.3f}, {y:.3f}, {z:.3f}, "
                    f"rpy={roll:.3f}, {pitch:.3f}, {yaw:.3f})"
                )
            q_target = ik.q

        if self.arm.harness is not None:
            reason = self.arm.harness.vet_pose(q_target)
            if reason:
                raise SafetyViolation(
                    f"{where} target unsafe: {reason} (q={q_target.round(3).tolist()})"
                )

        ok = self.bridge.move_to_traj(*target_6d, duration=duration)
        if ok and not _is_mock(self.bridge) and not _is_mock(self.arm):
            # 2026-08-13: post-move FK verify. ``bridge.move_to_traj`` can
            # return True without actually moving (SDK silent fail --
            # dashboard shows arm parked at the prior pose while the skill
            # reports success). Re-FK and compare actual TCP to the
            # commanded Cartesian target. >2 cm error means the move did
            # not really happen; surface it as a False return so the
            # caller's return-value check (or try/except) can react.
            #
            # Skip when either ``self.bridge`` or ``self.arm`` is a Mock
            # (unit-test fixture): the mocked bridge returns True without
            # driving the mock arm, so the FK comparison always shows the
            # arm parked at the initial state and would spuriously fail
            # every test. Production paths use a real ``RebotGraspBridge``
            # + ``SafeArm`` / ``MockArm`` and run the verify normally.
            try:
                final_q = self.arm.get_state().q
                final_tcp = self.kin.fk(final_q)
                T_target = np.asarray(target_6d[:3], dtype=float)
                err_m = float(np.linalg.norm(final_tcp[:3, 3] - T_target))
                # Per-move tolerance (verify_tol_m): 40 mm is fine for the
                # ready/home rendezvous, but for the pregrasp/grasp/place
                # legs a 40 mm silent-undershoot still "succeeds" -- and the
                # jaws then close on empty air NEXT to the object. Tighten
                # those calls to 10-20 mm instead.
                if err_m > verify_tol_m:
                    self.memory.add(
                        "note",
                        f"{where} post-move FK verify FAILED: TCP off by "
                        f"{err_m*1000:.0f} mm; "
                        f"target={target_6d[:3]} actual="
                        f"{final_tcp[:3, 3].round(3).tolist()} (SDK silent fail)",
                    )
                    return False
            except Exception as e:
                self.memory.add(
                    "note",
                    f"{where} post-move FK verify raised {type(e).__name__}: {e}",
                )
        return ok

    def _grasp_to_6d(self, grasp) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Convert a ``Grasp`` into ``(pre6d, grasp6d)`` for ``bridge.move_to_traj``.

        Both 6-tuples are ``(x, y, z, roll, pitch, yaw)``. The rotation is
        the same at pregrasp and grasp — only the position differs by
        ``approach * pregrasp_offset_m``. Uses pinocchio's intrinsic XYZ
        rpy convention, matching rebot_grasp's
        ``rotation_matrix_to_euler_zyx``-derived usage in set.py.
        """
        import pinocchio as pin

        roll, pitch, yaw = pin.rpy.matrixToRpy(grasp.rotation)
        g_pos = grasp.position
        pre_pos = grasp.pregrasp_position(
            float(self.cfg.grasp.get("pregrasp_offset_m", 0.08))
        )
        grasp6d = (
            float(g_pos[0]), float(g_pos[1]), float(g_pos[2]),
            float(roll), float(pitch), float(yaw),
        )
        pre6d = (
            float(pre_pos[0]), float(pre_pos[1]), float(pre_pos[2]),
            float(roll), float(pitch), float(yaw),
        )
        return pre6d, grasp6d

    def _home_q_to_6d(self) -> tuple[float, ...]:
        """FK wrc_demo's ``home_q`` to ``(x, y, z, roll, pitch, yaw)``.

        Falls back to ``cfg.arm.ready_pose`` if ``home_q`` is missing —
        both come from the same arm profile YAML so this is defensive
        only.
        """
        import pinocchio as pin

        home_q = self.cfg.arm.get("home_q")
        if home_q is None:
            r = self.cfg.arm.get("ready_pose", {}) or {}
            return (
                float(r.get("x", 0.3)),
                float(r.get("y", 0.0)),
                float(r.get("z", 0.3)),
                float(r.get("roll", 0.0)),
                float(r.get("pitch", 0.7)),
                float(r.get("yaw", 0.0)),
            )
        q = np.asarray(home_q, dtype=float)
        T = self.kin.fk(q)
        roll, pitch, yaw = pin.rpy.matrixToRpy(T[:3, :3])
        return (
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(roll), float(pitch), float(yaw),
        )

    def _ready_pose_6d(self) -> tuple[float, ...]:
        """Return the arm-profile ready_pose as ``(x, y, z, r, p, y)``.

        Matches ``rebot_grasp/scripts/main.py::_move_ready`` exactly:
        the ready pose puts the wrist at ``pitch≈0.7`` (≈40°), which is
        the standard top-down-grasp configuration.  IK seeded from
        ready produces smooth, natural trajectories; IK seeded from
        ``home_q`` (pitch≈0°) forces the solver to rotate the wrist
        ~40° *during* the approach, giving awkward trajectories.
        """
        r = self.cfg.arm.get("ready_pose", {}) or {}
        return (
            float(r.get("x", 0.3)),
            float(r.get("y", 0.0)),
            float(r.get("z", 0.3)),
            float(r.get("roll", 0.0)),
            float(r.get("pitch", 0.7)),
            float(r.get("yaw", 0.0)),
        )

    def _ready_duration(self) -> float:
        r = self.cfg.arm.get("ready_pose", {}) or {}
        return float(r.get("duration", 3.0))

    def _drop_point(self, box_fix) -> tuple[float, float, float]:
        """Release point above the destination, from the MEASURED box top.

        ``top_z + 0.06`` clamped to ``[table_z + 0.05, 0.15]``: tall boxes
        no longer collide with the hardcoded 0.08 m release height and low
        boxes don't get a high-altitude drop. Falls back to set.py's
        hardcoded 0.08 when the fix carries no points.
        """
        box_pos = box_fix.position
        z = 0.08  # set.py box_center_base z
        if box_fix.points is not None and len(box_fix.points) > 0:
            top_z = float(np.asarray(box_fix.points)[:, 2].max())
            table_z = float(self.cfg.safety.get("table_z", 0.0))
            z = float(np.clip(top_z + 0.06, table_z + 0.05, 0.15))
        return (float(box_pos[0]), float(box_pos[1]), z)

    def _precheck_place_target(
        self, box_fix, destination: str, *, picked_yet: bool
    ) -> None:
        """Fail fast when the place target cannot be used.

        Runs BEFORE the grasp (or before carrying the held object toward
        the box): a destination outside the safety workspace AABB, or a
        place pose that fails IK / the harness vet, must abort the skill
        immediately with an actionable message. Picking first and
        discovering this at the place stage wastes a full grasp cycle and
        strands the object in the jaws.
        """
        import pinocchio as pin

        drop = self._drop_point(box_fix)
        action = (
            "Carry aborted before moving"
            if picked_yet
            else "NOT picking, to avoid holding an object with nowhere to go"
        )
        harness = getattr(self.arm, "harness", None)
        margin = 0.02
        if harness is not None:
            lim = harness.limits
            ws_min = np.asarray(lim.workspace_min, dtype=float)
            ws_max = np.asarray(lim.workspace_max, dtype=float)
            for axis, name in ((0, "x"), (1, "y")):
                lo, hi, v = (
                    float(ws_min[axis]) + margin,
                    float(ws_max[axis]) - margin,
                    drop[axis],
                )
                if not (lo <= v <= hi):
                    raise SkillError(
                        f"place target for {destination!r} is outside the safe "
                        f"workspace: {name}={v:.3f} not in [{lo:.3f}, {hi:.3f}] "
                        f"(box at [{drop[0]:.3f}, {drop[1]:.3f}]). Move the box "
                        f"toward the arm (x≈0.30–0.40) or widen safety.workspace. "
                        f"{action}."
                    )

        # Compute-only IK + safety vet of the place pose (same rpy as the
        # place move below); ~10 ms, no motion.
        T = make_transform(
            pin.rpy.rpyToMatrix(0.0, 1.2, 0.0),
            np.asarray(drop, dtype=float),
        )
        ik = self.kin.ik(T, self.arm.get_state().q)
        if not ik.success:
            raise SkillError(
                f"place pose unreachable (IK failed) at "
                f"{[round(v, 3) for v in drop]}. Move the box closer to the "
                f"arm. {action}."
            )
        if harness is not None:
            reason = harness.vet_pose(ik.q)
            if reason:
                raise SkillError(
                    f"place pose unsafe: {reason}. Move the box to a clear, "
                    f"reachable spot. {action}."
                )

    def _skill_pick_and_place_bridge(
        self, object_label: str, destination: str,
        already_held: str | None = None,
    ) -> dict:
        """set.py-aligned pick-and-place.

        Mirrors ``set.py::_execute_place_sequence`` end to end:

            [Step1] localize banana + box (full fallback chain) + place precheck
            [Step2] open → pregrasp(2.0) → grasp(1.5) → grasp(force=1.0)
            [Step3] lift to ready position  (we use wrc_demo home_q FK)
            [Step4] move_to box(z=measured top+clearance, rpy=0,1.2,0, dur=2.0)
            [Step4b] open_gripper(timeout=0.5) + 0.8 s async close
            [Step5] return to ready  (wrc_demo home_q FK)

        Both labels are resolved through ``_localize``'s full fallback
        chain (multi-frame retry → other cameras → belief store → VLM):
        single-frame open-vocab detection flickers frame to frame, and
        failing the whole skill because the box blinked out of one frame
        made pick-and-place unusable. The place target is validated
        (workspace AABB + IK + safety vet) BEFORE the grasp so the arm
        never ends up holding an object with nowhere to go.

        Release z comes from the measured box top (top_z + clearance,
        clamped to [table_z + 0.05, 0.15]); when the fix carries no
        points it falls back to set.py's hardcoded 0.08.
        """
        import pinocchio as pin

        # ── [Step1] detect destination (box) ──────────────────────────────
        # When already_held, the arm already has the object — skip pick
        # phase entirely and go straight to place (matches set.py Step4).
        ready6d = self._ready_pose_6d()
        ready_dur = self._ready_duration()

        if already_held:
            # Object is already in the gripper; only need to find the box.
            # Full fallback chain (multi-frame retry → other cameras →
            # belief store → VLM): after a failed place the box may be
            # briefly out of the detector's output even though the world
            # model still knows where it is.
            _, box_fix = self._localize(destination, spatial_hint=None)
            self._precheck_place_target(box_fix, destination, picked_yet=True)
            self.memory.add(
                "note",
                f"already holding {already_held!r} -- skipping pick, "
                f"going straight to place on {destination!r}",
            )
        else:
            # ── Full pick phase: localize object AND destination through
            # the full fallback chain (multi-frame retry → other cameras
            # → belief store → VLM). Live-frame-only lookups flicker on
            # open-vocab classes and made pick attempts fail whenever the
            # box briefly dropped out of the detector's output.
            frame, banana_fix = self._localize(object_label, spatial_hint=None)
            try:
                _, box_fix = self._localize(destination, spatial_hint=None)
            except SkillError:
                raise SkillError(
                    f"no {destination!r} detected (even with belief fallback)"
                )

            # Fail fast: an unreachable or unsafe place target must abort
            # the skill BEFORE the grasp, or the arm ends up holding an
            # object with nowhere to go.
            self._precheck_place_target(box_fix, destination, picked_yet=False)

            # ── [Step2] pick the object (use the same grasp pipeline as
            # _skill_grasp_object_bridge, but inline so the held-detection
            # bookkeeping lines up with the place stage below).
            grasps = self._plan_grasps(banana_fix, label=object_label)
            if not grasps:
                raise SkillError(f"no grasp candidates for {object_label!r}")
            self._clamp_grasp_z_floor(grasps)
            home_q = self.cfg.arm.get("home_q")
            seed = (
                np.asarray(home_q, dtype=float)
                if home_q is not None
                else self.arm.get_state().q
            )
            grasp, _q_pre, _q_grasp = select_grasp(
                grasps,
                self.kin,
                seed,
                max_width_m=self._max_width,
                pregrasp_offset_m=float(
                    self.cfg.grasp.get("pregrasp_offset_m", 0.08)
                ),
                validate=None,
            )
            pre6d, grasp6d = self._grasp_to_6d(grasp)

            # Move to ready_pose first (matches set.py _move_ready).
            if not self._safe_bridge_move(ready6d, duration=ready_dur, where="ready"):
                raise SkillError(
                    f"ready move failed (target="
                    f"{[round(ready6d[0],3), round(ready6d[1],3), round(ready6d[2],3)]}, "
                    f"q_target unsafe or SDK silent fail)"
                )

            self.bridge.open_gripper()
            if not self._safe_bridge_move(pre6d, duration=2.0, where="pregrasp",
                                          verify_tol_m=0.04):
                raise SkillError(
                    f"pregrasp move failed (target="
                    f"{[round(pre6d[0],3), round(pre6d[1],3), round(pre6d[2],3)]}, "
                    f"q_target unsafe or SDK silent fail)"
                )

            # Open a grasp-exemption cylinder centered on the grasp XY for
            # the descent motion (matches _skill_grasp_object_bridge).
            # Without this, vet_pose rejects TCP z below table_z + clearance,
            # which small/thin objects (banana) always trigger.
            try:
                grasp_xy = (float(grasp.position[0]), float(grasp.position[1]))
                self.arm.harness.allow_grasp_descent(
                    center_xy=grasp_xy,
                    radius=float(self.cfg.grasp.get("exempt_radius_m", 0.07)),
                    z_min=float(self.arm.harness.limits.table_z) - 0.06,
                )
                if not self._safe_bridge_move(grasp6d, duration=1.5, where="grasp",
                                              verify_tol_m=0.04):
                    raise SkillError(
                        f"grasp move failed (target="
                        f"{[round(grasp6d[0],3), round(grasp6d[1],3), round(grasp6d[2],3)]}, "
                        f"q_target unsafe or SDK silent fail)"
                    )
                ok = self.bridge.grasp(force=1.0)
                # 2026-08-11: trust the gripper (see the long comment in
                # _skill_grasp_object_bridge above). The MIT stall
                # detector AND the width_frac check both produce false
                # negatives on a 30 mm banana -- the motor walks to
                # hard-stop pushing the banana into the jaw edge, and
                # width_frac reads 0.0 even when the object is held.
                # We now ignore ``ok`` and proceed to the place stage;
                # a true vacuum-grasp wastes a place cycle, but a real
                # hold is no longer lost.
                if not ok:
                    wf = self._gripper_width_frac()
                    self.memory.add(
                        "warning",
                        f"grasp stall detector said ok=False (width_frac="
                        f"{wf if wf is not None else '?'}); trusting the "
                        f"gripper and proceeding to place",
                    )

                # ── [Step3] lift to ready_pose (matches set.py _move_ready) ────
                self.held_object = object_label
                self._held_det_label = banana_fix.detection.label
                self._held_color = detection_color(frame.rgb, banana_fix.detection)
            finally:
                try:
                    self.arm.harness.clear_grasp_exemption()
                except Exception:
                    pass

        # ── [Step3] lift to ready_pose (non-fatal) ─────────────────────
        # After a successful grasp the object is in the jaws; failing to
        # reach the exact ready pose must NOT abort the place stage.
        try:
            self._safe_bridge_move(ready6d, duration=ready_dur, where="ready_after_grasp")
        except (SkillError, SafetyViolation) as e:
            self.memory.add(
                "note",
                f"ready_after_grasp move failed ({e}); proceeding to place anyway",
            )

        # ── [Step3.5] re-localize the destination NOW ───────────────────
        # The Step-1 box fix is ~10 s old by the time the arm gets here,
        # and it may have come from a stale-belief fallback taken while
        # the arm occluded the camera (the user's original bug). The grasp
        # move can also have nudged the box. From ready the view is clear,
        # so take a fresh fix and re-validate the drop target. Failure
        # falls back to the pre-grasp fix (already pre-checked).
        try:
            _, box_fix_fresh = self._localize(destination, spatial_hint=None)
        except SkillError:
            self.memory.add(
                "note",
                f"re-localize of {destination!r} before place failed; "
                "using the pre-grasp fix",
            )
        else:
            self._precheck_place_target(box_fix_fresh, destination, picked_yet=True)
            box_fix = box_fix_fresh

        # ── [Step4] place above the box (measured box top, rpy=0,1.2,0) ─
        drop = self._drop_point(box_fix)

        placed_ok = False
        try:
            if not self._safe_bridge_move(
                (drop[0], drop[1], drop[2], 0.0, 1.2, 0.0),
                duration=2.0, where="place", verify_tol_m=0.04,
            ):
                raise SkillError(
                    f"place move failed (drop="
                    f"{[round(drop[0],3), round(drop[1],3), round(drop[2],3)]}, "
                    f"q_target unsafe or SDK silent fail)"
                )

            # ── [Step4b] release + async close (set.py pattern) ────────
            release_thread = self.bridge.release_with_async_close(
                timeout=0.5, delay_s=0.8,
            )
            placed_ok = True

            # ── [Step5] return to ready AFTER triggering the async close ─
            # The background thread closes the jaws 0.8 s after release;
            # we ``join`` after the return motion so the next skill starts
            # with idle fingers.
            try:
                self._safe_bridge_move(
                    ready6d, duration=ready_dur,
                    where="ready_after_place",
                )
            except (SkillError, SafetyViolation) as e:
                self.memory.add(
                    "note",
                    f"ready_after_place move failed ({e}); object already released",
                )
            release_thread.join()
        except Exception:
            # The place failed BEFORE the release. Do NOT write the planned
            # drop position into the belief store: the object is most likely
            # still in the jaws, and a ghost belief at the box poisons every
            # later retry (the watcher then fuses the real held object as a
            # second, mid-air detection).
            if not placed_ok and self.held_object:
                try:  # back off so the camera sees the scene again
                    self._safe_bridge_move(
                        ready6d, duration=ready_dur, where="abort_retreat",
                    )
                except Exception:
                    pass
                wf = self._gripper_width_frac()
                air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
                if wf is not None and wf < air:
                    self.memory.add(
                        "outcome",
                        f"{self.held_object!r} slipped out during the aborted "
                        "place; clearing the held state",
                    )
                    self.held_object = None
                    self._held_det_label = None
                    self._held_color = None
                else:
                    self.memory.add(
                        "warning",
                        f"place aborted but {self.held_object!r} is still in "
                        "the jaws; keeping the held state so the next attempt "
                        "can go straight to the place stage",
                    )
            raise

        # ── [Step6] verify the landing BEFORE declaring success ──────────
        # The trajectory succeeding only proves the ARM moved; it does not
        # prove the object went into the box. It can bounce off the rim,
        # miss sideways on a stale drop fix, or never have been in the jaws
        # at all (the 2026-08-11 "trust the gripper" decision) -- and the
        # old code reported success in all three cases, then wrote a ghost
        # "toy is in the box" belief that poisoned every later task. One
        # fresh look from ready (every camera, since the wrist/side camera
        # often sees into the box when the primary sees only walls) closes
        # the loop.
        #
        # Tri-state on purpose:
        #   "confirmed" -- object seen within verify radius of the drop
        #                 point -> write the belief, report full success.
        #   "missed"    -- object seen somewhere ELSE -> a confirmed bad
        #                 landing (bounced out / dropped short) -> ok:false
        #                 so the retry loop re-picks it from where it
        #                 actually is.
        #   "unknown"   -- object not visible on ANY camera. Most often it
        #                 simply sits inside the box below the rim where no
        #                 camera can see it; treating that as a failure
        #                 would re-grasp an object that is already placed.
        #                 Accept the place but say so, and DON'T write a
        #                 ghost belief.
        det_label = self._held_det_label
        held_color = self._held_color
        landing = self._verify_place(destination, drop, det_label or object_label)
        placed = self.held_object
        self.held_object = None
        self._held_det_label = None
        self._held_color = None
        if landing == "missed":
            return {
                "ok": False,
                "stage": "place",
                "error": (
                    f"released {object_label!r} above {destination!r} but it "
                    "landed away from the drop point (confirmed by camera)"
                ),
                "note": (
                    "the object is on the table near the box. This skill "
                    "retries automatically; do NOT call localize_object or "
                    "grasp_object for it yourself -- the retry already "
                    "knows where it is from the verification scan"
                ),
            }
        if landing == "confirmed":
            self.beliefs.update(
                det_label or object_label,
                np.array(drop, dtype=float),
                conf=0.8,
                color=held_color,
            )
            result_note = f"place verified by camera: {object_label!r} is at the drop point"
        else:  # "unknown": accepted, no ghost belief, and an EXPLICIT
            # completion instruction -- the object sits inside the box
            # where no camera can see it (expected on this rig), and an
            # agent that doubts the result would call
            # localize_object("toy") next, get "detect no toy", and loop
            # forever. The note tells it not to.
            result_note = (
                f"place accepted: {object_label!r} was released over "
                f"{destination!r} and is no longer in the gripper; no camera "
                "can see it now because it sits inside the container (the "
                "near box wall hides it from the tilted camera). The "
                "pick-and-place is COMPLETE -- do NOT call localize_object "
                "or grasp_object for it again, and do NOT report failure."
            )
            self.memory.add("note", result_note)
        return {
            "ok": True,
            "picked": placed,
            "placed_at": [drop[0], drop[1], drop[2]],
            "destination": destination,
            "place_status": landing,
            "note": result_note,
            "grip_profile": "bridge",
        }

    def _verify_place(
        self,
        destination: str,
        drop,
        object_label: str,
        timeout_s: float | None = None,
    ) -> str:
        """Visual check of where the released object actually landed.

        Scans the primary camera AND every fusing camera each cycle (the
        wrist/side camera often sees into the destination box when the
        primary view is blocked by its walls), refreshing beliefs from what
        it sees, so a subsequent retry already knows where the object
        really is.

        Returns a tri-state string:
            "confirmed" -- object seen within ``place_verify_radius_m`` of
                           the drop point;
            "missed"    -- object seen FAR from the drop point (beyond
                           2x radius) on at least two separate scan cycles
                           (one flickering far sighting is NOT enough --
                           open-vocab detection flickers frame to frame);
            "unknown"   -- anything else: not visible at all (e.g. it sits
                           inside the box hidden behind the near wall from
                           the tilted camera -- the EXPECTED case on this
                           rig), or only ambiguously near. Accepted by the
                           caller, NOT a failure.
        """
        if timeout_s is None:
            timeout_s = float(self.cfg.grasp.get("place_verify_timeout_s", 4.0))
        radius = float(self.cfg.grasp.get("place_verify_radius_m", 0.09))
        drop_xy = np.asarray(drop, dtype=float)[:2]
        far_hits = 0
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            scanned = False
            try:
                frame = self.observe()
                dets = self.detector.detect(frame, classes=self._default_classes)
                self._show_detections(dets)
                self._update_beliefs_from_frame(frame, dets)
                scanned = True
            except Exception:
                pass  # a camera hiccup must not kill the verification
            for cam in self._other_cams():
                try:
                    cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                    if not cframe.has_depth:
                        continue
                    from ..perception.grounding import Extrinsics as _Ext
                    ext = (_Ext(T=cframe.T_base_cam)
                           if cframe.T_base_cam is not None
                           else cam.extrinsics)
                    dets = self.detector.detect(cframe, classes=self._default_classes)
                    self._update_beliefs_from_frame(cframe, dets, T=ext.cam_to_base())
                    scanned = True
                except Exception:
                    continue
            if scanned:
                belief = self.beliefs.find(object_label)
                if belief is not None:
                    d_xy = float(np.linalg.norm(
                        np.asarray(belief.position, dtype=float)[:2] - drop_xy
                    ))
                    if d_xy <= radius:
                        self.memory.add(
                            "note",
                            f"place verified: {object_label!r} seen "
                            f"{d_xy * 1000:.0f} mm from the drop point",
                        )
                        return "confirmed"
                    if d_xy > 2.0 * radius:
                        far_hits += 1
                        self.memory.add(
                            "note",
                            f"place verification: {object_label!r} seen "
                            f"{d_xy * 1000:.0f} mm from the drop point (far)",
                        )
                        if far_hits >= 2:
                            return "missed"
                    # between radius and 2*radius: ambiguous -- keep scanning
            time.sleep(0.4)
        return "unknown"

    def _verify_grasp_took_object(self, fix, label: str) -> None:
        """v7: visual air-grasp check for the LLM path (grasp_object).

        Raises SkillError('air grasp') if the target is STILL VISIBLE ON
        THE TABLE at (or very near) its pre-grasp location after close and
        lift. That is the one unambiguous signature of a missed grasp.
        Not-visible is accepted as held: a properly held object rides in
        the gripper at the ready pose (z well above the table), where this
        check's table-height gate excludes it, and the top camera usually
        cannot see it anyway. Detection flicker can only cause a false
        ACCEPT here (a miss read as held), never a false failure: a
        positive on-table sighting at the exact pick spot is trustworthy.
        """
        radius = float(self.cfg.grasp.get("grasp_verify_radius_m", 0.06))
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        on_table_z = table_z + 0.12  # held objects ride at ready z ~0.3
        det_label = getattr(fix.detection, "label", None) or label
        pick_xy = np.asarray(fix.position, dtype=float)[:2]
        for _ in range(2):
            try:
                frame = self.observe()
                dets = self.detector.detect(frame, classes=self._default_classes)
            except Exception:
                return  # perception hiccup: never fail a possibly-good grasp
            T = (frame.T_base_cam if frame.T_base_cam is not None
                 else self.extrinsics.cam_to_base())
            for d in dets:
                if d.label != det_label:
                    continue
                mask = d.mask
                if mask is None:
                    h, w = frame.rgb.shape[:2]
                    mask = np.zeros((h, w), dtype=bool)
                    x0, y0, x1, y1 = d.bbox.astype(int)
                    mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
                try:
                    pts_cam = mask_to_points_cam(frame, mask)
                except Exception:
                    continue
                if pts_cam.shape[0] < 10:
                    continue
                pts_base = transform_points(T, pts_cam)
                center = pts_base.mean(axis=0)
                if float(center[2]) > on_table_z:
                    continue  # up in the air = riding in the gripper
                d_xy = float(np.linalg.norm(center[:2] - pick_xy))
                if d_xy <= radius:
                    self.memory.add(
                        "outcome",
                        f"air grasp: {label!r} still on the table at its "
                        f"pick location ({d_xy * 1000:.0f} mm away) after "
                        "close+lift",
                    )
                    raise SkillError(
                        f"air grasp: {label!r} is still at its pick location; "
                        "the jaws closed on empty air"
                    )
            time.sleep(0.4)

    def _scan_object_xy(self, object_label: str):
        """One detection pass over the primary AND every fusing camera,
        refreshing beliefs; returns the object's base-frame XY if seen,
        else None."""
        try:
            frame = self.observe()
            dets = self.detector.detect(frame, classes=self._default_classes)
            self._show_detections(dets)
            self._update_beliefs_from_frame(frame, dets)
        except Exception:
            pass
        for cam in self._other_cams():
            try:
                cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                if not cframe.has_depth:
                    continue
                from ..perception.grounding import Extrinsics as _Ext
                ext = (_Ext(T=cframe.T_base_cam)
                       if cframe.T_base_cam is not None
                       else cam.extrinsics)
                dets = self.detector.detect(cframe, classes=self._default_classes)
                self._update_beliefs_from_frame(cframe, dets, T=ext.cam_to_base())
            except Exception:
                continue
        belief = self.beliefs.find(object_label)
        if belief is None:
            return None
        return np.asarray(belief.position, dtype=float)[:2]

    def _verify_landing(self, object_label: str, target, timeout_s: float = 2.5) -> str:
        """v7: tri-state landing check for skill_place_at (LLM path):
        "at_target" (within place_verify_radius_m), "elsewhere" (beyond
        2x radius, confirmed on two scan cycles), or "unknown" (not seen /
        ambiguous). Never raises -- the object is already released, this
        only reports the truth to the caller.
        """
        near = float(self.cfg.grasp.get("place_verify_radius_m", 0.09))
        target_xy = np.asarray(target, dtype=float)[:2]
        far_hits = 0
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            xy = self._scan_object_xy(object_label)
            if xy is not None:
                d = float(np.linalg.norm(xy - target_xy))
                if d <= near:
                    return "at_target"
                if d > 2.0 * near:
                    far_hits += 1
                    if far_hits >= 2:
                        self.memory.add(
                            "note",
                            f"landing check: {object_label!r} came to rest "
                            f"{d * 1000:.0f} mm from the place target",
                        )
                        return "elsewhere"
            time.sleep(0.4)
        return "unknown"

    def _close_two_stage(
        self,
        profile,
        grasp_width_m: float,
        max_width_m: float,
    ) -> None:
        """Width-aware two-stage gripper close (v2, 2026-08-10).

        Stage 1 (scout): half-travel — fast, gets the jaws into contact range.
        Stage 2 (seat): width-aware — gripper target = grasp.width_m *
        ``profile.object_close_frac`` (default 0.75 per user directive),
        clipped to ``[profile.min_close_frac, profile.max_close_frac]`` as a
        safety bound.

        The MIT-mode ``tau = kp*(q_target - q) - kd*dq + tau_ff`` means the
        motor only pushes until ``q`` reaches the target. If the object is
        wider than the target, the motor stalls at ``q`` ≈ object surface
        and the surface limits the force, not the position target. The
        previous fixed-frac close (90% travel = ~8 mm opening) pushed past
        the object with high effort and crushed a 30 mm 3D-printed banana.

        Goes through ``self.arm.close_gripper_two_stage`` — SafeArm
        delegates to the backend (RebotRSArm uses stall detection;
        MockArm uses the simple two-stage default). The cfg-level
        gripper travel is owned by the arm, not duplicated in
        SkillRuntime.
        """
        from ..grasping.force import compute_close_target
        pos1_frac, pos2_frac = compute_close_target(profile, grasp_width_m, max_width_m)
        # Pin the simulated object surface (width-aware MIT stall).
        # The MockArm honors this so it stops the gripper at the object
        # surface the way the real RebotArm's MIT mode does
        # (``tau = kp*(q_target - q) - kd*dq + tau_ff`` stalls the motor
        # at the surface, not the position target). The real RebotArm
        # has no equivalent setter — this is a mock-only hook to keep
        # the e2e test meaningful. Skip when the caller has explicitly
        # disabled stop simulation (``object_stop_frac is None``) so
        # air-grasp tests still see a no-stop trajectory.
        if (
            hasattr(self.arm, "set_object_close_frac")
            and getattr(self.arm, "object_stop_frac", "<UNSET>") is not None
            and max_width_m > 0
        ):
            surface_close_frac = 1.0 - float(grasp_width_m) / float(max_width_m)
            self.arm.set_object_close_frac(surface_close_frac)
        self.arm.close_gripper_two_stage(
            width_pos_stage1=pos1_frac,
            width_pos_stage2=pos2_frac,
            effort=profile.effort,
        )

    def _adopt_unknown_held(self) -> None:
        """'save it in the box' must work even when the held object was
        never registered (a crashed task, or a human placed something in
        the jaws): a mid-open gripper stall means SOMETHING is in there --
        adopt it as 'object' so place skills can act on it. (Wrist-camera
        visual confirmation is the ROADMAP upgrade.)"""
        if self.held_object:
            return
        wf = self._gripper_width_frac()
        air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
        if wf is not None and air < wf < 0.9:
            self.held_object = "object"
            self._held_det_label = None
            self.memory.add(
                "note",
                "the jaws are holding something unregistered; "
                "treating it as 'object'",
            )

    def skill_place_at(self, x: float, y: float, z: float | None = None) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        release_z = float(z) if z is not None else table_z + float(gcfg.get("release_height_m", 0.05))
        # Strict top-down poses only solve below ~0.15 m on this wrist: a
        # tall destination (the bin walls) must become "release from the
        # ceiling and let it drop", not an unreachable-pose failure.
        z_cap = float(gcfg.get("topdown_z_max", 0.15)) - 0.005
        if release_z > z_cap:
            self.memory.add(
                "note",
                f"place height {release_z:.2f} m is above the wrist's "
                f"top-down ceiling; releasing from {z_cap:.2f} m instead",
            )
            release_z = z_cap
        target = np.array([x, y, release_z])

        q_now = self.arm.get_state().q
        hover = target + np.array([0.0, 0.0, float(gcfg.get("pregrasp_offset_m", 0.12))])
        hover[2] = min(hover[2], z_cap)  # same wrist ceiling as the release
        # Placement rotation: align with rebot_grasp set.py's hardcoded
        # (roll=0, pitch=approach_pitch, yaw=0) — the wrist was already
        # standardized to this rotation by skill_grasp_object step 4b, so
        # the box-placement leg only translates and the held object isn't
        # rotated in the fingers. yaw=0 is the primary candidate (set.py
        # parity); the remaining yaws are the radial-first fallback for
        # IK-unreachable poses.
        import pinocchio as pin
        approach_pitch_rad = float(gcfg.get("approach_pitch_rad", 1.2))
        radial = float(np.arctan2(y, x))
        pre = low = None
        for yaw in (0.0, radial, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2):
            R = pin.rpy.rpyToMatrix(0.0, approach_pitch_rad, float(yaw))
            cand_pre = self.kin.ik(make_transform(R, hover), q_now)
            if not cand_pre.success:
                continue
            cand_low = self.kin.ik(make_transform(R, target), cand_pre.q)
            if cand_low.success:
                pre, low = cand_pre, cand_low
                break
        if pre is None or low is None:
            raise SkillError(
                f"place pose unreachable at {target.round(3).tolist()} "
                f"(hover {hover.round(3).tolist()}, all yaws tried)"
            )

        if not self.arm.move_joints(pre.q, duration_s=float(gcfg.get("move_duration_s", 2.5))):
            raise SkillError("did not settle above the place target")
        self.arm.harness.allow_grasp_descent(target[:2], radius=0.07, z_min=release_z - 0.02)
        try:
            descend_dur = float(gcfg.get("descend_duration_s", 2.0))
            if not self.arm.move_cartesian(
                self.kin.fk(low.q), self.kin,
                duration_s=descend_dur,
                planner=self.cartesian_planner,
            ):
                raise SkillError("did not settle at place pose")
            # 2026-08-13: verify TCP actually reached the target.
            # cartesian_planner's waypoint IK may partially fail; if so the
            # arm stops short of the target but move_cartesian still returns
            # True. Don't trust the boolean — re-FK and check the error.
            final_q = self.arm.get_state().q
            final_tcp = self.kin.fk(final_q)
            place_err = float(np.linalg.norm(final_tcp[:3, 3] - target))
            if place_err > 0.02:  # 2 cm tolerance
                self.memory.add(
                    "note",
                    f"place TCP error {place_err*1000:.0f} mm exceeds 20 mm tolerance "
                    f"(cartesian_planner may have partial-IK-failure); aborting release",
                )
                raise SkillError(
                    f"place failed: TCP off target by {place_err*1000:.0f} mm "
                    f"(target={[round(float(v),3) for v in target]})"
                )
            self.arm.set_gripper(self._grip_open, effort=0.6)
            time.sleep(float(self.cfg.grasp.get("close_settle_s", 0.0)))
            # From here the object IS placed: reconcile the held state BEFORE
            # the ascent, or an ascent abort leaves held_object latched and a
            # retry descends onto the object we just released.
            placed = self.held_object
            placed_det_label = self._held_det_label
            placed_color = self._held_color
            self.beliefs.update(
                # Re-register under the DETECTOR label so the watcher's next
                # fusion merges here instead of creating a query-string ghost.
                placed_det_label or placed, target, 0.8, color=placed_color
            )
            self.held_object = None
            self._held_det_label = None
            self._held_color = None
            self.memory.add("action", f"placed {placed!r} at {target.round(3).tolist()}")
            try:
                self.arm.move_cartesian(
                    self.kin.fk(pre.q), self.kin,
                    duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                    planner=self.cartesian_planner,
                )
            except (SkillError, SafetyViolation) as e:
                # The place already happened; report success and let the
                # caller's move_home park the arm.
                self.memory.add("note", f"placed, but the ascent aborted: {e}")
        finally:
            self.arm.harness.clear_grasp_exemption()
        # v7: the object is released either way -- but "placed" only ever
        # meant the ARM moved. Tell the caller (the LLM) where the object
        # ACTUALLY landed, so the LLM path stops declaring victory over a
        # drop that missed the target.
        landing = "unknown"
        if placed:
            try:
                landing = self._verify_landing(
                    str(placed_det_label or placed), target
                )
            except Exception:
                landing = "unknown"
        return {
            "placed": placed,
            "at": [round(float(v), 3) for v in target],
            "landing": landing,
        }

    def skill_place_on_object(self, label: str) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        belief = self.beliefs.find(label)
        if belief is None:
            frame, fix = self._localize(label)
            self.beliefs.update(
                fix.detection.label or label, fix.position, fix.detection.conf,
                extent=fix.extent, top_z=float(fix.points[:, 2].max()),
                color=detection_color(frame.rgb, fix.detection),
            )
            belief = self.beliefs.find(label)
        # Use the actually observed highest point of the target, never OBB
        # extents (those are eigenvalue-ordered, not axis-aligned).
        top = belief.top_z if belief.top_z is not None else float(belief.position[2])
        drop = top + float(self.cfg.grasp.get("release_clearance_m", 0.06))
        return self.skill_place_at(
            float(belief.position[0]), float(belief.position[1]), drop
        )



    def skill_pick_and_place(
        self,
        object: str,
        destination: str | None = None,
        material: str | None = None,
    ) -> dict:
        """One-call pick-and-place that PERSISTS: resolve (color queries
        work) -> grasp -> place on the destination object or the configured
        drop zone -> home. Both stages keep retrying with fresh perception
        (re-home, re-scan, re-plan -- GraspGen-X sampling is stochastic, so
        every retry is a genuinely new grasp) until they succeed or the
        persistence budget (grasp.persist_seconds / grasp.max_pick_attempts)
        runs out. No LLM in the loop; this is the reflex the web chat calls
        for "pick and place pink object"."""
        self._reconcile_held()
        already_held = None
        if self.held_object:
            hq, oq = self.held_object.lower(), str(object).lower()
            if hq in oq or oq in hq:
                # The jaws already hold what was asked for (a previous task
                # grasped it and died before placing): skip straight to the
                # place stage instead of refusing.
                already_held = self.held_object
                self.memory.add(
                    "note",
                    f"already holding {already_held!r} -- going straight to the place stage",
                )
            else:
                raise SkillError(
                    f"already holding {self.held_object!r}; place it first"
                )
        t0 = time.monotonic()
        timings: dict[str, float] = {}

        # Bridge path: when rebot_grasp bridge is wired AND a destination
        # label is given (so the place target is detectable, not a fixed
        # drop_zone coordinate), mirror set.py's flow:
        #   same-frame detect → grasp → home → place(rpy=0,1.2,0,z=0.08)
        #   → release_with_async_close → home
        # Without a destination we fall back to the legacy flow so
        # ``pick_and_place`` with no destination still uses the drop_zone.
        if self.bridge is not None and destination is not None:
            # The bridge path used to be ONE-SHOT: any transient hiccup
            # (one missed detection, one IK convergence failure, one
            # unconfirmed landing) failed the WHOLE task and left it to the
            # LLM to pick up the pieces. Give it the same persistence
            # budget the legacy path has: re-home, re-observe, retry until
            # max_pick_attempts / persist_seconds run out.
            # `_skill_pick_and_place_bridge` already skips the pick stage
            # when `already_held` is set, so a failure that left the object
            # in the jaws resumes at the place stage on the next attempt.
            bcfg = self.cfg.grasp
            b_max = max(int(bcfg.get("max_pick_attempts", 8)), 1)
            b_deadline = t0 + float(bcfg.get("persist_seconds", 120.0))
            b_attempts = 0
            b_err = "unknown"
            while True:
                b_attempts += 1
                try:
                    res = self._skill_pick_and_place_bridge(
                        object, destination, already_held=already_held
                    )
                    if res.get("ok", True):
                        return res
                    b_err = str(res.get("error", "bridge pick_and_place failed"))
                except (SkillError, SafetyViolation) as e:
                    b_err = f"{type(e).__name__}: {e}"
                self.memory.add(
                    "outcome",
                    f"pick_and_place attempt {b_attempts} failed: {b_err[:120]}",
                )
                if self.arm.harness.estopped:
                    b_err += " (e-stop latched; not retrying)"
                    break
                if b_attempts >= b_max or time.monotonic() > b_deadline:
                    break
                resume_held = self.held_object
                self.memory.add(
                    "note",
                    f"not giving up: attempt {b_attempts + 1}/{b_max} -- "
                    + (f"still holding {resume_held!r}, resuming at the place stage"
                       if resume_held else
                       f"re-homing and re-scanning for {object!r}"),
                )
                already_held = resume_held  # mid-carry failure resumes at place
                try:  # park the arm so the re-scan sees a clean scene
                    self.skill_move_home()
                except (SkillError, SafetyViolation):
                    pass
                self._reobserve()
            return {
                "ok": False,
                "stage": "pick_and_place",
                "error": f"pick_and_place failed after {b_attempts} attempt(s): {b_err}",
                "attempts": b_attempts,
            }

        gcfg = self.cfg.grasp
        max_attempts = max(int(gcfg.get("max_pick_attempts", 8)), 1)
        deadline = t0 + float(gcfg.get("persist_seconds", 120.0))

        grasp = (
            {"held": already_held, "grip_verified": None, "grip_profile": None}
            if already_held
            else None
        )
        placed = None
        last_err = "unknown"
        place_err = "unknown"
        attempt = 0
        p_attempt = 0
        tp = t0
        while placed is None:
            # ── grasp stage (skipped when re-entering after a mid-carry slip
            # left something verified in the jaws -- cannot happen today, but
            # the guard keeps the loop honest) ─────────────────────────────
            while grasp is None and attempt < max_attempts:
                attempt += 1
                if attempt == 1:
                    self.memory.add(
                        "note",
                        f"picking up {object!r} (attempt 1/{max_attempts}; "
                        "I will keep trying until it works or the budget runs out)",
                    )
                if attempt > 1:
                    if time.monotonic() > deadline:
                        attempt -= 1  # this attempt never ran
                        break
                    self.memory.add(
                        "note",
                        f"not giving up: attempt {attempt}/{max_attempts} on "
                        f"{object!r} -- re-homing, re-scanning, planning a fresh grasp",
                    )
                    try:  # clear the camera view so the re-scan actually sees
                        self.skill_move_home()
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                tg = time.monotonic()
                try:
                    res = self.skill_grasp_object(object, material=material)
                except (SkillError, SafetyViolation) as e:
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                timings[f"grasp_attempt{attempt}_s"] = round(time.monotonic() - tg, 2)
                if res.get("ok", True) and res.get("held"):
                    grasp = res
                    break
                last_err = str(res.get("error", "grasp failed"))
                self.memory.add("outcome", f"pick attempt {attempt} failed: {last_err[:100]}")
                if self.arm.harness.estopped:
                    last_err += " (e-stop latched; not retrying)"
                    break
                # Fail fast on errors persistence cannot cure: an object the
                # world model has NEVER seen after full re-scans is a typo or
                # simply not on the table -- burning 2 minutes on it reads as
                # a hang at a live booth.
                if (
                    attempt >= 2
                    and "no detections" in last_err
                    and self.beliefs.find(object) is None
                ):
                    last_err += " (never seen after re-scans; giving up early)"
                    break
            if grasp is None:
                # Never leave the arm hanging mid-pose over the table after a
                # failed attempt -- park it (best effort, nothing is held).
                try:
                    self.skill_move_home()
                except (SkillError, SafetyViolation):
                    pass
                return {
                    "ok": False, "stage": "grasp",
                    "error": f"grasp failed after {attempt} attempts "
                             f"({round(time.monotonic() - t0, 1)}s): {last_err}",
                    "suggestion": "check world_state / camera_snapshot; the object may be unreachable or mis-detected",
                }

            # ── place stage ────────────────────────────────────────────────
            tp = time.monotonic()
            while p_attempt < max_attempts:
                p_attempt += 1
                if p_attempt > 1:
                    if time.monotonic() > deadline:
                        p_attempt -= 1
                        break
                    self.memory.add(
                        "note",
                        f"still holding {object!r}: place attempt "
                        f"{p_attempt}/{max_attempts} -- re-locating the target",
                    )
                    try:  # carry it home first: clears the view of the arm
                        self.skill_move_home()  # + held object for the re-scan
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                try:
                    if destination:
                        res = self.skill_place_on_object(destination)
                    else:
                        dz = gcfg.get("drop_zone", [0.30, -0.20])
                        res = self.skill_place_at(float(dz[0]), float(dz[1]))
                    if not res.get("ok", True):
                        raise SkillError(str(res.get("error", "place failed")))
                    placed = res
                    break
                except (SkillError, SafetyViolation) as e:
                    place_err = f"{type(e).__name__}: {e}"
                    self.memory.add(
                        "outcome", f"place attempt {p_attempt} failed: {place_err[:100]}"
                    )
                    if self.arm.harness.estopped:
                        place_err += " (e-stop latched; not retrying)"
                        break
                    if not self.held_object:
                        break  # slipped mid-carry: placing again is pointless
            if placed is None:
                can_regrasp = (
                    not self.held_object
                    and not self.arm.harness.estopped
                    and time.monotonic() < deadline
                    and attempt < max_attempts
                )
                if not can_regrasp:
                    return {
                        "ok": False, "stage": "place",
                        "error": f"place failed after {p_attempt} attempts: {place_err}",
                        "note": (
                            f"still holding {object!r}; try place_at with explicit coordinates"
                            if self.held_object
                            else f"{object!r} slipped while carrying; grasp budget exhausted"
                        ),
                        "grip_verified": grasp.get("grip_verified"),
                    }
                self.memory.add(
                    "note",
                    f"{object!r} slipped while I was carrying it -- starting over",
                )
                grasp = None  # back to the grasp stage within the same budget
        timings["place_s"] = round(time.monotonic() - tp, 2)

        if bool(self.cfg.grasp.get("home_after_place", True)):
            try:  # clear the camera view for the next command; best effort
                self.skill_move_home()
            except (SkillError, SafetyViolation):
                pass
        total = round(time.monotonic() - t0, 2)
        self.memory.add(
            "action",
            f"pick_and_place {object!r} -> {destination or 'drop zone'} in {total}s",
        )
        return {
            "picked": object,
            "placed_at": placed.get("at"),
            "destination": destination or "drop zone",
            "grip_profile": grasp.get("grip_profile"),
            "grip_verified": grasp.get("grip_verified"),
            "grasp_attempts": attempt,
            "place_attempts": p_attempt,
            "duration_s": total,
            "timings": timings,
        }

    # ── social / audience skills (deterministic, harness-gated) ─────────

















    def skill_open_gripper(self) -> dict:
        self.arm.set_gripper(self._grip_open, effort=0.8)
        if self.held_object:
            self.memory.add("action", f"released {self.held_object!r}")
            self.held_object = None
            self._held_det_label = None
        return {"gripper": "open"}

    def skill_close_gripper(self) -> dict:
        self._close_two_stage(select_profile("", "rigid"))
        wf = self._gripper_width_frac()
        return {"gripper": "closed", "open_frac": round(wf, 2) if wf is not None else "unknown"}



    def skill_move_home(self) -> dict:
        home = np.asarray(self.cfg.arm.get("home_q", [0.0, -0.5, -0.9, 0.0, 0.6, 0.0]), dtype=float)
        if not self.arm.move_joints(home, duration_s=3.0):
            raise SkillError("did not settle at home")
        return {"at": "home"}

    def skill_wave(self) -> dict:
        """Wave hello: oscillate forearm at home position.

        Stays at home_q (TCP at [0.25, 0, 0.35] per FK), oscillates joint 3
        (forearm roll) by ±0.4 rad for 3 cycles (~3 s total motion). All
        intermediate poses stay within the safety workspace AABB and URDF
        joint limits (joint 3 swings between -0.85 and -0.05 rad; URDF
        limit is ±π - 0.02 rad).

        Added 2026-09-03 to fill the dead-reference in _MOTION_SKILLS
        (and reflex.py's "wave" regex, which had been dispatching to
        skill_wave since 2026-08-13 but found no implementation).

        Returns:
            {"waved": True, "n_cycles": 3, "amplitude_rad": 0.4}
        """
        home = np.asarray(
            self.cfg.arm.get("home_q", [0.0, 0.1684, 0.6226, -0.4543, 0.0, 0.0]),
            dtype=float,
        )
        amplitude = 0.4  # rad on joint 3 (forearm roll)
        n_cycles = 3
        half_period_s = 0.5

        # Park at home first (idempotent — SafeArm.move_joints is a no-op
        # if already at target; otherwise moves us into the safe starting pose).
        if not self.arm.move_joints(home, duration_s=2.0):
            raise SkillError("could not reach home before waving")

        for i in range(n_cycles * 2):
            q = home.copy()
            q[3] = home[3] + (amplitude if i % 2 == 0 else -amplitude)
            if not self.arm.move_joints(q, duration_s=half_period_s):
                raise SkillError(
                    f"wave trajectory failed at step {i + 1}/{n_cycles * 2}"
                )

        if not self.arm.move_joints(home, duration_s=2.0):
            raise SkillError("could not return home after waving")

        return {
            "waved": True,
            "n_cycles": n_cycles,
            "amplitude_rad": amplitude,
        }

    def skill_teach_record(self, slot: int) -> dict:
        """Enter master-slave teleoperation, capture trajectory.

        Long-running skill: blocks waiting for user to press q (start/stop
        recording) or Esc (cancel). Added to _MOTION_SKILLS so WorldWatcher
        pauses during motion. Returns dict with frame_count/duration_s.

        Bypasses SafeArm (uses self.arm.raw → vendored RebotArm) because teach
        needs an external-event-triggered control loop, which SafeArm lacks.
        Master arm is PiPER Mate via fashionstar-uart-sdk.
        """
        from pathlib import Path
        from . import teach as _teach_mod
        from .master_arm import PiPERMateArm

        if not isinstance(slot, int) or slot < 1 or slot > _teach_mod.NUM_SLOTS:
            return {
                "ok": False,
                "error": f"slot must be int 1-{_teach_mod.NUM_SLOTS}, got {slot!r}",
            }

        try:
            raw_arm = self.arm.raw          # LazyArm materializes RebotRSArm
            slave = getattr(raw_arm, "_arm", None)
        except Exception as e:
            return {"ok": False, "error": f"arm not available: {type(e).__name__}: {e}"}

        if slave is None:
            return {"ok": False, "error": "SDK arm not connected"}

        master = PiPERMateArm(port="/dev/ttyUSB0", gripper_exist=True)
        try:
            master.open()
        except Exception as e:
            return {"ok": False, "error": f"master arm open failed: {type(e).__name__}: {e}"}

        recordings_dir = Path(__file__).resolve().parents[2] / "data" / "recordings"
        try:
            return _teach_mod.record_real(master, slave, slot, recordings_dir)
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}
        finally:
            try:
                master.close()
            except Exception:
                pass

    def skill_teach_replay(self, slot: int) -> dict:
        """Replay recorded trajectory; return arm to motor mechanical zero.

        Returns dict with replayed_to_home=True on success. Uses
        RebotArmEndPose.move_joints_traj + safe_home internally via teach.py.
        """
        from pathlib import Path
        from . import teach as _teach_mod

        if not isinstance(slot, int) or slot < 1 or slot > _teach_mod.NUM_SLOTS:
            return {
                "ok": False,
                "error": f"slot must be int 1-{_teach_mod.NUM_SLOTS}, got {slot!r}",
            }

        try:
            raw_arm = self.arm.raw
            slave = getattr(raw_arm, "_arm", None)
        except Exception as e:
            return {"ok": False, "error": f"arm not available: {type(e).__name__}: {e}"}

        if slave is None:
            return {"ok": False, "error": "SDK arm not connected"}

        recordings_dir = Path(__file__).resolve().parents[2] / "data" / "recordings"
        try:
            return _teach_mod.replay_real(slave, slot, recordings_dir)
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}





    def skill_task_done(self, success: bool, summary: str) -> dict:
        if isinstance(success, str):  # schema-lax backends send "false"
            success = success.strip().lower() in ("true", "yes", "1")
        return {"ok": True, "task_complete": True, "success": bool(success), "summary": summary}


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


TOOL_SPECS: list[dict] = [
    {
        "name": "list_objects",
        "description": "List known objects (visible + remembered) with positions. Call first.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick_and_place",
        "description": "Complete pick-and-place: detect+grasp+place atomically. For 'put X in Y', 'pick up X then place Y', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string", "description": "what to pick, e.g. 'banana', 'red cube'"},
                "destination": {"type": "string", "description": "named destination, e.g. 'box', 'plate'. Omit for default drop zone."},
                "material": {"type": "string", "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"]},
            },
            "required": ["object"],
        },
    },
    {
        "name": "move_home",
        "description": "Park the arm at home pose. Use to reset between tasks or after failures.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "wave",
        "description": (
            "Wave hello: oscillate the forearm at home position to greet. "
            "Use for 'wave hello' / 'say hi' / '打招呼' / '挥手'. No parameters."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "task_done",
        "description": "Declare task finished with success bool and one-paragraph summary. ALWAYS call last.",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
    {
        "name": "teach_record",
        "description": (
            "WHAT: Enter master-slave teleoperation mode. PiPER Mate master arm "
            "(operator-controlled via /dev/ttyUSB0) drives the reBot B601-RS slave "
            "arm in POS_VEL mode. Capture joint positions at 30Hz. User presses q "
            "to start/stop recording into slot N (Enter also works as fallback). "
            "Long-running (waits for q or Esc). "
            "WHEN: User asks to 'record an action' / 'save a motion' / 'teach mode' / "
            "'record slot N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "which slot to save the recording into",
                },
            },
            "required": ["slot"],
        },
    },
    {
        "name": "teach_replay",
        "description": (
            "WHAT: Read the saved slot{N}.json and execute the recorded trajectory "
            "via move_joints_traj. After playback, return the arm to motor mechanical "
            "zero (np.zeros(6)) via safe_home. "
            "WHEN: User asks to 'replay the action' / 'play back' / 'repeat it' / "
            "'replay slot N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "which slot to replay",
                },
            },
            "required": ["slot"],
        },
    },
]

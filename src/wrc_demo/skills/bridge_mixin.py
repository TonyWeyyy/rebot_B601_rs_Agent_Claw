"""RebotGraspBridge skill flows and post-motion visual verification mixin."""

from __future__ import annotations

import time

import numpy as np

from ..perception.colors import detection_color
from ..types import SafetyViolation, SkillError, make_transform
from .specs import _is_mock


def _rt():
    from . import runtime as _runtime_mod
    return _runtime_mod


class _BridgeSkillsMixin:
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
        grasp, q_pre, q_grasp = _rt().select_grasp(
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

            # 5. Lift back to ready_pose BEFORE clearing the grasp exemption
            # cylinder so per-waypoint CLIK lifting out of the target zone
            # is not falsely vetoed by minor lateral motion below table_clearance.
            try:
                self._safe_bridge_move(ready6d, duration=ready_dur, where="ready")
            except (SkillError, SafetyViolation) as e:
                self.memory.add(
                    "note",
                    f"ready move after grasp failed ({e}); grasp still succeeded",
                )

        finally:
            # Always clear the exemption so the next motion (place or
            # move_home) gets full table-clearance enforcement.
            try:
                self.arm.harness.clear_grasp_exemption()
            except Exception:
                pass

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
            grasp, _q_pre, _q_grasp = _rt().select_grasp(
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

                # ── [Step3] lift to ready_pose BEFORE clearing the exemption ──
                self.held_object = object_label
                self._held_det_label = banana_fix.detection.label
                self._held_color = detection_color(frame.rgb, banana_fix.detection)
                try:
                    self._safe_bridge_move(ready6d, duration=ready_dur, where="ready_after_grasp")
                except (SkillError, SafetyViolation) as e:
                    self.memory.add(
                        "note",
                        f"ready_after_grasp move failed ({e}); proceeding to place anyway",
                    )
            finally:
                try:
                    self.arm.harness.clear_grasp_exemption()
                except Exception:
                    pass
        if already_held:
            # Already holding an object: ensure we lift to ready before placing.
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
            # _rt().localize_object("toy") next, get "detect no toy", and loop
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
                    bpos = np.asarray(belief.position, dtype=float).reshape(-1)
                    if bpos.size < 2:
                        continue
                    d_xy = float(np.linalg.norm(bpos[:2] - drop_xy))
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


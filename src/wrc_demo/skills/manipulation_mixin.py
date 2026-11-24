"""High-level manipulation, social, and teach skills mixin for SkillRuntime."""

from __future__ import annotations

import time

import numpy as np

from ..grasping import select_profile
from ..perception.colors import detection_color
from ..types import SafetyViolation, SkillError, make_transform


def _rt():
    from . import runtime as _runtime_mod
    return _runtime_mod


class _ManipulationMixin:
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
            grasp, q_pre, q_grasp = _rt().select_grasp(
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


    def _resolve_teach_handles(self):
        """Resolve ``(slave_arm, controller, gripper)`` for teach skills.

        Supports both production ``SafeArm`` (which stores ``_arm`` / ``_ctrl``
        / ``_gripper`` directly without a ``.raw`` attribute) and legacy/mock
        ``arm.raw._arm`` doubles used by unit tests.
        """
        raw_arm = getattr(self.arm, "raw", None)
        if raw_arm is not None:
            slave = getattr(raw_arm, "_arm", None)
        else:
            slave = getattr(self.arm, "_arm", None)
        controller = getattr(self.arm, "_ctrl", None)
        gripper = getattr(self.arm, "_gripper", None)
        return slave, controller, gripper


    def skill_teach_record(self, slot: int) -> dict:
        """Enter master-slave teleoperation, capture trajectory.

        Long-running skill: blocks waiting for user to press q (start/stop
        recording) or Esc (cancel). Added to _MOTION_SKILLS so WorldWatcher
        pauses during motion. Returns dict with frame_count/duration_s.
        """
        from ..config import PACKAGE_ROOT
        from . import teach as _teach_mod
        from .master_arm import PiPERMateArm

        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 1 or slot > _teach_mod.NUM_SLOTS:
            return {
                "ok": False,
                "error": f"slot must be int 1-{_teach_mod.NUM_SLOTS}, got {slot!r}",
            }

        try:
            slave, controller, gripper = self._resolve_teach_handles()
        except Exception as e:
            return {"ok": False, "error": f"arm not available: {type(e).__name__}: {e}"}

        if slave is None:
            return {"ok": False, "error": "SDK arm not connected"}

        master = PiPERMateArm(port="/dev/ttyUSB0", gripper_exist=True)
        try:
            master.open()
        except Exception as e:
            return {"ok": False, "error": f"master arm open failed: {type(e).__name__}: {e}"}

        recordings_dir = PACKAGE_ROOT / "data" / "recordings"
        try:
            return _teach_mod.record_real(
                master, slave, slot, recordings_dir,
                controller=controller, gripper=gripper,
            )
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}
        finally:
            try:
                master.close()
            except Exception:
                pass


    def skill_teach_replay(self, slot: int) -> dict:
        """Replay recorded trajectory safely and return to home pose.

        When called on a live ``SafeArm`` with an active 500 Hz MIT + gravity
        feed-forward loop, replays through ``SafeArm`` (safety-gated, zero
        control-loop tear-down, zero torque-off / arm drop).
        """
        from ..config import PACKAGE_ROOT
        from . import teach as _teach_mod

        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 1 or slot > _teach_mod.NUM_SLOTS:
            return {
                "ok": False,
                "error": f"slot must be int 1-{_teach_mod.NUM_SLOTS}, got {slot!r}",
            }

        try:
            slave, controller, gripper = self._resolve_teach_handles()
        except Exception as e:
            return {"ok": False, "error": f"arm not available: {type(e).__name__}: {e}"}

        if slave is None:
            return {"ok": False, "error": "SDK arm not connected"}

        recordings_dir = PACKAGE_ROOT / "data" / "recordings"
        try:
            return _teach_mod.replay_real(
                slave, slot, recordings_dir,
                safe_arm=self.arm, controller=controller, gripper=gripper,
            )
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}


    def skill_task_done(self, success: bool, summary: str) -> dict:
        if isinstance(success, str):  # schema-lax backends send "false"
            success = success.strip().lower() in ("true", "yes", "1")
        return {"ok": True, "task_complete": True, "success": bool(success), "summary": summary}


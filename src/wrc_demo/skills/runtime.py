"""The curated skill API the agent programs against (ASPIRE Appendix E.2
adapted to this rig). Every call is safety-gated, trace-logged with
before/after keyframes, and mirrored into episodic memory.

Skills return plain dicts with an ``ok`` flag; ``SkillError`` and
``SafetyViolation`` become ``{"ok": false, "error": ...}`` so the agent can
reason about failures instead of crashing the loop.

Modular layout:
- ``skills/specs.py``: ``TOOL_SPECS``, ``_MOTION_SKILLS``, ``_DESTINATION_FAMILY``
- ``skills/perception_mixin.py``: observation, localization, VLM grounding, grasp planning
- ``skills/bridge_mixin.py``: RebotGraspBridge grasp/place flows & visual landing verification
- ``skills/manipulation_mixin.py``: grasp, place, pick_and_place, wave, teach, task_done
- ``skills/runtime.py``: ``SkillRuntime`` composition root & ``execute()`` choke point
"""

from __future__ import annotations

import time

import numpy as np

from ..agent.trace import TraceLogger
from ..grasping import plan_grasps_from_fix, select_grasp, select_profile  # noqa: F401
from ..grasping.camera_grasp import DEFAULT_INSERTION_DEPTH_M, plan_grasp_from_mask  # noqa: F401
from ..memory import BeliefStore, EpisodicMemory
from ..perception.grounding import Extrinsics, localize_object  # noqa: F401
from ..types import SafetyViolation, SkillError
from .bridge_mixin import _BridgeSkillsMixin
from .manipulation_mixin import _ManipulationMixin
from .perception_mixin import _PerceptionMixin
from .specs import (  # noqa: F401
    TOOL_SPECS,
    _DESTINATION_FAMILY,
    _MOTION_SKILLS,
    _is_mock,
    _jpeg,
    _short,
)


class SkillRuntime(_PerceptionMixin, _BridgeSkillsMixin, _ManipulationMixin):
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
        #: optional rebot_grasp motion/gripper bridge (resolved lazily via
        #: property so LazySafeArm is never materialized at startup).
        self._bridge_override = None
        self._bridge_overridden = False
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
        self._default_classes = list(
            cfg.camera.get("detect_classes")
            or cfg.get("detect_classes", ["cup", "bottle", "box", "fruit", "toy"])
        )
        self._cartesian_planner = None  # lazy CartesianPlanner (planning module)


    @property
    def bridge(self):
        if self._bridge_overridden:
            return self._bridge_override
        return getattr(self.arm, "grasp_bridge", None)


    @bridge.setter
    def bridge(self, value) -> None:
        self._bridge_override = value
        self._bridge_overridden = True


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


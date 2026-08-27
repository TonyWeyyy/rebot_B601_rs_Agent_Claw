"""Tests for the rebot_grasp-aligned skill_grasp_object bridge path.

The legacy ``skill_grasp_object`` uses SafeArm.move_joints / move_cartesian +
_close_two_stage + air-grasp verification. The new bridge path mirrors
``rebot_grasp/scripts/main.py::execute_grasp`` exactly:

    open_gripper → move_to_traj(pre, 2.0) → move_to_traj(grasp, 1.5)
              → grasp(force=1.0) → move_to_traj(home, 3.0)

The legacy path is still exercised by existing tests with mock arm (no
bridge); these tests assert the bridge path is used when ``runtime.bridge``
is set.
"""

from unittest.mock import MagicMock, patch, call as mock_call

import numpy as np
import pinocchio as pin
import pytest

from wrc_demo.config import load_demo_config
from wrc_demo.skills.runtime import SkillRuntime
from wrc_demo.types import SkillError


# ──────────────── helpers ────────────────


def _grasp_pose(position=(0.30, -0.10, 0.05), rpy=(0.0, 1.2, 0.0)):
    """Build a Grasp-like MagicMock with .position / .rotation / .width_m / .approach."""
    grasp = MagicMock()
    grasp.position = np.array(position, dtype=float)
    grasp.rotation = pin.rpy.rpyToMatrix(*rpy)
    grasp.width_m = 0.030
    grasp.approach = np.array([0.0, 0.0, -1.0])
    grasp.quality = 0.9
    grasp.label = "banana"
    grasp.pregrasp_position = lambda offset_m: (
        grasp.position - grasp.approach * offset_m
    )
    return grasp


def _fix():
    fix = MagicMock()
    fix.detection.label = "banana"
    fix.detection.mask = None
    fix.detection.bbox = np.array([100, 100, 200, 200])
    fix.position = np.array([0.30, -0.10, 0.05])
    return fix


def _frame():
    """A minimal Frame mock whose .rgb is a real yellow BGR image (so
    detection_color can extract a color name)."""
    frame = MagicMock()
    frame.rgb = np.full((480, 640, 3), [0, 255, 255], dtype=np.uint8)  # yellow BGR
    return frame


def _make_runtime_with_bridge(bridge):
    """Build a SkillRuntime wired to use the bridge path."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    arm = MagicMock()
    arm.grasp_bridge = bridge
    arm.harness = MagicMock()
    arm.harness.vet_pose.return_value = None  # always safe by default
    arm.harness.allow_grasp_descent = MagicMock()
    arm.harness.clear_grasp_exemption = MagicMock()
    arm.get_state.return_value.q = np.array([0.0] * 6)

    # kin.fk is called for home_q → 6D; return a real matrix so
    # pin.rpy.matrixToRpy succeeds. (0.25, 0, 0.35) with rpy=(0, π/2, 0)
    # matches wrc_demo's home_q FK approximation.
    kin = MagicMock()
    home_T = np.eye(4)
    home_T[:3, 3] = [0.25, 0.0, 0.35]
    home_T[:3, :3] = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)
    kin.fk.return_value = home_T
    ik_ok = MagicMock(success=True, q=np.zeros(6))
    kin.ik.return_value = ik_ok

    rt = SkillRuntime(
        camera=MagicMock(),
        depth_provider=MagicMock(),
        detector=MagicMock(),
        extrinsics=MagicMock(),
        kin=kin,
        safe_arm=arm,
        memory=MagicMock(),
        beliefs=MagicMock(),
        trace=MagicMock(),
        cfg=cfg,
    )
    rt._reconcile_held = MagicMock()
    rt._localize = MagicMock(return_value=(_frame(), _fix()))
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])
    rt._max_width = 0.095
    return rt


def _patched_select_grasp(rt):
    """Context manager: stub select_grasp to return canned grasp + q seeds."""
    q_pre = np.array([0.1, 0.2, 0.3, -0.1, 0.0, 0.0])
    q_grasp = np.array([0.1, 0.2, 0.4, -0.2, 0.0, 0.0])
    return patch(
        "wrc_demo.skills.runtime.select_grasp",
        return_value=(_grasp_pose(), q_pre, q_grasp),
    )


# ──────────────── tests ────────────────


def test_bridge_path_calls_open_then_pre_then_grasp_then_home():
    """Bridge flow: ready(3.0) → open → pre(2.0) → grasp(1.5) → grasp(force=1.0) → ready(3.0)."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)

    with _patched_select_grasp(rt):
        rt.skill_grasp_object("banana")

    # open first (after ready)
    bridge.open_gripper.assert_called_once()
    # then grasp with force=1.0
    bridge.grasp.assert_called_once_with(force=1.0)
    # move_to_traj called 4 times: ready, pre, grasp, ready
    durations = [c.kwargs.get("duration") for c in bridge.move_to_traj.call_args_list]
    assert durations == [3.0, 2.0, 1.5, 3.0], (
        f"move_to_traj durations should be [3.0, 2.0, 1.5, 3.0], got {durations}"
    )


def test_bridge_path_pregrasp_position_is_offset_from_grasp():
    """pregrasp z should be grasp z + 0.08 (pregrasp_offset_m)."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)

    with _patched_select_grasp(rt):
        rt.skill_grasp_object("banana")

    # call 0 = ready, call 1 = pregrasp, call 2 = grasp, call 3 = ready
    pre_call = bridge.move_to_traj.call_args_list[1]
    assert pre_call.args[2] == pytest.approx(0.05 + 0.08)
    assert pre_call.kwargs["duration"] == 2.0

    grasp_call = bridge.move_to_traj.call_args_list[2]
    assert grasp_call.args[2] == pytest.approx(0.05)
    assert grasp_call.kwargs["duration"] == 1.5


def test_bridge_path_pregrasp_ik_fail_raises_skill_error():
    """If IK fails for the pregrasp target, raise SkillError."""
    from wrc_demo.types import SkillError

    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)
    # kin.ik calls: ready succeeds, pregrasp fails, rest succeed
    rt.kin.ik.side_effect = [
        MagicMock(success=True, q=np.zeros(6)),  # ready IK succeeds
        MagicMock(success=False, q=None),  # pregrasp IK fails
        MagicMock(success=True, q=np.zeros(6)),  # grasp IK succeeds
        MagicMock(success=True, q=np.zeros(6)),  # safety extra
        MagicMock(success=True, q=np.zeros(6)),
    ]

    with _patched_select_grasp(rt):
        with pytest.raises(SkillError, match="pregrasp IK failed"):
            rt.skill_grasp_object("banana")


def test_bridge_path_grasp_ik_fail_raises_skill_error():
    """If pregrasp IK succeeds but grasp IK fails, raise SkillError too."""
    from wrc_demo.types import SkillError

    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)
    rt.kin.ik.side_effect = [
        MagicMock(success=True, q=np.zeros(6)),  # ready OK
        MagicMock(success=True, q=np.zeros(6)),  # pregrasp OK
        MagicMock(success=False, q=None),  # grasp IK fails
        MagicMock(success=True, q=np.zeros(6)),  # safety extra
        MagicMock(success=True, q=np.zeros(6)),
    ]

    with _patched_select_grasp(rt):
        with pytest.raises(SkillError, match="grasp IK failed"):
            rt.skill_grasp_object("banana")


def test_bridge_path_empty_grasp_returns_failure_dict():
    """bridge.grasp returns False (no contact) → ok=False, no exception."""
    bridge = MagicMock()
    bridge.move_to_traj.return_value = True
    bridge.grasp.return_value = False
    rt = _make_runtime_with_bridge(bridge)

    with _patched_select_grasp(rt):
        result = rt.skill_grasp_object("banana")

    assert result["ok"] is False
    assert "empty grasp" in result["error"].lower()


def test_bridge_path_success_sets_held_object_and_marks_belief():
    """On successful grasp, runtime.held_object is set + belief removed."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)

    fix = _fix()
    rt._localize = MagicMock(return_value=(_frame(), fix))

    with _patched_select_grasp(rt):
        result = rt.skill_grasp_object("banana")

    assert result["ok"] is True
    assert rt.held_object == "banana"
    rt.beliefs.mark_removed.assert_called_once()


def test_legacy_path_used_when_bridge_is_none():
    """If bridge is None (mock arm), the legacy v5 path runs.

    We don't run the full legacy flow (needs harness limits mocks);
    instead we patch harness limits + early helpers so we can confirm
    the dispatch reached the legacy branch.
    """
    rt = _make_runtime_with_bridge(None)
    rt.bridge = None

    # Make harness limits return real numeric values so the legacy code
    # can compute z_floor / vet_pose etc.
    rt.arm.harness.limits.workspace_min = np.array([-1.0, -1.0, 0.0])
    rt.arm.harness.limits.table_z = 0.0
    rt.arm.harness.limits.max_joint_vel = 1.0
    rt.arm.harness.vet_pose = MagicMock(return_value=None)
    rt.arm.harness.allow_grasp_descent = MagicMock()
    rt.arm.harness.clear_grasp_exemption = MagicMock()

    # Patch perception + IK helpers so we don't need a real environment.
    rt._plan_grasps = MagicMock(return_value=[])  # empty → legacy raises
    rt.kin.fk.return_value = np.eye(4)
    rt.kin.ik.return_value = MagicMock(success=True, q=np.zeros(6))

    with pytest.raises(SkillError, match="no executable grasp"):
        rt.skill_grasp_object("banana")

    rt._plan_grasps.assert_called_once()
    assert rt.bridge is None


# ──────────────── pick_and_place bridge path ────────────────


def _fix_at(position, label="banana"):
    fix = MagicMock()
    fix.detection.label = label
    fix.detection.mask = None
    fix.detection.bbox = np.array([100, 100, 200, 200])
    fix.position = np.array(position, dtype=float)
    return fix


def _make_pick_and_place_runtime(bridge, *, already_held=None):
    """Build a SkillRuntime for pick_and_place tests."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    arm = MagicMock()
    arm.grasp_bridge = bridge
    arm.harness = MagicMock()
    arm.harness.vet_pose.return_value = None  # always safe by default
    arm.harness.allow_grasp_descent = MagicMock()
    arm.harness.clear_grasp_exemption = MagicMock()
    arm.get_state.return_value.q = np.zeros(6)

    kin = MagicMock()
    home_T = np.eye(4)
    home_T[:3, 3] = [0.25, 0.0, 0.35]
    home_T[:3, :3] = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)
    kin.fk.return_value = home_T
    ik_ok = MagicMock(success=True, q=np.zeros(6))
    kin.ik.return_value = ik_ok

    rt = SkillRuntime(
        camera=MagicMock(),
        depth_provider=MagicMock(),
        detector=MagicMock(),
        extrinsics=MagicMock(),
        kin=kin,
        safe_arm=arm,
        memory=MagicMock(),
        beliefs=MagicMock(),
        trace=MagicMock(),
        cfg=cfg,
    )
    rt._reconcile_held = MagicMock()
    rt._reobserve = MagicMock()
    rt.observe = MagicMock(return_value=_frame())
    rt._max_width = 0.095
    rt.held_object = already_held
    return rt


def test_pick_and_place_bridge_path_calls_full_sequence():
    """Bridge pick_and_place: same-frame detect → grasp → home → place → home."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    # Stubs: same-frame detect returns banana + box
    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        result = rt.skill_pick_and_place("banana", destination="yellow box")

    # Sequence: open → pre(2.0) → grasp(1.5) → grasp(force=1.0) → home(3.0)
    #        → box(2.0) → release_with_async_close(0.5, 0.8) → home(3.0)
    durations = [c.kwargs.get("duration") for c in bridge.move_to_traj.call_args_list]
    assert durations == [3.0, 2.0, 1.5, 3.0, 2.0, 3.0], (
        f"expected 6 move_to_traj calls with durations [3.0,2.0,1.5,3.0,2.0,3.0], got {durations}"
    )
    bridge.open_gripper.assert_called_once()
    bridge.grasp.assert_called_once_with(force=1.0)
    bridge.release_with_async_close.assert_called_once_with(timeout=0.5, delay_s=0.8)


def test_pick_and_place_bridge_path_uses_same_frame_for_both_targets():
    """Both banana and box must come from ONE observe() call."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # observe() called exactly once (same-frame detect)
    rt.observe.assert_called_once()
    # _single_localize called twice (once for each label) on the same frame
    assert rt._single_localize.call_count == 2
    # Both calls received the same frame argument
    frames_passed = [c.args[0] for c in rt._single_localize.call_args_list]
    assert frames_passed[0] is frames_passed[1], (
        "banana and box should be detected on the SAME frame"
    )


def test_pick_and_place_bridge_path_place_uses_hardcoded_z_0_08():
    """Place move_to_traj z is 0.08 hardcoded (matches set.py)."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # Call order: ready(0), pre(1), grasp(2), ready(3), place(4), ready(5)
    place_call = bridge.move_to_traj.call_args_list[4]
    assert place_call.args[2] == 0.08, (
        f"place z must be 0.08 hardcoded, got {place_call.args[2]}"
    )
    # rpy=(0, 1.2, 0)
    assert (place_call.args[3], place_call.args[4], place_call.args[5]) == (0.0, 1.2, 0.0)
    assert place_call.kwargs["duration"] == 2.0


def test_pick_and_place_bridge_path_returns_failure_if_no_box():
    """If box not detected, raise SkillError before grasp."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    rt._single_localize = MagicMock(side_effect=[banana_fix, None])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with pytest.raises(SkillError, match="yellow box"):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # Bridge.grasp NOT called (we never reached the grasp stage)
    bridge.grasp.assert_not_called()


def test_pick_and_place_bridge_path_clears_held_after_release():
    """After release, held_object is None + belief updated to drop point."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    assert rt.held_object is None
    # Belief was updated to the drop point
    assert rt.beliefs.update.called


def test_pick_and_place_bridge_path_skipped_when_legacy_no_destination():
    """Bridge path requires destination (set.py flow is pick-and-place); without
    it, the legacy path runs.
    """
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)
    rt._localize = MagicMock(side_effect=SkillError("no banana detected"))

    try:
        rt.skill_pick_and_place("banana", destination=None)  # no destination
    except (SkillError, Exception):
        pass  # legacy flow may raise; we just want to verify bridge NOT used

    # Bridge methods were never called (no destination → legacy path)
    bridge.open_gripper.assert_not_called()
    bridge.grasp.assert_not_called()
    bridge.move_to_traj.assert_not_called()


def test_pick_and_place_bridge_warns_when_box_top_above_release_height():
    """If box top is higher than the hardcoded release_z=0.08, log a warning."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    # Box top is 0.15 m — well above 0.08 release height
    box_fix.points = np.array([
        [0.35, 0.10, 0.10],
        [0.35, 0.11, 0.15],  # highest point
    ])
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # memory.add was called with a "warning" event mentioning the collision risk
    warning_calls = [
        c for c in rt.memory.add.call_args_list
        if c.args[0] == "warning"
    ]
    assert len(warning_calls) >= 1, (
        f"expected at least one warning event, got {rt.memory.add.call_args_list}"
    )
    msg = warning_calls[0].args[1]
    assert "box top_z" in msg
    assert "0.15" in msg


def test_pick_and_place_bridge_no_warning_when_box_top_below_release_height():
    """If box top is below 0.08 (e.g. flat box), no warning."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.05], label="yellow box")
    # Box top is 0.05 m — below 0.08 release height
    box_fix.points = np.array([[0.35, 0.10, 0.05]])
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # No "warning" event for box top
    warning_calls = [
        c for c in rt.memory.add.call_args_list
        if c.args[0] == "warning" and "box top_z" in c.args[1]
    ]
    assert len(warning_calls) == 0


def test_pick_and_place_bridge_clears_held_on_place_failure():
    """If the home-after-place motion fails, held_object must NOT be latched.

    The bridge wraps step 4-5 in try/except that reconciles held_object
    + beliefs so a failed release doesn't leave the runtime thinking it
    is still holding the banana.
    """
    from wrc_demo.types import SafetyViolation

    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    box_fix.points = np.array([[0.35, 0.10, 0.10]])
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    # First 4 move_to_traj succeed; 5th (home after place) raises.
    # The IK calls succeed up to that point so the bridge reaches step 5.
    call_count = {"n": 0}

    def move_to_traj_with_failure(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 5:
            raise SafetyViolation("home_after_place IK failed")
        return True  # 2026-08-13: explicit success (was implicit None; new strict return check requires True)

    bridge.move_to_traj.side_effect = move_to_traj_with_failure

    with _patched_select_grasp(rt):
        with pytest.raises(SafetyViolation, match="home_after_place"):
            rt.skill_pick_and_place("banana", destination="yellow box")

    # held_object must be cleared (try/except reconciled)
    assert rt.held_object is None
    assert rt._held_det_label is None
    assert rt._held_color is None
    # beliefs.update was called to record the object at the drop point
    rt.beliefs.update.assert_called()

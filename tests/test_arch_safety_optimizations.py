"""Regression & safety tests for architecture, control, and anti-drop fixes."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from wrc_demo.agent.llm import LLMResponse, MockLLM, ToolCall
from wrc_demo.agent.orchestrator import AgentOrchestrator
from wrc_demo.agent.reflex import ExperienceMemory, parse_command
from wrc_demo.calibration import CalibrationSample, joint_solve, so3_exp
from wrc_demo.config import Cfg, load_demo_config
from wrc_demo.control.lazy_arm import LazySafeArm
from wrc_demo.grasping.selector import select_grasp
from wrc_demo.memory.episodic import EpisodicMemory
from wrc_demo.memory.grasp_memory import GraspOutcomeMemory, object_profile
from wrc_demo.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from wrc_demo.skills.rebot_grasp_bridge import RebotGraspBridge
from wrc_demo.skills.teach import KeyboardState, record_real, replay_real, save_slot
from wrc_demo.types import Grasp, ObjectFix, SafetyViolation


class _StubKin:
    n_joints = 6
    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
            [q[0], q[1], max(float(q[2]), 0.1)],
        ])


def _make_harness(**kw):
    base = dict(
        workspace_min=np.array([0.10, -0.30, -0.01]),
        workspace_max=np.array([0.50, 0.30, 0.55]),
        table_z=0.0,
        table_clearance=0.02,
        max_joint_vel=1.2,
        watchdog_s=1e9,
    )
    base.update(kw)
    return SafetyHarness(SafetyLimits(**base), _StubKin())


# ── 1. Anti-drop & SafeArm / Bridge safety invariants ─────────────────────


def test_safe_arm_stop_freezes_without_torque_off():
    """Soft stop must NEVER call sdk_arm.estop()/disable_all() (prevents arm drop)."""
    sdk_arm = MagicMock()
    sdk_ctrl = MagicMock()
    sdk_ctrl._n = 6
    sdk_ctrl._stop_send = threading.Event()
    sdk_ctrl._qd_target = np.ones(6)
    h = _make_harness()
    arm = SafeArm(sdk_arm, sdk_ctrl, h, _StubKin(), Cfg({"gripper": {"open_pos": 5.0, "closed_pos": 0.0}}))

    arm.stop()

    assert h.estopped is True
    assert sdk_ctrl._stop_send.is_set()
    assert np.allclose(sdk_ctrl._qd_target, 0.0)
    sdk_arm.estop.assert_not_called()
    sdk_arm.disable_all.assert_not_called()
    sdk_arm.disconnect.assert_not_called()


def test_safe_arm_disconnect_never_raw_disconnects_if_safe_home_fails():
    """If ctrl.end() (which runs safe_home before disconnect) raises while the
    control loop is running, SafeArm.disconnect must NOT blindly call raw
    sdk_arm.disconnect() and drop a suspended arm."""
    sdk_arm = MagicMock()
    sdk_ctrl = MagicMock()
    sdk_ctrl._n = 6
    sdk_ctrl._running = True
    sdk_ctrl._stop_send = threading.Event()
    sdk_ctrl._send_thread = None
    sdk_ctrl.end.side_effect = RuntimeError("safe_home interrupted")

    arm = SafeArm(sdk_arm, sdk_ctrl, _make_harness(), _StubKin(), Cfg({}))
    arm.disconnect()

    sdk_ctrl._stop_send.set()
    sdk_ctrl.end.assert_called_once()
    sdk_arm.disconnect.assert_not_called()


def test_safe_arm_disconnect_joins_send_thread_before_ctrl_end():
    """Active _send_thread must exit BEFORE safe_home/ctrl.end() writes _q_target."""
    order = []
    sdk_arm = MagicMock()
    sdk_ctrl = MagicMock()
    sdk_ctrl._n = 6
    sdk_ctrl._running = True
    sdk_ctrl._stop_send = threading.Event()

    def _worker():
        while not sdk_ctrl._stop_send.is_set():
            time.sleep(0.005)
        order.append("send_thread_exited")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    sdk_ctrl._send_thread = t
    sdk_ctrl.end.side_effect = lambda: order.append("ctrl_end")

    arm = SafeArm(sdk_arm, sdk_ctrl, _make_harness(), _StubKin(), Cfg({}))
    arm.disconnect()

    assert order == ["send_thread_exited", "ctrl_end"]


def test_safe_arm_gripper_polarity_estop_and_state():
    """open_gripper respects signed span; gripper ops obey e-stop; width_frac is 0=closed..1=open."""
    sdk_arm = MagicMock()
    sdk_arm.has_gripper = True
    sdk_arm.get_state.return_value = (np.array([0, 0, 0, 0, 0, 0, -3.4]), np.zeros(7), np.zeros(7))
    sdk_ctrl = MagicMock()
    sdk_ctrl._n = 6
    h = _make_harness()
    cfg = Cfg({"gripper": {"open_pos": -6.8, "closed_pos": 0.0, "max_width_m": 0.09}})
    arm = SafeArm(sdk_arm, sdk_ctrl, h, _StubKin(), cfg)

    # Half-open with negative open_pos should target -3.4, NOT +3.4
    arm.open_gripper(distance_m=0.045)
    pos_cmd = sdk_ctrl.set_gripper_target.call_args[0][0]
    assert np.isclose(pos_cmd, -3.4, atol=1e-6)

    # get_state populates gripper_pos and _gripper_width_frac returns 0.5 (half open)
    st = arm.get_state()
    assert st.gripper_valid is True
    assert np.isclose(st.gripper_pos, -3.4)
    assert np.isclose(arm._gripper_width_frac(), 0.5)

    # When fully closed (0.0), width_frac == 0.0 (not 1.0)
    sdk_arm.get_state.return_value = (np.zeros(7), np.zeros(7), np.zeros(7))
    assert np.isclose(arm._gripper_width_frac(), 0.0)

    # E-stop vetoes all gripper operations
    h.estop("test")
    with pytest.raises(SafetyViolation):
        arm.open_gripper()
    with pytest.raises(SafetyViolation):
        arm.set_gripper(-1.0)
    with pytest.raises(SafetyViolation):
        arm.close_gripper_two_stage()


def test_bridge_move_to_traj_enforces_waypoint_hook_and_estop():
    """RebotGraspBridge passes waypoint_hook to SDK controller and raises
    SafetyViolation if any intermediate waypoint violates the harness."""
    h = _make_harness()

    class _FakeSdkCtrl:
        _dt = 0.01

        def move_to_traj(self, x, y, z, roll, pitch, yaw, duration=2.0, waypoint_hook=None):
            assert waypoint_hook is not None, "bridge must pass waypoint_hook"
            q0 = np.array([0.30, 0.0, 0.30, 0.0, 0.0, 0.0])
            # Waypoint 1 jumps way outside workspace (x=0.95)
            q_bad = np.array([0.95, 0.0, 0.30, 0.0, 0.0, 0.0])
            assert waypoint_hook(q0) is True
            assert waypoint_hook(q_bad) is False
            return True

        def wait_motion_done(self, timeout=5.0):
            return True

    drv = MagicMock()
    bridge = RebotGraspBridge(_FakeSdkCtrl(), drv, harness=h)
    with pytest.raises(SafetyViolation):
        bridge.move_to_traj(0.30, 0.0, 0.30, 0.0, 1.2, 0.0, duration=1.0)
    assert h.estopped is True

    # Gripper ops on bridge also obey e-stop
    with pytest.raises(SafetyViolation):
        bridge.open_gripper()
    with pytest.raises(SafetyViolation):
        bridge.grasp(force=1.0)
    with pytest.raises(SafetyViolation):
        bridge.release_gripper()


def test_lazy_safe_arm_does_not_materialize_until_motion():
    """LazySafeArm supports heartbeats & e-stop without powering motors."""
    h = _make_harness()
    built = []
    inner = MagicMock()
    inner.harness = h
    inner.n_joints = 6

    def _factory():
        built.append(1)
        return inner

    lazy = LazySafeArm(_factory, harness=h)
    assert lazy.connected is False
    lazy.harness.heartbeat()
    assert built == []

    # Stopping before materialization latches e-stop without powering motors
    lazy.stop()
    assert h.estopped is True
    assert built == []
    with pytest.raises(SafetyViolation):
        lazy.move_joints(np.zeros(6))
    assert built == []

    # After resume, first motion materializes exactly once
    lazy.resume()
    lazy.move_joints(np.zeros(6))
    assert len(built) == 1
    assert lazy.connected is True


# ── 2. Teach skills anti-drop & keyboard listener fixes ───────────────────


def test_keyboard_state_stays_alive_for_start_and_stop():
    """First 'q' starts recording without killing the listener; 'esc' stops it."""
    from types import SimpleNamespace
    from pynput import keyboard

    KeyboardState.reset()
    ret1 = KeyboardState.on_press(SimpleNamespace(char="q"))
    assert ret1 is None
    assert KeyboardState.ENTER is True
    KeyboardState.ENTER = False

    # Listener still alive -> second 'q' can stop recording
    ret2 = KeyboardState.on_press(SimpleNamespace(char="q"))
    assert ret2 is None
    assert KeyboardState.ENTER is True

    # Esc stops listener
    ret3 = KeyboardState.on_press(keyboard.Key.esc)
    assert ret3 is False
    assert KeyboardState.ESC is True


def test_teach_record_uses_active_mit_controller_and_rate_limits(tmp_path):
    """When 500 Hz MIT controller is running, record_real writes rate-limited
    targets into controller._q_target and never calls send_pos_vel."""
    master = MagicMock()
    master.joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    calls = {"n": 0}

    def _read():
        calls["n"] += 1
        if calls["n"] == 1:
            KeyboardState.ENTER = True  # start recording
        elif calls["n"] == 4:
            KeyboardState.ENTER = True  # stop recording
        return {
            "joint1": 10.0 * calls["n"], "joint2": 5.0, "joint3": 5.0,
            "joint4": 0.0, "joint5": 0.0, "joint6": 0.0, "gripper": 0.0,
        }

    master.get_joint_states.side_effect = _read
    slave = MagicMock()
    slave.arm.num_joints = 6
    slave.has_gripper = True

    ctrl = MagicMock()
    ctrl._running = True
    ctrl._n = 6
    ctrl._q_target = np.zeros(6)
    gripper = MagicMock()

    fake_listener = MagicMock()
    res = record_real(
        master, slave, slot=1, recordings_dir=tmp_path,
        keyboard_listener_factory=lambda _ks: fake_listener,
        controller=ctrl, gripper=gripper,
        mapper=lambda d: np.array([0.5, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0]),
    )
    assert res["ok"] is True
    slave.arm.send_pos_vel.assert_not_called()
    # Rate-limited: after 4 steps of max 0.08 rad, joint1 cannot exceed 0.32 rad
    assert abs(ctrl._q_target[0]) <= 0.33
    gripper.set_position_target.assert_called()


def test_teach_replay_active_controller_never_disconnects(tmp_path):
    """replay_real on an active SafeArm uses move_joints/move_joints_path/home
    and never spins up a competing controller or disconnects the motors."""
    frames = [
        {"t": 0.0, "master_q": [0.0] * 7, "slave_q": [0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]},
        {"t": 0.5, "master_q": [0.0] * 7, "slave_q": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]},
    ]
    save_slot(1, frames, np.zeros(7), np.zeros(7), tmp_path)

    slave = MagicMock()
    slave.arm.num_joints = 6
    ctrl = MagicMock()
    ctrl._running = True
    safe_arm = MagicMock()
    safe_arm.move_joints.return_value = True
    safe_arm.move_joints_path.return_value = True
    safe_arm.home.return_value = True
    gripper = MagicMock()

    res = replay_real(
        slave, slot=1, recordings_dir=tmp_path,
        safe_arm=safe_arm, controller=ctrl, gripper=gripper,
    )
    assert res["ok"] is True
    assert res["replayed_to_home"] is True
    safe_arm.move_joints.assert_called_once()
    safe_arm.move_joints_path.assert_called_once()
    safe_arm.home.assert_called_once()
    slave.disconnect.assert_not_called()
    ctrl.end.assert_not_called()


# ── 3. Orchestrator, sliding window, switch_llm, multi-object ─────────────


def test_trim_history_never_orphans_tool_message():
    msgs = [{"role": "user", "content": "intro"}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [ToolCall("list_objects", {}, f"c{i}")]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "list_objects", "content": "{}"})
    trimmed = AgentOrchestrator._trim_history(msgs, max_history=8)
    assert trimmed[0]["role"] == "user"
    assert trimmed[1]["role"] != "tool", "orphaned tool message breaks OpenAI/Anthropic API"


def test_switch_llm_loads_real_profile_yaml():
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime = MagicMock()
    runtime.trace = MagicMock()
    orch = AgentOrchestrator(MockLLM([]), runtime, cfg=cfg)
    new_llm = orch.switch_llm("mock")
    assert isinstance(new_llm, MockLLM)
    runtime.trace.record_event.assert_called_with("llm_switch", {"profile": "mock"})


def test_multi_object_pick_and_place_allowed_same_pair_blocked():
    """Different objects can be picked in one task; repeating the SAME pair is blocked."""
    script = [
        LLMResponse(tool_calls=[ToolCall("pick_and_place", {"object": "banana", "destination": "box"}, "c1")]),
        LLMResponse(tool_calls=[ToolCall("pick_and_place", {"object": "cube", "destination": "box"}, "c2")]),
        LLMResponse(tool_calls=[ToolCall("pick_and_place", {"object": "banana", "destination": "box"}, "c3")]),
    ]
    rt = MagicMock()
    rt.memory.digest.return_value = ""
    rt.last_frame = None
    rt.execute.side_effect = lambda name, args: (
        {"ok": True, "task_complete": True} if name == "task_done" else {"ok": True, "placed_at": [0.3, 0, 0.1]}
    )
    orch = AgentOrchestrator(MockLLM(script), rt, decompose=False)
    rep = orch.run_task("put banana and cube in box")
    assert rep.success is True
    tools_called = [e["tool"] for e in rep.tool_log]
    assert tools_called == ["pick_and_place", "pick_and_place", "task_done"]


# ── 4. Hand-eye solver outlier robustness ─────────────────────────────────


def test_handeye_solver_consistent_floor_noise_keeps_inliers():
    """With uniform small noise across all samples, med + 3*MAD must NOT reject everything."""
    rng = np.random.default_rng(42)
    X_true = np.eye(4)
    X_true[:3, 3] = [0.25, -0.05, 0.65]
    X_true[:3, :3] = so3_exp(np.array([0.1, 0.2, -0.1]))
    Y_true = np.eye(4)
    Y_true[:3, 3] = [0.01, 0.0, 0.03]

    samples = []
    for i in range(12):
        G = np.eye(4)
        G[:3, 3] = [0.25 + 0.015 * i, 0.04 * math.sin(i), 0.25 + 0.01 * math.cos(i)]
        G[:3, :3] = so3_exp(np.array([0.2 * math.cos(i), 0.15 * math.sin(i), 0.1 * i]))
        M = np.linalg.inv(X_true) @ G @ Y_true
        M[:3, 3] += rng.normal(scale=0.001, size=3)
        samples.append(CalibrationSample(f"p{i}", G, M))

    fit = joint_solve(samples)
    assert len(fit.inliers) >= 9, f"too many false outliers: {len(fit.outliers)}/{len(samples)}"
    assert fit.metrics["translation_rmse_mm"] < 5.0


# ── 5. P2 optimizations: grasp memory rerank, CJK reflex/memory, episodic ─


def test_grasp_memory_rerank_propagates_through_select_grasp():
    """GraspOutcomeMemory.rerank must actually influence select_grasp ordering."""
    mem = GraspOutcomeMemory(path=None)
    fix = ObjectFix(
        label="cube",
        position=np.array([0.3, 0.0, 0.03]),
        points=np.array([[0.28, -0.02, 0.0], [0.32, 0.02, 0.06]]),
        detection=MagicMock(conf=0.9),
        extent=np.array([0.06, 0.04, 0.04]),
        axes=np.eye(3),
    )
    g_side = Grasp(
        position=np.array([0.3, 0.0, 0.03]),
        rotation=np.eye(3),
        width_m=0.04,
        approach=np.array([1.0, 0.0, 0.0]),  # horizontal (vert=0)
        quality=0.95,
        label="cube",
    )
    g_top = Grasp(
        position=np.array([0.3, 0.0, 0.03]),
        rotation=np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float),
        width_m=0.04,
        approach=np.array([0.0, 0.0, -1.0]),  # top-down (vert=1)
        quality=0.80,
        label="cube",
    )
    # Record repeated wins for top-down grasps
    for _ in range(5):
        mem.record("cube", fix, g_top, success=True)

    reranked = mem.rerank([g_side, g_top], "cube", fix)
    assert reranked[0].approach[2] == -1.0

    kin = MagicMock()
    kin.ik.return_value = MagicMock(success=True, q=np.zeros(6), error=0.0)
    chosen, _, _ = select_grasp(reranked, kin, np.zeros(6))
    assert chosen.approach[2] == -1.0, "select_grasp overrode grasp_memory rerank!"


def test_cjk_experience_memory_and_wave_reflex():
    plan = parse_command("挥手")
    assert plan is not None and plan.calls == [("wave", {})]
    em = ExperienceMemory(path=None)
    em.record("把香蕉放到盒子里", [("pick_and_place", {"object": "banana", "destination": "box"})], True, 4.0)
    hit = em.recall("把香蕉放到盒子里")
    assert hit is not None and hit["calls"][0][0] == "pick_and_place"


def test_wrc_gripper_prime_refuses_unready_arm_feedback():
    """WrcGripper._prime_arm_target must refuse to start MIT control loop if
    any arm motor has no feedback yet (prevents snapping to bogus zeros)."""
    from wrc_demo.control.gripper import WrcGripper

    bad_motor = MagicMock()
    bad_motor.get_state.return_value = None
    good_motor = MagicMock()
    good_motor.get_state.return_value = MagicMock(pos=0.2, vel=0.0, torq=0.0)

    arm_group = MagicMock()
    arm_group.num_joints = 2
    arm_group.joint_names = ["j1", "j2"]
    arm_group._mm = {"j1": good_motor, "j2": bad_motor}

    gripper_group = MagicMock()
    gripper_group._jcfgs = [MagicMock(name="g")]
    arm = MagicMock()
    arm.groups = {"arm": arm_group, "gripper": gripper_group}
    arm.has_gripper = True
    arm.get_state.return_value = (np.zeros(3), np.zeros(3), np.zeros(3))

    ctrl = MagicMock()
    ctrl._running = False
    ctrl._q_target = np.zeros(2)
    ctrl._qd_target = np.zeros(2)

    g = WrcGripper(arm, ctrl)
    with pytest.raises(RuntimeError, match="Arm motor feedback not ready"):
        g._prime_arm_target(timeout=0.05)


def test_episodic_memory_compacts_vector_index():
    now = [0.0]
    mem = EpisodicMemory(horizon_s=5.0, max_events=5, embed_dim=16, clock=lambda: now[0])
    for i in range(25):
        now[0] = float(i)
        mem.add("note", f"ev{i}", embedding=np.ones(16) * (i + 1))
    assert len(mem._index) <= 10

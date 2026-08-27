"""wrc_demo control backends + the ``make_arm`` factory.

After the SDK migration (2026-07-24) this package contains:

- ``MockArm`` — perfect-tracking test double; the harness is real so
  per-waypoint approval exercises the same code as production.
- ``IsaacArm`` — Isaac Sim bridge (legacy path; not wired to ``SafeArm``
  yet, see module docstring).
- ``make_arm(cfg, kinematics)`` — single factory used by ``demo.py`` /
  ``mcp_server.py``: returns the right arm type (MockArm, IsaacArm, or
  a ``SafeArm`` wrapping the vendored SDK).

Real hardware (``rebot_rs``) is now driven by the vendored SDK at
``sdk/reBotArm_control_py/`` and wrapped by ``SafeArm`` in
``wrc_demo.safety.harness``. There is no ``ArmBase`` ABC here anymore
— the SDK is the single source of truth for the motion surface, and
``MockArm`` / ``SafeArm`` are duck-typed to share it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ..config import Cfg, SDK_DIR
from ..safety.harness import Kinematics, SafeArm, SafetyHarness, SafetyLimits
from .isaac_arm import IsaacArm
from .mock_arm import MockArm
from .usd_model import (
    apply_joint_signs,
    urdf_xml_from_usd,
)

# Union of every "arm the skill runtime can hold" type. All three
# expose the same public surface (move_joints / move_cartesian /
# set_gripper / close_gripper_two_stage / stop / resume / disconnect /
# harness).
Arm = Union[MockArm, IsaacArm, SafeArm]

__all__ = [
    "Arm",
    "IsaacArm",
    "MockArm",
    "apply_joint_signs",
    "make_arm",
    "urdf_xml_from_usd",
]


def make_arm(cfg: Cfg, kinematics: Kinematics | None = None) -> Arm:
    """Build the arm requested by ``cfg.arm['type']``.

    For ``rebot_rs`` this assembles the full real-arm stack:

        RebotArm(hw_yaml) → connect → RebotArmEndPose(arm, mit)
            → WrcGripper.start() (extends the SDK control loop with
              gripper_tick)
            → SafeArm(arm, ctrl, harness, kin, arm_cfg)
            → RebotGraspBridge(ctrl, gripper, harness) attached as
              ``safe_arm.grasp_bridge``

    For ``mock`` it returns a ``MockArm`` (perfect tracking, real
    harness — exercises every gating code path used in production).

    For ``isaac`` it returns ``IsaacArm`` directly (legacy bridge path;
    not wrapped by ``SafeArm`` yet — that's a follow-up).
    """
    arm_type = str(cfg.arm.get("type", "mock"))

    if arm_type == "mock":
        arm = MockArm(cfg.arm, kinematics=kinematics)
        arm.connect()
        arm.grasp_bridge = None
        return arm

    if arm_type == "isaac":
        arm = IsaacArm(cfg.arm, kinematics=kinematics)
        arm.connect()
        arm.grasp_bridge = None
        return arm

    if arm_type == "rebot_rs":
        # Real hardware: vendored SDK + WrcGripper + SafeArm wrapping.
        # wrc_demo ``__init__.py`` already inserts ``wrc_demo/sdk/`` into
        # ``sys.path``, so ``reBotArm_control_py`` is importable here
        # without any PYTHONPATH edit from the operator.
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose

        from ..skills.rebot_grasp_bridge import RebotGraspBridge
        from .gripper import WrcGripper

        hw_yaml_name = str(cfg.arm.get("hw_yaml", "rebotarm_rs.yaml"))
        # SDK's RebotArm resolves hw_yaml relative to sdk/config/.
        sdk_cfg_path = SDK_DIR / "config" / hw_yaml_name
        if not sdk_cfg_path.exists():
            raise FileNotFoundError(
                f"SDK hardware config not found: {sdk_cfg_path} "
                f"(wrc_demo/sdk/config/{hw_yaml_name})"
            )
        sdk_arm = RebotArm(hw_yaml=hw_yaml_name)
        sdk_arm.connect()
        # Clear stale fault bits on every motor.  The SDK enable()
        # does NOT call clear_error; motors left in fault state from a
        # previous session silently reject MIT commands.
        #
        # Some RobStride firmwares ACKing the clear_error with a small
        # delay right after power-up or right after the previous session
        # bus-off'd.  Retry up to 3 times with a 100 ms gap so transient
        # comm_type=4 ack timeouts don't kill the boot.
        from motorbridge.errors import CallError as _CE
        for _name, _motor in sdk_arm._motor_map.items():
            for _attempt in range(3):
                try:
                    _motor.clear_error()
                    break
                except _CE as _e:
                    if "ack timeout" not in str(_e).lower() or _attempt == 2:
                        raise
                    import time as _t; _t.sleep(0.1)
        sdk_ctrl = RebotArmEndPose(
            sdk_arm,
            arm_control_mode="mit",
            use_gravity_ff=True,
        )
        # NOTE: sdk_ctrl.start() is NOT called here. WrcGripper
        # cooperates with the SDK control loop via ``_loop_cb`` wrapper
        # — see WrcGripper.start() / docs/adr/0005-port-gripper-to-wrc.md.
        gripper = WrcGripper(sdk_arm, sdk_ctrl)
        gripper.start()

        harness = SafetyHarness(
            SafetyLimits.from_config(cfg.safety), kinematics=kinematics
        )
        safe_arm = SafeArm(sdk_arm, sdk_ctrl, harness, kinematics, cfg.arm)
        safe_arm.grasp_bridge = RebotGraspBridge(
            sdk_ctrl, gripper, harness=harness,
        )
        return safe_arm

    raise ValueError(
        f"unknown arm type {arm_type!r}; "
        f"expected one of: mock, isaac, rebot_rs"
    )

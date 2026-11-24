"""LazyArm: defer motor bring-up until the first motion command.

The MCP server pre-warms perception at startup so the world model is hot
when a chat command arrives -- but powering the CAN bus / enabling motors as
a side effect of *starting a chat gateway* would be surprising and unsafe.
LazyArm materializes the real arm on first use (~1-2 s of CAN parameter
reads), keeping startup safe and the first grasp fast.
"""

from __future__ import annotations

import threading

from ..types import SafetyViolation
from .arm_base import ArmBase


class LazySafeArm:
    """Defer real-arm motor bring-up until the first motion command.

    Holds a pre-built ``SafetyHarness`` so ``WorldWatcher`` can heartbeat it
    at 3 Hz and ``emergency_stop`` can latch e-stop WITHOUT powering the CAN
    bus or enabling motors. Once materialized, delegates all motion and
    anti-drop shutdown (``SafeArm.disconnect``) to the underlying ``SafeArm``.
    """

    def __init__(self, factory, harness, n_joints: int = 6):
        self._factory = factory
        self._harness = harness
        self._n_joints_default = int(n_joints)
        self._arm = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._arm is None:
                if self._harness is not None and self._harness.estopped:
                    raise SafetyViolation("e-stopped")
                self._arm = self._factory()
            return self._arm

    @property
    def connected(self) -> bool:
        return self._arm is not None

    @property
    def harness(self):
        return self._arm.harness if self._arm is not None else self._harness

    @property
    def n_joints(self) -> int:
        return self._arm.n_joints if self._arm is not None else self._n_joints_default

    @property
    def grasp_bridge(self):
        return getattr(self._ensure(), "grasp_bridge", None)

    def connect(self) -> None:
        """Deliberate no-op: materialization happens on first motion."""

    def disconnect(self) -> None:
        with self._lock:
            if self._arm is not None:
                self._arm.disconnect()
                self._arm = None

    def stop(self) -> None:
        """Latch e-stop without powering up a never-materialized arm."""
        if self._harness is not None:
            self._harness.estop(reason="lazy_safe_arm.stop()")
        if self._arm is not None:
            self._arm.stop()

    def resume(self) -> None:
        if self._harness is not None:
            self._harness.reset_estop()
        if self._arm is not None:
            self._arm.resume()

    def get_state(self):
        return self._ensure().get_state()

    def move_joints(self, *args, **kwargs) -> bool:
        return self._ensure().move_joints(*args, **kwargs)

    def move_joints_path(self, *args, **kwargs) -> bool:
        return self._ensure().move_joints_path(*args, **kwargs)

    def move_cartesian(self, *args, **kwargs) -> bool:
        return self._ensure().move_cartesian(*args, **kwargs)

    def home(self, *args, **kwargs) -> bool:
        return self._ensure().home(*args, **kwargs)

    def set_gripper(self, *args, **kwargs) -> None:
        self._ensure().set_gripper(*args, **kwargs)

    def open_gripper(self, *args, **kwargs) -> None:
        self._ensure().open_gripper(*args, **kwargs)

    def close_gripper_two_stage(self, *args, **kwargs):
        return self._ensure().close_gripper_two_stage(*args, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ensure(), name)


class LazyArm(ArmBase):
    def __init__(self, factory, n_joints: int = 6):
        self._factory = factory
        self._n_joints_default = n_joints
        self._arm: ArmBase | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> ArmBase:
        with self._lock:
            if self._arm is None:
                arm = self._factory()
                arm.connect()
                self._arm = arm
            return self._arm

    @property
    def connected(self) -> bool:
        return self._arm is not None

    @property
    def n_joints(self) -> int:  # type: ignore[override]
        return self._arm.n_joints if self._arm is not None else self._n_joints_default

    @property
    def settle_tol(self) -> float:  # type: ignore[override]
        return self._arm.settle_tol if self._arm is not None else ArmBase.settle_tol

    # ── ArmBase surface ──────────────────────────────────────────────────

    def connect(self) -> None:
        """Deliberate no-op: materialization happens on first *use*."""

    def disconnect(self) -> None:
        with self._lock:
            if self._arm is not None:
                self._arm.disconnect()
                self._arm = None

    def get_state(self):
        return self._ensure().get_state()

    def send_joint_target(self, q) -> None:
        self._ensure().send_joint_target(q)

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        self._ensure().set_gripper(pos, effort)

    def stop(self) -> None:
        # An e-stop on a never-materialized arm must NOT power the bus up.
        if self._arm is not None:
            self._arm.stop()

    def stream_to(self, *args, **kwargs) -> bool:
        # Delegate wholesale so backend overrides (settle_tol, pacing) apply.
        return self._ensure().stream_to(*args, **kwargs)

    def wait_settled(self, *args, **kwargs) -> bool:
        return self._ensure().wait_settled(*args, **kwargs)

    def resume(self) -> None:
        """Clear a soft stop on the underlying arm, whatever its idiom."""
        if self._arm is None:
            return
        if hasattr(self._arm, "resume"):
            self._arm.resume()
        elif hasattr(self._arm, "_stopped"):
            self._arm._stopped = False

    def __getattr__(self, name: str):
        # Backend-specific extras (close_gripper_two_stage, ...) reach the
        # real arm; touching them materializes it, which is the caller's
        # intent when invoking hardware-specific ops.
        if name.startswith("_"):  # never materialize on private/dunder probes
            raise AttributeError(name)
        return getattr(self._ensure(), name)

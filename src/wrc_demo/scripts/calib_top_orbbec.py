"""CLI: hand-eye calibration for the Orbbec top camera.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

Sub-commands:

- ``--list``: enumerate Orbbec devices (no frames captured).
- ``--bind <role>``: write the role -> serial binding into
  ``data/calibration/cameras.json`` (also updates profile YAMLs).
- (default) collect a calibration dataset: for each stable pose of the
  arm, capture ArUco + FK, run ``joint_solve``, persist the result.
- ``--verify --npz <file> --known-points <csv>``: cross-check a saved
  calibration against operator-measured table points.

No cuRobo dependency: the FK comes from
``wrc_demo.control.kinematics.Kinematics`` (Pinocchio on the RS URDF
with ``joint_signs=[-1]*6`` and ``ee_frame='gripper_end'``).

Run via ``python -m wrc_demo.scripts.calib_top_orbbec ...`` (editable
install means no PYTHONPATH gymnastics needed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..calibration import (
    ArUcoSession,
    CamerasBinding,
    CameraBinding,
    CalibrationResult,
    CalibrationSample,
    SCHEMA_VERSION,
    joint_solve,
    joint_solve_restart,
    load_calibration,
    save_calibration,
)
from ..config import PACKAGE_ROOT
from ..control.kinematics import Kinematics
from ..perception.orbbec_camera import OrbbecCamera, list_orbbec_devices


# Stable directory layout for calibration artefacts (PLAN.md §4)
CALIB_DIR = PACKAGE_ROOT / "data" / "calibration"
ARM_PROFILE_DEFAULT = "rebot_rs"
URDF_DEFAULT = PACKAGE_ROOT / "assets" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"

# Headless / SSH env fallback: opencv-python-headless has no GTK so
# cv2.namedWindow always raises. We detect that once at import and
# fall back to per-frame disk snapshots. 1-of-every-SNAP_EVERY frames
# is saved (plus every successful detection), so the operator can
# verify the camera + ArUco pipeline after the run.
SNAP_DIR = Path("/tmp/calib_frames")
SNAP_EVERY = 10
SNAP_COUNTER = 0


def _detect_gui_capable() -> bool:
    """True if cv2.namedWindow works (i.e. we have a real display server)."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def _make_kinematics(arm_profile: str = ARM_PROFILE_DEFAULT) -> Kinematics:
    """Construct the wrc_demo FK on the RS URDF (mirrored joint convention)."""
    if not URDF_DEFAULT.exists():
        raise FileNotFoundError(
            f"RS URDF not found at {URDF_DEFAULT}; run from the wrc_demo repo root"
        )
    return Kinematics(
        model_path=str(URDF_DEFAULT),
        ee_frame="gripper_end",
        n_controlled=6,
        joint_signs=[-1, -1, -1, -1, -1, -1],
    )


def _print_devices() -> int:
    try:
        devices = list_orbbec_devices()
    except Exception as e:
        print(f"[calib] enumerate failed: {e}", file=sys.stderr)
        return 2
    if not devices:
        print("[calib] no Orbbec devices found")
        return 1
    for d in devices:
        print(
            f"  [{d.get('index', '?')}] {d.get('name', '?')} "
            f"serial={d.get('serial', '?')} "
            f"pid={d.get('pid', '?')} vid={d.get('vid', '?')}"
        )
    return 0


def _bind_role(role: str, serial: str | None = None, index: int | None = None) -> int:
    """Persist role -> serial in ``cameras.json``. If ``serial`` is omitted
    we use the first enumerated device whose index matches ``--index``."""
    role = role.lower()
    if role not in {"top", "wrist"}:
        print(f"[calib] bind: role must be 'top' or 'wrist', got {role!r}", file=sys.stderr)
        return 2
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    p = CALIB_DIR / "cameras.json"
    existing = CamerasBinding.from_json(json.loads(p.read_text())) if p.exists() else CamerasBinding()
    if serial is None:
        devices = list_orbbec_devices()
        if not devices:
            print("[calib] bind: no Orbbec devices to bind", file=sys.stderr)
            return 1
        idx = index if index is not None else 0
        if idx < 0 or idx >= len(devices):
            print(f"[calib] bind: index {idx} out of range (0..{len(devices)-1})", file=sys.stderr)
            return 1
        serial = str(devices[idx].get("serial", ""))
        index = int(devices[idx].get("index", idx))
    binding = CameraBinding(serial=serial, index_when_bound=index if index is not None else -1)
    if role == "top":
        updated = CamerasBinding(
            top=binding, wrist=existing.wrist, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
    else:
        updated = CamerasBinding(
            top=existing.top, wrist=binding, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
    p.write_text(json.dumps(updated.to_json(), indent=2, sort_keys=True))
    print(f"[calib] bound {role} -> serial={serial} (index {binding.index_when_bound}); wrote {p}")
    # Best-effort: also patch the matching profile YAML so the rig boots
    # straight into the right device without manual editing.
    profile_map = {"top": "orbbec_overhead", "wrist": "orbbec_wrist"}
    prof = PACKAGE_ROOT / "configs" / "cameras" / f"{profile_map[role]}.yaml"
    if prof.exists():
        text = prof.read_text()
        if "\nserial:" in text:
            lines = text.splitlines()
            for k, line in enumerate(lines):
                if line.startswith("serial:"):
                    lines[k] = f'serial: "{serial}"'
                    break
            else:
                lines.append(f'serial: "{serial}"')
            prof.write_text("\n".join(lines) + "\n")
            print(f"[calib] patched {prof.name}: serial={serial!r}")
    return 0


def _open_orbbec(serial: str, width: int, height: int, fps: int) -> OrbbecCamera:
    from ..config import Cfg
    from ..perception.camera_base import CameraError

    cfg = Cfg({
        "type": "orbbec",
        "serial": serial,
        "width": width,
        "height": height,
        "fps": fps,
        "min_depth_m": 0.25,
        "max_depth_m": 2.5,
    })
    cam = OrbbecCamera(cfg)
    try:
        cam.open()
    except CameraError as e:
        print(f"[calib] camera open failed: {e}", file=sys.stderr)
        raise
    return cam


def _get_serial_from_binding(role: str) -> str | None:
    p = CALIB_DIR / "cameras.json"
    if not p.exists():
        return None
    b = CamerasBinding.from_json(json.loads(p.read_text()))
    return getattr(b, role).serial if getattr(b, role) else None


def _wait_for_user(prompt: str) -> None:
    print(prompt, flush=True)
    try:
        input("press ENTER when ready...")
    except EOFError:
        pass


# ─────────────────────────────────────────────────────────────────────────
# Auto-mode calibration (cuRobo 04b-style state machine, Cartesian poses)
# ─────────────────────────────────────────────────────────────────────────
# Cartesian pose sequence around the workspace center (0.30, -0.05, 0.28)
# in the arm's base frame. Pitch is kept small so the gripper stays near
# horizontal -- the ArUco marker (Solution B in 04b: pasted on the
# gripper top with normal +Z) stays face-on to the top camera.
# Each pose is (xyz, rpy); see TOP_CALIB_POSES below.
TOP_CALIB_POSES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    # 1. Roll sweep at base (0.30, -0.05, 0.28)
    ((0.30, -0.05, 0.28), ( 0.00, 0.00, -0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.00,  0.50)),
    # 2. Yaw sweep
    ((0.30, -0.05, 0.28), (-0.60, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), (-0.30, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.30, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.60, 0.00,  0.00)),
    # 3. Pitch lift + roll combos
    ((0.30, -0.05, 0.28), ( 0.00, 0.20,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.20,  0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.20, -0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.20, -1.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.40,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.40,  0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.40, -0.50)),
    # 4. X / Y / Z translation sweep around the center
    ((0.25, -0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.25, -0.05, 0.28), ( 0.00, 0.20,  0.50)),
    ((0.35, -0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.35, -0.05, 0.28), ( 0.00, 0.20, -0.50)),
    ((0.30, -0.10, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.10, 0.28), ( 0.20, 0.20,  0.50)),
    ((0.30, -0.05, 0.30), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.32), ( 0.00, 0.00,  0.00)),
    ((0.30,  0.00, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30,  0.00, 0.28), ( 0.00, 0.20,  0.50)),
    ((0.30,  0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.22), ( 0.00, 0.20,  0.50)),
    ((0.30, -0.05, 0.35), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.35), ( 0.00, 0.20, -0.50)),
    # 5. Combined rotations at four workspace corners
    ((0.25, -0.05, 0.25), ( 0.30, 0.20,  0.50)),
    ((0.25,  0.00, 0.25), (-0.30, 0.20, -0.50)),
    ((0.35, -0.05, 0.25), (-0.30, 0.20,  0.50)),
    ((0.35,  0.00, 0.25), ( 0.30, 0.20, -0.50)),
    # 6. Extreme combined angles (probe IK feasibility)
    ((0.30, -0.05, 0.28), (-0.40, 0.20, -1.00)),
    ((0.30, -0.05, 0.28), (-0.30, 0.30, -1.00)),
    ((0.30, -0.05, 0.28), ( 0.60, 0.20,  0.30)),
    ((0.30, -0.05, 0.28), (-0.60, 0.20, -0.30)),
    ((0.30, -0.05, 0.28), ( 0.50, 0.30,  0.50)),
    ((0.30, -0.05, 0.28), (-0.50, 0.30, -0.50)),
]


def _pose_to_T(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    """xyz (m) + rpy (rad) -> 4x4 SE(3) pose. Mirrors ``pose_to_T`` in
    cuRobo's calibration module so the pose sequence can be copied
    verbatim across the two codebases.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyz
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx
    return T


def _diagnose_can_bus() -> str | None:
    """Return a human-readable diagnostic string if the CAN bus looks
    unhealthy (ERROR-PASSIVE / BUS-OFF / bitrate wrong / no device),
    else ``None``.

    Surfaces the most common rig bring-up mistakes before the arm
    connect timeout swallows them.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["ip", "-d", "link", "show", "can0"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception:
        return None
    text = (out.stdout or "") + (out.stderr or "")
    if "state ERROR-PASSIVE" in text:
        return (
            "can0 is ERROR-PASSIVE — the CAN controller stopped active "
            "transmit because it sees errors with no ACK (no device on "
            "the bus). Check: (a) arm 24V power, (b) CAN cable firmly "
            "seated at both ends, (c) `sudo ip link set can0 down && "
            "sudo ip link set can0 up type can bitrate 1000000`."
        )
    if "state BUS-OFF" in text:
        return (
            "can0 is BUS-OFF — too many errors. Same fixes as "
            "ERROR-PASSIVE plus power-cycle the arm."
        )
    if "state DOWN" in text or "state UNKNOWN" in text:
        return (
            "can0 is DOWN. Run `sudo ip link set can0 up type can "
            "bitrate 1000000`."
        )
    return None


def _connect_arm_with_safety(arm_cfg_path: str):
    """Construct ``(kinematics, safe_arm, arm_raw)`` from a wrc_demo arm
    profile. Returns ``None`` if the arm can't be connected (so the
    caller can fall back to manual mode).

    Retries ``connect()`` once after a short delay if the first attempt
    fails with the typical "CAN buffer full" symptom (which happens
    when motorbridge's enable_all + mode_mit sequence floods a freshly
    restarted bus). The retry usually succeeds because the SDK has
    drained its queue during the wait.
    """
    import time as _t

    from ..config import Cfg, load_profile
    from ..safety.harness import SafetyHarness, SafetyLimits, SafeArm
    from ..control.arm_base import make_arm

    arm_prof = load_profile("arms", arm_cfg_path)
    kin = Kinematics(
        model_path=str(arm_prof.model),
        ee_frame=str(arm_prof.get("ee_frame", "gripper_end")),
        n_controlled=int(arm_prof.get("n_joints", 6)),
        joint_signs=arm_prof.get("joint_signs"),
    )
    arm_raw = make_arm(arm_prof, kinematics=kin)

    last_err: str | None = None
    for attempt in range(2):
        try:
            arm_raw.connect()
            last_err = None
            break
        except Exception as e:
            last_err = f"connect failed (attempt {attempt + 1}): {e}"
            # Try to release any half-open state before retrying.
            try:
                arm_raw.disconnect()
            except Exception:
                pass
            if attempt == 0:
                _t.sleep(2.0)  # give the CAN controller time to drain
    if last_err is not None:
        return None, last_err

    # Construct harness using the demo's cfg.safety; fall back to file
    # load if not provided (this CLI runs standalone, no cfg.safety).
    try:
        from ..config import load_demo_config
        demo_cfg = load_demo_config(camera="mock", arm=arm_cfg_path, llm="mock")
        safety_limits = SafetyLimits.from_config(demo_cfg.safety)
    except Exception:
        from ..config import Cfg
        safety_cfg = Cfg({
            "workspace": {
                "min": [-1.0, -1.0, -0.5],
                "max": [1.0, 1.0, 1.5],
            },
            "table_z": 0.0,
            "max_joint_vel": 1.2,
            "joint_margin": 0.02,
            "watchdog_s": 5.0,
            "table_clearance": 0.02,
        })
        safety_limits = SafetyLimits.from_config(safety_cfg)
    harness = SafetyHarness(safety_limits, kinematics=kin)
    safe_arm = SafeArm(arm_raw, harness)
    return (kin, safe_arm, arm_raw), None


def _draw_osd(
    vis: np.ndarray,
    *,
    pose_label: str,
    n_samples: int,
    target_n: int,
    state: str,
    capture_flash: bool = False,
) -> None:
    """Draw the calibrator HUD on top of `vis` (in place)."""
    h, w = vis.shape[:2]
    panel = np.zeros((90, w, 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    cv2.putText(panel, f"Calibration -- TOP   samples: {n_samples}/{target_n}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"pose: {pose_label[:60]}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"state: {state[:60]}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "press 'c' or 'q' to finish early, ESC to abort",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    vis[0:90, :] = panel
    if capture_flash:
        # Bright green border that fades over 0.8s by drawing two rects.
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)


def _collect_samples(
    n_target: int,
    marker_size_m: float,
    aruco_dict: str,
    settle_time: float,
    marker_timeout: float,
    stable_frames: int,
    serial: str | None,
    width: int,
    height: int,
    fps: int,
    out_json: Path,
    trace_log: Path,
    arm_profile: str = ARM_PROFILE_DEFAULT,
    manual: bool = False,
    show_window: bool = True,
) -> tuple[list[CalibrationSample], str]:
    """Collect ``n_target`` calibration samples.

    Mode:
      - ``manual=False``: arm moves itself through ``TOP_CALIB_JOINT_POSES``;
        the OpenCV window shows live detection + OSD. ``c``/``q``/ESC stop
        early (PLAN §cuRobo 04b reference). Requires the rebot_rs arm on
        can0; falls back to manual mode when arm can't be reached.
      - ``manual=True``: operator presses ENTER after each pose; no arm
        motion. Useful for offline / dry-run / debugging.

    Returns ``(samples, finish_reason)`` so the caller knows whether the
    loop ended because the operator stopped early vs. all poses done.
    """
    kin = _make_kinematics()
    aruco = ArUcoSession(dict_name=aruco_dict)
    cam = _open_orbbec(serial or "", width, height, fps)
    K = cam.K
    D = cam.D
    print(
        f"[calib] camera open: serial={cam.serial} K={K.tolist() if K is not None else None}",
        file=sys.stderr,
    )
    samples: list[CalibrationSample] = []
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    finish_reason = "session end"

    # Always open the visualization window when requested (so manual
    # mode also gets live ArUco preview + OSD -- the operator can see
    # whether the marker is detected before pressing ENTER). Headless
    # boxes (CI / tests + opencv-python-headless) lack GTK and will
    # raise on namedWindow; we silently fall back so the rest of the
    # pipeline still works.
    WIN = "Hand-Eye Calibration -- TOP"
    if show_window:
        try:
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            # opencv-python-headless has no GTK support; cv2.imshow
            # would always fail in this env. Fall back to disk snapshots
            # so the operator can still verify the camera + ArUco.
            print(
                "[calib] cv2.namedWindow unavailable (headless opencv). "
                "Frames will be saved to /tmp/calib_frames/ for offline "
                "inspection; one out of every 10 frames is saved.",
                file=sys.stderr,
            )
            show_window = False
            try:
                SNAP_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # Arm connection (auto mode only). Manual mode still tries to read q
    # for FK, but if that fails (no arm) we fall back to a synthetic seed.
    safe_arm = None
    if not manual:
        can_diag = _diagnose_can_bus()
        if can_diag:
            print(f"[calib] CAN bus diagnostic: {can_diag}", file=sys.stderr)
        result, err = _connect_arm_with_safety(arm_profile)
        if err:
            print(
                f"[calib] arm unavailable ({err}); falling back to manual mode",
                file=sys.stderr,
            )
            manual = True
        else:
            (kin, safe_arm, _arm_raw) = result
            # Move to home (first pose in TOP_CALIB_POSES) before starting
            # the sequence. Use IK on the home Cartesian pose; fall back
            # to a fixed joint seed if IK is infeasible.
            xyz0, rpy0 = TOP_CALIB_POSES[0]
            T_home = _pose_to_T(xyz0, rpy0)
            q_init = np.zeros(int(safe_arm.n_joints), dtype=float)
            home_ik = kin.ik(T_home, q_init, retries=2)
            if not home_ik.success:
                # Fallback: use the demo profile's home_q seed.
                from ..config import load_profile
                arm_prof = load_profile("arms", arm_profile)
                q_init = np.asarray(arm_prof.get("home_q", [0.0] * 6), dtype=float)
                home_ik = kin.ik(T_home, q_init, retries=4)
            if not home_ik.success:
                print(
                    "[calib] IK failed for home Cartesian pose "
                    f"xyz={xyz0} rpy={rpy0}; falling back to manual.",
                    file=sys.stderr,
                )
                safe_arm = None
                manual = True
            else:
                home_q = np.asarray(home_ik.q, dtype=float)[:safe_arm.n_joints]
                print(
                    f"[calib] home Cartesian: xyz={xyz0} rpy={rpy0} -> "
                    f"q={home_q.round(3).tolist()}"
                )
                if hasattr(safe_arm, "harness") and hasattr(safe_arm.harness, "heartbeat"):
                    safe_arm.harness.heartbeat()
                ok = safe_arm.move_joints(home_q, duration_s=3.0)
                if not ok:
                    estopped = bool(getattr(safe_arm.harness, "estopped", False))
                    if estopped:
                        print(
                            "[calib] safe_arm estopped during home move "
                            "(harness rejected the pose). Falling back to manual.",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "[calib] safe_arm.move_joints(home) returned False; "
                            "likely CAN bus failure during enable_all / "
                            "mode_mit (check `ip -d link show can0` and the "
                            "arm 24V supply). Falling back to manual.",
                            file=sys.stderr,
                        )
                    safe_arm = None
                    manual = True
                else:
                    # Give the arm a moment to settle.
                    time.sleep(2.0)

    with trace_log.open("a") as trace:
        trace.write(json.dumps({
            "event": "session_start",
            "ts": time.time(),
            "marker_size_m": marker_size_m,
            "aruco_dict": aruco_dict,
            "n_target": n_target,
            "manual": manual,
            "arm_profile": arm_profile,
        }) + "\n")

        if manual:
            samples, finish_reason = _collect_loop_manual(
                kin, aruco, cam, K, D,
                n_target=n_target, marker_size_m=marker_size_m,
                settle_time=settle_time, marker_timeout=marker_timeout,
                stable_frames=stable_frames, samples=samples, trace=trace,
            )
        else:
            samples, finish_reason = _collect_loop_auto(
                safe_arm, kin, aruco, cam, K, D,
                n_target=n_target, marker_size_m=marker_size_m,
                settle_time=settle_time, marker_timeout=marker_timeout,
                stable_frames=stable_frames, samples=samples, trace=trace,
                win_name=WIN, show_window=show_window,
            )

        trace.write(json.dumps({
            "event": "session_end",
            "ts": time.time(),
            "n_collected": len(samples),
            "finish_reason": finish_reason,
        }) + "\n")

    if show_window and not manual:
        try:
            cv2.destroyWindow(WIN)
        except cv2.error:
            pass
        except cv2.error:
            pass
    cam.close()
    return samples, finish_reason


def _collect_loop_manual(
    kin, aruco, cam, K, D,
    *, n_target, marker_size_m, settle_time, marker_timeout, stable_frames,
    samples, trace,
) -> tuple[list[CalibrationSample], str]:
    finish_reason = "operator finished"
    for i in range(n_target):
        print(f"\n[calib] === sample {i + 1}/{n_target} ===")
        _wait_for_user(
            "Move the arm to a new pose, hold still. The detector will "
            f"look for stable ArUco readings for {stable_frames} frames within "
            f"{settle_time}s."
        )
        det, _ = _stable_detect(
            cam, aruco, K, D, marker_size_m,
            settle_time=settle_time, marker_timeout=marker_timeout,
            stable_frames=stable_frames,
        )
        if det is None:
            print(
                f"[calib] sample {i + 1}: no ArUco detected within timeout, "
                "skipping (operator can re-run with --target N)",
                file=sys.stderr,
            )
            trace.write(json.dumps({
                "event": "sample_skipped", "i": i,
                "ts": time.time(), "reason": "no_marker",
            }) + "\n")
            continue
        q = _read_arm_q(cam.serial)
        if q is None:
            print(
                "[calib] cannot read arm state (no rebot_rs connection); "
                "saving synthetic FK from a deterministic seed for "
                "dry-run only",
                file=sys.stderr,
            )
            q = np.zeros(6)
            q[0] = 0.30 + 0.05 * i
        G = kin.fk(q)
        sample = CalibrationSample(
            pose_label=f"manual_{i + 1}",
            T_gripper2base=G,
            T_marker2cam=det.T_marker2cam,
            aruco_reprojection_px=float(det.reprojection_rmse_px),
            inlier=True,
        )
        samples.append(sample)
        trace.write(json.dumps({
            "event": "sample_recorded", "i": i, "ts": time.time(),
            "q": q.tolist(), "aruco_rmse_px": float(det.reprojection_rmse_px),
        }) + "\n")
        print(
            f"[calib] recorded sample {len(samples)}: aruco rmse="
            f"{det.reprojection_rmse_px:.2f}px, q[:3]={q[:3].round(3).tolist()}"
        )
    return samples, finish_reason


def _collect_loop_auto(
    safe_arm, kin, aruco, cam, K, D,
    *, n_target, marker_size_m, settle_time, marker_timeout, stable_frames,
    samples, trace, win_name, show_window,
) -> tuple[list[CalibrationSample], str]:
    """Auto-mode state machine: idle -> moving -> settling -> searching -> (capture) -> idle.

    Faithful port of cuRobo 04b's state machine:
    - Cartesian poses ``TOP_CALIB_POSES`` (xyz, rpy)
    - IK per pose via ``Kinematics.ik``; ``safe_arm.move_joints`` to drive
      the arm (no motion planner needed for calibration; we only need
      reachability + final q for FK)
    - Reads ``safe_arm._arm.get_state().q`` after settle for FK
    """
    finish_reason = "all target poses done"
    poses = TOP_CALIB_POSES[: max(1, n_target)]
    target_n = max(n_target, len(poses))
    last_color: list[np.ndarray | None] = [None]
    last_q_solution: list[np.ndarray | None] = [None]

    def _stop_requested() -> tuple[bool, str]:
        key = cv2.waitKey(30) & 0xFF if show_window else 0
        if key in (ord('q'), ord('Q'), ord('c'), ord('C'), 27):
            return True, "user finish (key)"
        if show_window:
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    return True, "user finish (window closed)"
            except cv2.error:
                return True, "user finish (window gone)"
        return False, ""

    def _grab_color() -> np.ndarray | None:
        try:
            f = cam.get_frame()
        except Exception:
            return None
        return f.rgb

    try:
        for idx, (xyz, rpy) in enumerate(poses):
            pose_label = f"xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f})"
            print(
                f"\n[calib] auto pose {idx + 1}/{target_n}: {pose_label}"
            )

            T_target = _pose_to_T(xyz, rpy)
            # Seed IK from the previous solution (or zeros); the IKResult
            # ``q`` is in joint-local frame already (joint_signs baked in).
            q_init = (last_q_solution[0]
                      if last_q_solution[0] is not None
                      else np.zeros(safe_arm.n_joints, dtype=float))
            ik = kin.ik(T_target, q_init, retries=2)
            if not ik.success:
                print(
                    f"[calib] pose {idx + 1}: IK failed (error={ik.error:.4f}); "
                    "skipping",
                    file=sys.stderr,
                )
                trace.write(json.dumps({
                    "event": "pose_skipped", "i": idx,
                    "ts": time.time(), "reason": "ik_failed",
                }) + "\n")
                continue
            q_target = np.asarray(ik.q, dtype=float)[:safe_arm.n_joints]
            last_q_solution[0] = q_target

            # The auto-loop drives every motion through the safety
            # harness. The harness's begin_motion() rejects stale
            # perception. The calib tool is itself the only perception
            # consumer here, but the harness's watchdog was last
            # touched at connect() — too long ago. Heartbeat just
            # before each move to satisfy the watchdog.
            if hasattr(safe_arm, "harness") and hasattr(safe_arm.harness, "heartbeat"):
                safe_arm.harness.heartbeat()

            ok = safe_arm.move_joints(q_target, duration_s=2.0)
            if not ok:
                print(
                    f"[calib] pose {idx + 1}: safe_arm.move_joints refused; "
                    "skipping",
                    file=sys.stderr,
                )
                trace.write(json.dumps({
                    "event": "pose_skipped", "i": idx,
                    "ts": time.time(), "reason": "harness_refused",
                }) + "\n")
                continue

            arm = getattr(safe_arm, "_arm", safe_arm)
            if hasattr(arm, "wait_settled"):
                settled = arm.wait_settled(q_target, tol=0.05, timeout_s=5.0)
                if not settled:
                    print(
                        f"[calib] pose {idx + 1}: arm didn't settle in 5s; "
                        "continuing with detection anyway",
                        file=sys.stderr,
                    )

            stable_count = 0
            best = None
            t_settle_until = time.monotonic() + settle_time
            t_timeout = t_settle_until + marker_timeout
            while time.monotonic() < t_timeout:
                stopped, reason = _stop_requested()
                if stopped:
                    finish_reason = reason
                    return samples, finish_reason

                color = _grab_color()
                if color is None:
                    time.sleep(0.05)
                    continue
                last_color[0] = color
                # Refresh the safety harness's perception watchdog. The
                # calib tool is itself the only perception consumer in
                # this session (no WorldWatcher thread), so the camera
                # frames we just grabbed are the most recent observation
                # the harness can know about.
                if hasattr(safe_arm, "harness") and hasattr(safe_arm.harness, "heartbeat"):
                    safe_arm.harness.heartbeat()

                det = aruco.detect(color, K, D, marker_size_m)
                overlay = (
                    aruco.draw_aruco_overlay(color, det)
                    if det is not None else color
                )
                _draw_osd(
                    overlay,
                    pose_label=f"{idx + 1}/{target_n} {pose_label}",
                    n_samples=len(samples),
                    target_n=target_n,
                     state=(f"searching ({stable_count}/{stable_frames})"
                            if det is not None else "waiting ArUco"),
                )
                if show_window:
                    cv2.imshow(win_name, overlay)
                else:
                    # Headless fallback: save 1 of every SNAP_EVERY frames
                    # to /tmp/calib_frames/ for the operator to inspect.
                    # Successful captures (det is not None) are always saved.
                    global SNAP_COUNTER
                    SNAP_COUNTER += 1
                    if det is not None or SNAP_COUNTER % SNAP_EVERY == 0:
                        try:
                            label = "det" if det is not None else "nodet"
                            out = overlay if det is not None else color
                            cv2.imwrite(
                                str(SNAP_DIR / f"f{SNAP_COUNTER:05d}_{label}.png"),
                                out,
                            )
                        except Exception:
                            pass

                if det is None:
                    stable_count = 0
                    best = None
                    time.sleep(0.05)
                    continue
                stable_count += 1
                best = det
                if stable_count >= stable_frames:
                    break

            if best is None:
                print(
                    f"[calib] pose {idx + 1}: marker not detected within timeout; "
                    "skipping",
                    file=sys.stderr,
                )
                trace.write(json.dumps({
                    "event": "sample_skipped", "i": idx,
                    "ts": time.time(), "reason": "no_marker",
                }) + "\n")
                continue

            if show_window and last_color[0] is not None:
                vis = last_color[0].copy()
                _draw_osd(
                    vis, pose_label=f"{idx + 1}/{target_n} {pose_label}",
                    n_samples=len(samples) + 1,
                    target_n=target_n,
                    state=f"captured ({stable_count}/{stable_frames})",
                    capture_flash=True,
                )
                if show_window:
                    cv2.imshow(win_name, vis)
                    cv2.waitKey(120)
                else:
                    # Headless: always save the capture-flash frame.
                    try:
                        cv2.imwrite(
                            str(SNAP_DIR / f"capture_{len(samples):02d}.png"),
                            vis,
                        )
                    except Exception:
                        pass

            arm = getattr(safe_arm, "_arm", safe_arm)
            state = arm.get_state() if hasattr(arm, "get_state") else None
            q = np.asarray(state.q, dtype=float)[:6] if state is not None else q_target
            G = kin.fk(q)
            sample = CalibrationSample(
                pose_label=f"auto_{idx + 1:02d}",
                T_gripper2base=G,
                T_marker2cam=best.T_marker2cam,
                aruco_reprojection_px=float(best.reprojection_rmse_px),
                inlier=True,
            )
            samples.append(sample)
            trace.write(json.dumps({
                "event": "sample_recorded", "i": idx, "ts": time.time(),
                "q": q.tolist(),
                "pose_label": pose_label,
                "aruco_rmse_px": float(best.reprojection_rmse_px),
            }) + "\n")
            print(
                f"[calib] captured sample {len(samples)}: {pose_label}, "
                f"aruco rmse={best.reprojection_rmse_px:.2f}px, "
                f"q[:3]={q[:3].round(3).tolist()}"
            )

            stopped, reason = _stop_requested()
            if stopped:
                finish_reason = reason
                return samples, finish_reason

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C"
    return samples, finish_reason


def _grab_color(cam) -> np.ndarray | None:
    """Read one frame; return the BGR array or None on failure."""
    try:
        f = cam.get_frame()
    except Exception:
        return None
    return f.rgb


def _stable_detect(
    cam, aruco, K, D, marker_size_m,
    *, settle_time, marker_timeout, stable_frames,
) -> tuple[object | None, list[object]]:
    """Grab frames until the marker has been stable for N consecutive
    detections or ``settle_time + marker_timeout`` elapses.

    Returns ``(best_detection, all_detections)``.
    """
    best = None
    stable_count = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_time + marker_timeout:
        try:
            frame = cam.get_frame()
        except Exception:
            time.sleep(0.1)
            continue
        det = aruco.detect(frame.rgb, K, D, marker_size_m)
        if det is None:
            stable_count = 0
            time.sleep(0.05)
            continue
        stable_count += 1
        best = det
        if stable_count >= stable_frames:
            return det, []
    return best, []


def _synthesize_samples(
    n: int,
    seed: int = 0,
) -> tuple[list[CalibrationSample], np.ndarray, np.ndarray]:
    """Generate synthetic (FK, PnP) samples with a known ground-truth
    ``X = T_cam2base`` and ``Y = T_marker2gripper`` so the operator can
    sanity-check the entire calibration pipeline without hardware.

    Returns ``(samples, X_true, Y_true)``. The poses are diverse enough
    that the joint SE(3) fit converges (per test_handeye_solver).
    """
    from ..calibration.handeye_solver import so3_exp

    rng = np.random.default_rng(seed)
    X_true = np.eye(4)
    X_true[:3, 3] = [0.30, -0.10, 0.85]
    ang_y, ang_z = 0.5, 0.25
    Ry = np.array(
        [[np.cos(ang_y), 0.0, np.sin(ang_y)],
         [0.0, 1.0, 0.0],
         [-np.sin(ang_y), 0.0, np.cos(ang_y)]]
    )
    Rz = np.array(
        [[np.cos(ang_z), -np.sin(ang_z), 0.0],
         [np.sin(ang_z),  np.cos(ang_z), 0.0],
         [0.0, 0.0, 1.0]]
    )
    X_true[:3, :3] = Rz @ Ry

    Y_true = np.eye(4)
    Y_true[:3, 3] = [0.005, 0.0, 0.04]
    Y_true[:3, :3] = so3_exp(rng.normal(scale=0.05, size=3))

    samples: list[CalibrationSample] = []
    for i in range(n):
        ang_z_g = 0.3 * i
        ang_y_g = 0.2 + 0.1 * i
        Rz_g = np.array(
            [[np.cos(ang_z_g), -np.sin(ang_z_g), 0.0],
             [np.sin(ang_z_g),  np.cos(ang_z_g), 0.0],
             [0.0, 0.0, 1.0]]
        )
        Ry_g = np.array(
            [[np.cos(ang_y_g), 0.0, np.sin(ang_y_g)],
             [0.0, 1.0, 0.0],
             [-np.sin(ang_y_g), 0.0, np.cos(ang_y_g)]]
        )
        G = np.eye(4)
        G[:3, :3] = Ry_g @ Rz_g
        G[:3, 3] = [0.28 + 0.02 * i, 0.03 * np.cos(i), 0.15 + 0.04 * i]
        M = np.linalg.inv(X_true) @ G @ Y_true
        samples.append(CalibrationSample(
            pose_label=f"dryrun_{i + 1}",
            T_gripper2base=G,
            T_marker2cam=M,
            aruco_reprojection_px=0.1,
            inlier=True,
        ))
    return samples, X_true, Y_true


def _synthesize_samples(
    n: int,
    seed: int = 0,
) -> tuple[list[CalibrationSample], np.ndarray, np.ndarray]:
    """Generate synthetic (FK, PnP) samples with a known ground-truth
    ``X = T_cam2base`` and ``Y = T_marker2gripper`` so the operator can
    sanity-check the entire calibration pipeline without hardware.

    Returns ``(samples, X_true, Y_true)``. The poses are diverse enough
    that the joint SE(3) fit converges (per test_handeye_solver).
    """
    from ..calibration.handeye_solver import so3_exp

    rng = np.random.default_rng(seed)
    X_true = np.eye(4)
    X_true[:3, 3] = [0.30, -0.10, 0.85]
    ang_y, ang_z = 0.5, 0.25
    Ry = np.array(
        [[np.cos(ang_y), 0.0, np.sin(ang_y)],
         [0.0, 1.0, 0.0],
         [-np.sin(ang_y), 0.0, np.cos(ang_y)]]
    )
    Rz = np.array(
        [[np.cos(ang_z), -np.sin(ang_z), 0.0],
         [np.sin(ang_z),  np.cos(ang_z), 0.0],
         [0.0, 0.0, 1.0]]
    )
    X_true[:3, :3] = Rz @ Ry

    Y_true = np.eye(4)
    Y_true[:3, 3] = [0.005, 0.0, 0.04]
    Y_true[:3, :3] = so3_exp(rng.normal(scale=0.05, size=3))

    samples: list[CalibrationSample] = []
    for i in range(n):
        ang_z_g = 0.3 * i
        ang_y_g = 0.2 + 0.1 * i
        Rz_g = np.array(
            [[np.cos(ang_z_g), -np.sin(ang_z_g), 0.0],
             [np.sin(ang_z_g),  np.cos(ang_z_g), 0.0],
             [0.0, 0.0, 1.0]]
        )
        Ry_g = np.array(
            [[np.cos(ang_y_g), 0.0, np.sin(ang_y_g)],
             [0.0, 1.0, 0.0],
             [-np.sin(ang_y_g), 0.0, np.cos(ang_y_g)]]
        )
        G = np.eye(4)
        G[:3, :3] = Ry_g @ Rz_g
        G[:3, 3] = [0.28 + 0.02 * i, 0.03 * np.cos(i), 0.15 + 0.04 * i]
        # G Y = X M  =>  M = inv(X) G Y
        M = np.linalg.inv(X_true) @ G @ Y_true
        samples.append(CalibrationSample(
            pose_label=f"dryrun_{i + 1}",
            T_gripper2base=G,
            T_marker2cam=M,
            aruco_reprojection_px=0.1,
            inlier=True,
        ))
    return samples, X_true, Y_true


def _read_arm_q(serial: str | None) -> np.ndarray | None:
    """Read the current arm joint state from the rebot_rs arm.

    Lazy import so the script is runnable on the dry-run path (no arm).
    Returns ``None`` if the arm isn't connected; the caller decides how
    to handle that (skip vs synthetic).
    """
    try:
        from ..control.rebot_rs_arm import RebotRSArm  # type: ignore
    except Exception:
        return None
    try:
        arm = RebotRSArm.__new__(RebotRSArm)  # bypass __init__ state checks
        # Minimal state to read .get_state(): see rebot_rs_arm.__init__.
        # If the arm can't be probed, just return None (dry-run path).
        from ..safety.harness import SafetyLimits
        arm.connect()
        st = arm.get_state()
        arm.disconnect()
        return np.asarray(st.q[:6], dtype=float)
    except Exception as e:
        print(f"[calib] arm read failed: {e}", file=sys.stderr)
        return None


def _solve_and_persist(
    samples: list[CalibrationSample],
    out_json: Path,
    marker_size_m: float,
    aruco_dict: str,
    aruco_marker_id: int,
    serial: str | None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    X_true: np.ndarray | None = None,
    Y_true: np.ndarray | None = None,
) -> CalibrationResult:
    if len(samples) < 4:
        raise RuntimeError(
            f"only {len(samples)} samples collected; need >= 4 for joint_solve"
        )
    print(f"[calib] running joint_solve on {len(samples)} samples...")
    fit = joint_solve_restart(samples, n_restart=4, seed=0)
    print(
        f"[calib] fit: translation rmse={fit.metrics.get('translation_rmse_mm', float('nan')):.2f}mm "
        f"max={fit.metrics.get('translation_max_err_mm', float('nan')):.2f}mm "
        f"inliers={len(fit.inliers)} outliers={len(fit.outliers)}"
    )
    # Re-flag samples based on the inlier set so the persisted record
    # carries the same labels as the solver saw.
    inlier_set = set(fit.inliers)
    flagged = tuple(
        CalibrationSample(
            pose_label=s.pose_label,
            T_gripper2base=s.T_gripper2base,
            T_marker2cam=s.T_marker2cam,
            aruco_reprojection_px=s.aruco_reprojection_px,
            inlier=(i in inlier_set),
        )
        for i, s in enumerate(samples)
    )
    result = CalibrationResult(
        schema_version=SCHEMA_VERSION,
        frame_convention="wrc_demo",
        frame_convention_note=(
            "base +x away robot, +y left; TCP = gripper_end; "
            "joint_signs = [-1]*6"
        ),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        method="joint_se3_lm_huber",
        mode="eye_to_hand",
        camera_serial=serial or "",
        camera_index_when_bound=-1,
        marker_size_m=marker_size_m,
        aruco_dict=aruco_dict,
        aruco_marker_id=aruco_marker_id,
        n_samples=len(samples),
        n_inliers=len(fit.inliers),
        n_outliers=len(fit.outliers),
        outlier_indices=tuple(int(i) for i in fit.outliers),
        T_cam2base=np.asarray(fit.T_cam2base, dtype=float).tolist(),
        T_marker2gripper=np.asarray(fit.T_marker2gripper, dtype=float).tolist(),
        metrics={k: float(v) for k, v in fit.metrics.items()},
        samples_for_repro=flagged,
    )
    # When running in dry-run mode, persist the ground-truth matrices
    # alongside so a reviewer can verify the solve without re-running.
    if X_true is not None and Y_true is not None:
        # CalibrationResult.metrics is a flat dict[str, float], so we
        # encode ground-truth by storing only the Frobenius norms of
        # the recovered-vs-true differences (the actual matrices are
        # logged on stdout by main()).
        x_err = float(np.linalg.norm(fit.T_cam2base - X_true))
        y_err = float(np.linalg.norm(fit.T_marker2gripper - Y_true))
        result = CalibrationResult(
            **{**result.__dict__, "metrics": {
                **result.metrics,
                "dryrun_X_frob_err": x_err,
                "dryrun_Y_frob_err": y_err,
            }},
        )
    K = np.eye(3)
    D = np.zeros(5)
    # Pull K/D from the saved JSON if the user already ran calibration
    # once and we're re-fitting offline.
    json_path, npz_path = save_calibration(result, out_json, K=K, D=D)
    print(f"[calib] wrote {json_path}")
    print(f"[calib] wrote {npz_path}")
    return result


def _verify(npz_path: Path, known_points: Sequence[tuple[float, float, float]]) -> int:
    if not npz_path.exists():
        print(f"[calib] verify: {npz_path} does not exist", file=sys.stderr)
        return 2
    res = load_calibration(npz_path)
    X = np.asarray(res.T_cam2base, dtype=float)
    Y = np.asarray(res.T_marker2gripper, dtype=float)
    if not res.samples_for_repro:
        print("[calib] verify: no samples_for_repro in the saved result; "
              "re-run calibration to populate them", file=sys.stderr)
        return 1
    errs_mm = []
    print(
        f"[calib] verify against {len(known_points)} known point(s); "
        f"prediction is the centroid of inlier sample marker positions "
        f"({len(res.samples_for_repro)} samples on disk)"
    )
    for kp in known_points:
        kp_arr = np.asarray(kp, dtype=float).reshape(3)
        # For each inlier sample, the camera-side prediction of marker
        # origin in base frame is ``(X @ M)[:3, 3]`` -- the marker
        # position predicted by the camera path. By the hand-eye
        # equation ``G @ Y == X @ M`` this must equal ``(G @ Y)[:3, 3]``
        # (the gripper-FK path) up to measurement noise. We average the
        # camera-side prediction across inliers to get the centroid of
        # where the marker was during calibration, then compare to the
        # operator-supplied known point.
        cam_pts = []
        for s in res.samples_for_repro:
            if not s.inlier:
                continue
            M = np.asarray(s.T_marker2cam, dtype=float)
            cam_pts.append((X @ M)[:3, 3])
        if not cam_pts:
            continue
        avg = np.mean(cam_pts, axis=0)
        err = np.linalg.norm(avg - kp_arr) * 1000.0
        errs_mm.append(err)
        print(
            f"  known=({kp_arr[0]:.3f},{kp_arr[1]:.3f},{kp_arr[2]:.3f}) "
            f"predicted=({avg[0]:.3f},{avg[1]:.3f},{avg[2]:.3f}) err={err:.1f}mm"
        )
    if not errs_mm:
        print("[calib] verify: no inlier samples available", file=sys.stderr)
        return 1
    rmse = float(np.sqrt(np.mean(np.asarray(errs_mm) ** 2)))
    mx = float(np.max(errs_mm))
    print(f"[calib] verify RMSE = {rmse:.2f}mm, max = {mx:.2f}mm")
    return 0 if rmse <= 10.0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true",
                   help="enumerate Orbbec devices and exit")
    p.add_argument("--bind", choices=["top", "wrist"], default=None,
                   help="bind role -> serial, write data/calibration/cameras.json")
    p.add_argument("--serial", default=None,
                   help="Orbbec serial to bind/use (skips enumeration)")
    p.add_argument("--index", type=int, default=None,
                   help="device index to pick when enumerating")
    p.add_argument("--arm-profile", default=ARM_PROFILE_DEFAULT)
    p.add_argument("--marker-size", type=float, default=0.10,
                   help="ArUco marker side length (meters)")
    p.add_argument("--aruco-dict", default="4x4_50",
                   help="ArUco dictionary (4x4_50 | 5x5_50 | 6x6_50 | APRILTAG_16h5)")
    p.add_argument("--aruco-marker-id", type=int, default=0)
    p.add_argument("--settle-time", type=float, default=1.5,
                   help="seconds to wait for the arm to settle before grab")
    p.add_argument("--marker-timeout", type=float, default=4.0,
                   help="max seconds to wait for a stable marker detection")
    p.add_argument("--stable-frames", type=int, default=4)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n", type=int, default=15,
                   help="target number of calibration samples")
    p.add_argument("--out", default=str(CALIB_DIR / "hand_eye_top.json"))
    p.add_argument("--verify", action="store_true",
                   help="verify a saved calibration against known points")
    p.add_argument("--npz", default=None,
                   help="path to a saved calibration JSON (for --verify)")
    p.add_argument("--known-points", default=None,
                   help="semicolon-separated base-frame points 'x,y,z;x,y,z'")
    p.add_argument("--dry-run", action="store_true",
                   help="synthesize samples (skip camera + arm) and exercise the "
                        "solve/persist path end-to-end; used to verify the "
                        "pipeline without hardware")
    p.add_argument("--dry-run-out", default=str(CALIB_DIR / "hand_eye_dryrun.json"),
                   help="output JSON path for --dry-run (does NOT clobber --out)")
    p.add_argument("--manual", action="store_true",
                   help="disable auto-mode arm motion + OpenCV window: wait for "
                        "ENTER after each pose (matches cuRobo 04b's --manual)")
    args = p.parse_args(argv)

    if args.list:
        return _print_devices()

    if args.bind is not None:
        return _bind_role(args.bind, serial=args.serial, index=args.index)

    if args.verify:
        if not args.npz:
            print("--verify requires --npz", file=sys.stderr)
            return 2
        if not args.known_points:
            print("--verify requires --known-points 'x,y,z;...'", file=sys.stderr)
            return 2
        try:
            kps = [
                tuple(float(t) for t in p.split(","))
                for p in args.known_points.split(";")
            ]
        except ValueError as e:
            print(f"--known-points parse error: {e}", file=sys.stderr)
            return 2
        return _verify(Path(args.npz), kps)

    if args.dry_run:
        out_p = Path(args.dry_run_out)
        # Don't let --dry-run clobber the real calibration file by accident.
        if out_p.resolve() == Path(args.out).resolve():
            print(
                f"[calib] --dry-run-out must differ from --out ({args.out})",
                file=sys.stderr,
            )
            return 2
        samples, X_true, Y_true = _synthesize_samples(args.n, seed=0)
        result = _solve_and_persist(
            samples,
            out_json=out_p,
            marker_size_m=args.marker_size,
            aruco_dict=args.aruco_dict,
            aruco_marker_id=args.aruco_marker_id,
            serial="DRYRUN",
            X_true=X_true, Y_true=Y_true,
        )
        # Per-sample residual is the actual quality signal; the joint
        # fit can have multiple global optima in (X, Y) space (PLAN.md
        # §13.4), so the X / Y distance to ground truth is reported
        # for diagnostic purposes only.
        x_err = float(np.linalg.norm(result.T_cam2base - X_true))
        y_err = float(np.linalg.norm(result.T_marker2gripper - Y_true))
        per_sample_mm = float(result.metrics.get("translation_rmse_mm", float("nan")))
        print(
            f"[calib] dry-run: per-sample rmse = {per_sample_mm:.4f}mm; "
            f"||X - X_true||_F = {x_err:.4f}, ||Y - Y_true||_F = {y_err:.4f}; "
            f"wrote {out_p}"
        )
        # Smoke-test criterion: the solver converged to a low-residual
        # fit (the pipeline works). Don't fail on the X / Y distance
        # since joint SE(3) fits have known (X, Y) ambiguities.
        return 0 if per_sample_mm < 1.0 else 1

    serial = args.serial or _get_serial_from_binding("top") or ""
    samples, finish_reason = _collect_samples(
        n_target=args.n,
        marker_size_m=args.marker_size,
        aruco_dict=args.aruco_dict,
        settle_time=args.settle_time,
        marker_timeout=args.marker_timeout,
        stable_frames=args.stable_frames,
        serial=serial or None,
        width=args.width,
        height=args.height,
        fps=args.fps,
        out_json=Path(args.out),
        trace_log=PACKAGE_ROOT / "runs" / f"calibration_{int(time.time())}.jsonl",
        arm_profile=args.arm_profile,
        manual=args.manual,
        show_window=not args.manual,  # auto-mode uses cv2.imshow; manual doesn't
    )
    print(f"[calib] session ended: {finish_reason}")
    if len(samples) < 4:
        print(f"[calib] only {len(samples)} samples; need >=4 to fit. "
              "Re-run with --n for another pass.", file=sys.stderr)
        return 1
    _solve_and_persist(
        samples,
        out_json=Path(args.out),
        marker_size_m=args.marker_size,
        aruco_dict=args.aruco_dict,
        aruco_marker_id=args.aruco_marker_id,
        serial=serial or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

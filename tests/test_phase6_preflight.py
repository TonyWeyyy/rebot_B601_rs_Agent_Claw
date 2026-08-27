"""Phase 6 pre-flight checks before real robot demo.

Run this on the wrc_demo rig (with rebot_rs + Orbbec Gemini2 connected)
BEFORE the actual demo. Fails fast on any missing dependency / config.

Run with:
    /home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/test_phase6_preflight.py -v --no-header

Hardware checks (can0 UP, orbbec serial) FAIL on non-rig machines —
that's expected. The tests still tell you which config-side items
are missing.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _ok(cond, msg):
    return cond, msg


# ──────────── Python env ────────────


def test_python_is_conda_DM_env():
    """The 'DM' conda env carries motorbridge + pyorbbecsdk — see README §Setup.

    Accepts any prefix that resolves to a conda env named 'DM' (the project
    convention; install location doesn't matter for new checkouts)."""
    # sys.executable looks like '<prefix>/envs/DM/bin/python' on conda; 'DM'
    # may appear in the basename too. We check for the env-name suffix.
    exe = Path(sys.executable).resolve()
    parts = exe.parts
    assert "envs" in parts and "DM" in parts, (
        f"Wrong Python: {sys.executable}. Activate the DM conda env "
        f"(`conda activate DM` or `conda run -n DM python ...`) and re-run."
    )


def test_motorbridge_present():
    """motorbridge drives the rebot_rs CAN bus."""
    import motorbridge  # noqa: F401


def test_ultralytics_8_4():
    """ultralytics 8.4.x for YOLOE-26L-seg."""
    import ultralytics
    assert ultralytics.__version__.startswith("8.4"), ultralytics.__version__


def test_pyorbbecsdk_present():
    """pyorbbecsdk drives the Orbbec Gemini2."""
    import pyorbbecsdk
    assert hasattr(pyorbbecsdk, "Pipeline")


def test_pinocchio_present():
    """pinocchio for IK + rpy conversions."""
    import pinocchio  # noqa: F401


# ──────────── wrc_demo native gripper + bridge ────────────


def test_wrc_demo_native_gripper_module():
    """WrcGripper is wrc_demo's port of rebot_grasp's GraspDriver (ADR-0005)."""
    from wrc_demo.control.gripper import WrcGripper  # noqa: F401


def test_wrc_demo_bridge_module():
    from wrc_demo.skills.rebot_grasp_bridge import RebotGraspBridge  # noqa: F401


# ──────────── Config files ────────────


def test_hand_eye_top_json_exists():
    assert (REPO / "data/calibration/hand_eye_top.json").exists()


def test_rebot_rs_arm_yaml_exists():
    assert (REPO / "configs/arms/rebot_rs.yaml").exists()


def test_orbbec_overhead_camera_yaml_exists():
    assert (REPO / "configs/cameras/orbbec_overhead.yaml").exists()


def test_hand_eye_top_z_baked_negative_0_02():
    """Verify the -0.02 compensation is still baked into T_cam2base.

    The full 4x4 matrix is stored under the ``T_cam2base`` key; the z
    translation should be ~0.609 (= 0.629 nominal camera height − 0.02
    compensation). If this is off, the v5 calibration that produces
    xy < 1 mm is broken and the demo will visibly mismatch set.py.
    """
    data = json.loads((REPO / "data/calibration/hand_eye_top.json").read_text())
    T = data["T_cam2base"]
    z = T[2][3]
    assert abs(z - 0.609) < 0.001, (
        f"T_cam2base z = {z}, expected ~0.609 (baked -0.02). "
        f"See docs/adr/0003-hand-eye-compensation-delta.md."
    )


def test_hand_eye_compensation_recorded():
    """The JSON should record the -0.02 compensation as documentation."""
    data = json.loads((REPO / "data/calibration/hand_eye_top.json").read_text())
    comp = data.get("hand_eye_compensation_m_applied", {})
    assert comp.get("z") == -0.02, (
        f"hand_eye_compensation_m_applied.z = {comp.get('z')}, "
        f"expected -0.02. See ADR-0003."
    )


def test_camera_serial_matches():
    """camera_serial in hand_eye_top.json must match cameras.json so
    make_arm binds to the same physical camera that the calibration
    was done with.
    """
    he = json.loads((REPO / "data/calibration/hand_eye_top.json").read_text())
    he_serial = he.get("camera_serial")

    cams_path = REPO / "data/calibration/cameras.json"
    if not cams_path.exists():
        pytest.skip("cameras.json not present (fresh clone)")
    cams = json.loads(cams_path.read_text())
    orbbec = next(
        (c for c in cams.get("cameras", []) if "orbbec" in c.get("type", "").lower()),
        None,
    )
    if orbbec is None:
        pytest.skip("no orbbec camera in cameras.json")
    assert orbbec.get("serial") == he_serial, (
        f"camera serial mismatch: hand_eye={he_serial!r}, "
        f"cameras.json={orbbec.get('serial')!r}"
    )


# ──────────── ADRs (Phase 6 deliverables) ────────────


@pytest.mark.parametrize("adr", [
    "0001-ready-pose-uses-wrc-home-q.md",
    "0002-grasp-force-1.0.md",
    "0003-hand-eye-compensation-delta.md",
    "0004-drop-gripper-config.md",
])
def test_adr_exists(adr):
    p = REPO / "docs/adr" / adr
    assert p.exists(), f"ADR missing: {p}"


# ──────────── Hardware (Linux) ────────────
# These FAIL on non-rig machines. They are gated by an environment
# variable so they only run when explicitly enabled (Phase 6
# runbook). Set ``WRC_PHASE6_HARDWARE=1`` on the rig to enable.


def _rig_only():
    """Skip unless the rig-only flag is set.

    The flag is opt-in: pre-flight tests on a dev machine skip the
    hardware probes; on the rig, the operator sets the flag in the
    shell that runs the pre-flight before powering the motors.
    """
    import os
    if os.environ.get("WRC_PHASE6_HARDWARE") != "1":
        pytest.skip(
            "hardware probes disabled (set WRC_PHASE6_HARDWARE=1 on the rig)"
        )


def test_can0_link_up():
    """CAN bus must be UP for rebot_rs."""
    _rig_only()
    r = subprocess.run(["ip", "link", "show", "can0"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"can0 not present: {r.stderr}")
    if "state UP" not in r.stdout:
        pytest.fail(f"can0 not UP: {r.stdout}")


def test_orbbec_serial_present():
    """Orbbec camera with the calibrated serial must be configured."""
    _rig_only()
    cams_path = REPO / "data/calibration/cameras.json"
    if not cams_path.exists():
        pytest.fail("cameras.json missing")
    cams = json.loads(cams_path.read_text())
    orbbec = next(
        (c for c in cams.get("cameras", []) if "orbbec" in c.get("type", "").lower()),
        None,
    )
    if orbbec is None:
        pytest.fail("no orbbec camera in cameras.json")
    assert "AY3Z331006L" in orbbec.get("serial", ""), orbbec.get("serial")

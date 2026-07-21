"""Tests for Extrinsics.from_config: new JSON, cuRobo NPZ, legacy NPZ, inline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from wrc_demo.calibration import CalibrationResult, save_calibration
from wrc_demo.config import Cfg
from wrc_demo.perception.grounding import Extrinsics


def _new_calibration(tmp: Path, serial: str = "AY3Z331006L") -> Path:
    res = CalibrationResult(
        schema_version=1,
        frame_convention="wrc_demo",
        frame_convention_note="test",
        created_at="2026-07-20T12:00:00Z",
        method="joint_se3_lm_huber",
        mode="eye_to_hand",
        camera_serial=serial,
        camera_index_when_bound=0,
        marker_size_m=0.10,
        aruco_dict="4x4_50",
        aruco_marker_id=0,
        n_samples=10,
        n_inliers=10,
        n_outliers=0,
        outlier_indices=(),
        T_cam2base=np.eye(4).tolist(),
        T_marker2gripper=np.eye(4).tolist(),
        metrics={"translation_rmse_mm": 1.0},
    )
    p = tmp / "hand_eye_top.json"
    save_calibration(res, p)
    return p


def test_loader_new_json(tmp_path: Path):
    p = _new_calibration(tmp_path)
    cfg = Cfg({"mode": "eye_to_hand", "source": str(p), "serial": "AY3Z331006L"})
    e = Extrinsics.from_config(cfg)
    assert e.source.startswith("new-json:")
    assert e.frame_convention == "wrc_demo"
    assert e.schema_version == 1
    assert np.allclose(e.T, np.eye(4))


def test_loader_serial_mismatch_warns(tmp_path: Path, capsys):
    p = _new_calibration(tmp_path, serial="AY3Z331006L")
    cfg = Cfg({"mode": "eye_to_hand", "source": str(p), "serial": "OTHER"})
    e = Extrinsics.from_config(cfg)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "AY3Z331006L" in captured.err
    # Even on mismatch, the calibration value is used.
    assert np.allclose(e.T, np.eye(4))


def test_loader_curobo_npz(tmp_path: Path):
    p = tmp_path / "curobo.npz"
    np.savez(
        p,
        T_cam2base=np.eye(4),
        T_cam2gripper=np.eye(4),
        camera_serial="AY3Z331006L",
        mode="eye_to_hand",
    )
    cfg = Cfg({"mode": "eye_to_hand", "hand_eye_npz": str(p), "serial": "AY3Z331006L"})
    e = Extrinsics.from_config(cfg)
    assert e.source.startswith("curobo-npz:")
    assert np.allclose(e.T, np.eye(4))


def test_loader_legacy_npz(tmp_path: Path):
    p = tmp_path / "legacy.npz"
    np.savez(p, T_result=np.eye(4), mode="eye_to_hand")
    cfg = Cfg({"mode": "eye_to_hand", "hand_eye_npz": str(p)})
    e = Extrinsics.from_config(cfg)
    assert e.source.startswith("legacy-npz:")
    assert e.mode == "eye_to_hand"
    assert np.allclose(e.T, np.eye(4))


def test_loader_inline_T(tmp_path: Path):
    T = np.eye(4).tolist()
    cfg = Cfg({"mode": "eye_to_hand", "T": T})
    e = Extrinsics.from_config(cfg)
    assert e.source == "inline"
    assert np.allclose(e.T, np.eye(4))


def test_loader_prefers_json_over_npz(tmp_path: Path):
    jp = _new_calibration(tmp_path)
    np_p = tmp_path / "ignored.npz"
    np.savez(np_p, T_result=np.eye(4) * 99, mode="eye_to_hand")
    cfg = Cfg({
        "mode": "eye_to_hand",
        "source": str(jp),
        "hand_eye_npz": str(np_p),
        "serial": "AY3Z331006L",
    })
    e = Extrinsics.from_config(cfg)
    assert e.source.startswith("new-json:")


def test_loader_unknown_npz_raises(tmp_path: Path):
    p = tmp_path / "weird.npz"
    np.savez(p, unrelated=np.zeros(3))
    cfg = Cfg({"mode": "eye_to_hand", "hand_eye_npz": str(p)})
    try:
        Extrinsics.from_config(cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised

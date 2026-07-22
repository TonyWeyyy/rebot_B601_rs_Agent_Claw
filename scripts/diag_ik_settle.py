#!/usr/bin/env python3
"""诊断 IK settle 失败问题。

用法:
    python scripts/diag_ik_settle.py
    
检查:
1. IK 求解是否成功
2. pregrasp pose 是否在关节极限内
3. 从 home 到 pregrasp 的路径是否安全
4. settle 超时原因（位置误差）
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/seeed/reBotArm_control_py")

from wrc_demo.config import load_demo_config
from wrc_demo.control.kinematics import Kinematics
from wrc_demo.safety.harness import SafetyHarness, SafetyLimits
from wrc_demo.types import Grasp

def main():
    print("=== IK Settle 诊断工具 ===\n")
    
    # Load config
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    print(f"✓ Config loaded: arm={cfg.arm.type}")
    
    # Load kinematics
    kin = Kinematics(
        model_path=cfg.arm.model,
        ee_frame=cfg.arm.ee_frame,
        joint_signs=cfg.arm.joint_signs,
    )
    print(f"✓ Kinematics: n={kin.n}, fid={kin.fid}")
    
    # Home position
    q_home = np.array(cfg.arm.get("home_q", [0.0, -0.5, -0.9, 0.0, 0.6, 0.0]))
    print(f"\nHome position:")
    print(f"  q (rad): {q_home.round(3)}")
    print(f"  q (deg): {np.degrees(q_home).round(1)}")
    T_home = kin.fk(q_home)
    print(f"  TCP: {T_home[:3, 3].round(3)}")
    
    # Simulate a toy at [0.37, -0.06, 0.08] (from yesterday's trace)
    toy_pos = np.array([0.37, -0.06, 0.08])
    print(f"\nToy position: {toy_pos}")
    
    # Top-down rotation matrix
    R_topdown = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    approach = np.array([0.0, 0.0, -1.0])
    
    # Create a mock grasp
    grasp = Grasp(
        position=toy_pos,
        rotation=R_topdown,
        width_m=0.05,
        approach=approach,
        quality=0.9,
        label="toy",
    )
    
    # Pregrasp offset
    pregrasp_offset = float(cfg.grasp.get("pregrasp_offset_m", 0.04))
    print(f"Pregrasp offset: {pregrasp_offset} m")
    
    # Solve IK for pregrasp and grasp
    print(f"\n--- IK 求解 ---")
    
    # Grasp pose (top-down)
    T_grasp = np.eye(4)
    T_grasp[:3, 3] = toy_pos
    T_grasp[:3, :3] = R_topdown
    
    ik_grasp = kin.ik(T_grasp, q_home)
    print(f"Grasp IK:")
    print(f"  success: {ik_grasp.success}")
    print(f"  error: {ik_grasp.error:.6f}")
    print(f"  iterations: {ik_grasp.iterations}")
    if ik_grasp.success:
        print(f"  q (rad): {ik_grasp.q.round(3)}")
        print(f"  q (deg): {np.degrees(ik_grasp.q).round(1)}")
        T_check = kin.fk(ik_grasp.q)
        print(f"  TCP check: {T_check[:3, 3].round(3)}")
    
    # Pregrasp pose (offset along -approach)
    pregrasp_pos = grasp.pregrasp_position(pregrasp_offset)
    T_pregrasp = T_grasp.copy()
    T_pregrasp[:3, 3] = pregrasp_pos
    
    print(f"\nPregrasp position: {pregrasp_pos.round(3)}")
    
    ik_pre = kin.ik(T_pregrasp, q_home)
    print(f"Pregrasp IK:")
    print(f"  success: {ik_pre.success}")
    print(f"  error: {ik_pre.error:.6f}")
    print(f"  iterations: {ik_pre.iterations}")
    if ik_pre.success:
        print(f"  q (rad): {ik_pre.q.round(3)}")
        print(f"  q (deg): {np.degrees(ik_pre.q).round(1)}")
        T_check = kin.fk(ik_pre.q)
        print(f"  TCP check: {T_check[:3, 3].round(3)}")
        
        # Check joint limits
        lo, hi = kin.joint_limits
        print(f"\n  Joint limits check:")
        for i in range(kin.n):
            in_range = lo[i] <= ik_pre.q[i] <= hi[i]
            margin = min(ik_pre.q[i] - lo[i], hi[i] - ik_pre.q[i])
            print(f"    J{i+1}: {ik_pre.q[i]:.3f} in [{lo[i]:.3f}, {hi[i]:.3f}] {'✓' if in_range else '✗'} (margin: {margin:.3f})")
    
    # Safety harness check
    print(f"\n--- 安全检查 ---")
    limits = SafetyLimits.from_config(cfg.safety)
    harness = SafetyHarness(limits, kinematics=kin)
    
    if ik_pre.success:
        reason = harness.vet_pose(ik_pre.q)
        if reason:
            print(f"✗ Pregrasp pose rejected: {reason}")
        else:
            print(f"✓ Pregrasp pose passed safety check")
    
    # Settle tolerance analysis
    print(f"\n--- Settle 分析 ---")
    settle_tol = float(cfg.arm.get("settle_tol", 0.05))
    print(f"Settle tolerance: {settle_tol} rad ({np.degrees(settle_tol):.1f} deg)")
    
    if ik_pre.success:
        # Simulate what would happen with small errors
        print(f"\nPosition error sensitivity:")
        for err_rad in [0.01, 0.02, 0.03, 0.05, 0.10]:
            q_err = ik_pre.q + np.random.randn(kin.n) * err_rad
            q_err = kin.clamp(q_err)
            T_err = kin.fk(q_err)
            pos_err = np.linalg.norm(T_err[:3, 3] - pregrasp_pos)
            joint_err = np.abs(q_err - ik_pre.q).max()
            print(f"  Joint err {joint_err:.3f} rad -> TCP err {pos_err*1000:.1f} mm")
    
    print(f"\n=== 诊断完成 ===")

if __name__ == "__main__":
    main()

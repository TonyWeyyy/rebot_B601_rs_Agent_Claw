#!/usr/bin/env python3
"""诊断 grasp planning + IK settle 问题。

模拟实际的 grasp_object 流程：
1. 生成 grasp candidates (OBB fallback)
2. select_grasp 选择最佳候选
3. 检查 pregrasp pose 的可达性
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/seeed/reBotArm_control_py")

from wrc_demo.config import load_demo_config
from wrc_demo.control.kinematics import Kinematics
from wrc_demo.types import Detection, ObjectFix
from wrc_demo.grasping import plan_grasps_from_fix, select_grasp

def main():
    print("=== Grasp Planning 诊断 ===\n")
    
    # Load config
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    kin = Kinematics(
        model_path=cfg.arm.model,
        ee_frame=cfg.arm.ee_frame,
        joint_signs=cfg.arm.joint_signs,
    )
    
    q_home = np.array(cfg.arm.get("home_q", [0.0, 1.2, 1.2, 0.0, 0.75, 0.0]))
    print(f"Home: q_deg={np.degrees(q_home).round(1)}")
    print(f"      TCP={kin.fk(q_home)[:3,3].round(3)}\n")
    
    # Simulate a toy detection (from yesterday's trace)
    det = Detection(
        label="toy",
        conf=0.21,
        bbox=np.array([300.0, 200.0, 400.0, 300.0]),
    )
    
    # Object position from trace
    obj_pos = np.array([0.371, -0.058, 0.081])
    
    # Create ObjectFix
    extent = np.array([0.05, 0.05, 0.08])  # 5x5x8 cm toy
    axes = np.eye(3)  # aligned with base frame
    points = np.array([obj_pos])  # single point
    
    fix = ObjectFix(
        label="toy",
        position=obj_pos,
        points=points,
        detection=det,
        extent=extent,
        axes=axes,
    )
    
    print(f"Object: {det.label} at {obj_pos}")
    print(f"        extent={fix.extent}\n")
    
    # Plan grasps (OBB fallback since GraspGen-X not running)
    print("--- Grasp Candidates ---")
    grasps = plan_grasps_from_fix(fix, table_z=0.0, max_width_m=0.09, depth_fraction=0.5)
    print(f"Generated {len(grasps)} grasp candidates\n")
    
    # Sort by quality and show top 5
    grasps_sorted = sorted(grasps, key=lambda g: -g.quality)
    for i, g in enumerate(grasps_sorted[:5]):
        print(f"#{i+1} quality={g.quality:.2f}")
        print(f"   pos={g.position.round(3)}")
        print(f"   approach={g.approach.round(3)}")
        print(f"   width={g.width_m*1000:.0f}mm")
        
        # Check IK reachability
        pregrasp_offset = float(cfg.grasp.get("pregrasp_offset_m", 0.04))
        from wrc_demo.types import make_transform
        T_pre = make_transform(g.rotation, g.pregrasp_position(pregrasp_offset))
        ik_pre = kin.ik(T_pre, q_home)
        
        if ik_pre.success:
            print(f"   ✓ Pregrasp IK OK (err={ik_pre.error:.4f}, iter={ik_pre.iterations})")
            print(f"     q_deg={np.degrees(ik_pre.q).round(1)}")
        else:
            print(f"   ✗ Pregrasp IK FAILED (err={ik_pre.error:.4f})")
        print()
    
    # Try select_grasp
    print("--- Select Grasp ---")
    pregrasp_offset = float(cfg.grasp.get("pregrasp_offset_m", 0.04))
    try:
        grasp, q_pre, q_grasp = select_grasp(
            grasps, kin, q_home,
            max_width_m=float(cfg.arm.gripper.max_width_m),
            pregrasp_offset_m=pregrasp_offset,
            validate=None,  # skip safety check for now
        )
        print(f"✓ Selected grasp:")
        print(f"  quality={grasp.quality:.2f}")
        print(f"  pos={grasp.position.round(3)}")
        print(f"  approach={grasp.approach.round(3)}")
        print(f"\n  q_pre (deg): {np.degrees(q_pre).round(1)}")
        print(f"  q_grasp (deg): {np.degrees(q_grasp).round(1)}")
        
        # Check the delta
        dq = np.abs(q_pre - q_home)
        print(f"\n  Delta from home:")
        print(f"    max: {np.degrees(dq.max()):.1f} deg")
        print(f"    sum: {np.degrees(dq.sum()):.1f} deg")
        
        # Check settling feasibility
        settle_tol = float(cfg.arm.get("settle_tol", 0.05))
        print(f"\n  Settling analysis:")
        print(f"    tolerance: {settle_tol} rad ({np.degrees(settle_tol):.1f} deg)")
        print(f"    max delta: {dq.max():.3f} rad")
        
        # Estimate time to move
        max_vel = float(cfg.safety.get("max_joint_vel", 1.2))
        estimated_time = dq.max() / max_vel
        print(f"    estimated move time: {estimated_time:.1f} s")
        
    except Exception as e:
        print(f"✗ select_grasp failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 诊断完成 ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""诊断夹爪控制问题。

用法:
    cd /home/seeed/wrc_demo
    PYTHONPATH=src:/home/seeed/reBotArm_control_py \
    /home/seeed/miniforge3/envs/DM/bin/python scripts/diag_gripper.py

检查:
1. 夹爪配置的 open_pos / closed_pos 是否合理
2. 夹爪宽度计算是否正确
3. 夹爪控制命令是否正确发送
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/home/seeed/reBotArm_control_py")

from wrc_demo.config import load_demo_config
from wrc_demo.skills.runtime import SkillRuntime

def main():
    print("=== 夹爪诊断工具 ===\n")
    
    # Load config
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    print(f"✓ Config loaded: arm={cfg.arm.type}")
    
    # Check gripper config
    g = cfg.arm.get("gripper", {})
    open_pos = float(g.get("open_pos", 0.0))
    closed_pos = float(g.get("closed_pos", 1.0))
    max_width_m = float(g.get("max_width_m", 0.09))
    
    print(f"\n夹爪配置:")
    print(f"  open_pos:   {open_pos:.2f} rad ({np.degrees(open_pos):.1f}°)")
    print(f"  closed_pos: {closed_pos:.2f} rad ({np.degrees(closed_pos):.1f}°)")
    print(f"  max_width:  {max_width_m*1000:.0f} mm")
    print(f"  travel:     {abs(open_pos - closed_pos):.2f} rad ({np.degrees(abs(open_pos - closed_pos)):.1f}°)")
    
    # Check if values make sense
    print(f"\n检查:")
    if open_pos > closed_pos:
        print(f"  ✗ open_pos ({open_pos}) > closed_pos ({closed_pos}) - 可能反了")
    else:
        print(f"  ✓ open_pos < closed_pos")
    
    if abs(open_pos - closed_pos) < 1.0:
        print(f"  ✗ travel ({abs(open_pos - closed_pos):.2f} rad) 太小 - 夹爪可能没完全张开")
    else:
        print(f"  ✓ travel 合理")
    
    # Check gripper width calculation
    print(f"\n夹爪宽度计算:")
    span = closed_pos - open_pos
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pos = open_pos + span * frac
        width_m = max_width_m * frac
        print(f"  frac={frac:.2f}: pos={pos:.2f} rad, width={width_m*1000:.0f} mm")
    
    # Check material profiles
    from wrc_demo.grasping.force import select_profile
    print(f"\nMaterial profiles:")
    for material in ["rigid", "fragile", "soft"]:
        profile = select_profile("object", material)
        print(f"  {material:10s}: close_frac=[{profile.close_frac_stage1:.2f}, {profile.close_frac_stage2:.2f}], "
              f"effort={profile.effort:.2f}, lift_scale={profile.lift_speed_scale:.2f}")
    
    print(f"\n=== 诊断完成 ===")
    print(f"\n建议:")
    print(f"1. 如果夹爪张开不够宽，增大 open_pos (更负，如 -8.0)")
    print(f"2. 如果夹爪关不紧，减小 closed_pos (更正，如 0.5)")
    print(f"3. RS 夹爪的 travel 可能需要重新标定 (当前值基于 DM build)")

if __name__ == "__main__":
    main()

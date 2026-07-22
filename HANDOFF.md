# WRC Demo Handoff

## Today's Work (2026-07-22)

### 1. Cartesian Trajectory Planning (e97998c)
- Added `planning/cartesian_planner.py` wrapping `reBotArm_control_py.trajectory`
- SE(3) geodesic interpolation + CLIK tracking for smooth Cartesian paths
- `SafeArm` gained `move_joints_path()` and `move_cartesian()` methods
- Upgraded 4 skills to use Cartesian trajectories:
  - `grasp_object`: descend/lift now follow straight vertical paths
  - `place_at`: descend/lift now follow straight vertical paths  
  - `push_object`: all 3 segments use Cartesian motion
  - `move_relative`: now moves in straight Cartesian lines
- Added new `move_to_pose` skill (22nd skill total)
- All changes have automatic PTP fallback if trajectory planning unavailable

### 2. Home Pose Optimization (f6fca0d)
- **Root cause**: Top-down grasp IK has unique solution requiring J4≈-87°, J6≈-81°
- Old home `[0, 68.8, 68.8, 0, 43, 0]°` had J4=0°, J6=0° → 86.7° max delta → oscillation
- New home `[0, 56.1, 76.2, -60.2, 0, -80.8]°` (based on rebot_grasp ready pose)
  - J4 pre-positioned at -60° (closer to -87° grasp target)
  - J6 pre-positioned at -81° (matches grasp target)
- Result: Max joint delta reduced from 86.7° → 65.4° (24% improvement)

### 3. Live Testing Fixes (b70f3a0)
- **E-stop auto-reset**: Ctrl+C latched e-stop permanently; now auto-clears before each task
- **Reflex expansion**: "back to zero" / "return to home" / "reset" now map to `move_home`
- **Clean shutdown**: `KeyboardInterrupt` now caught during disconnect (was crashing)

## Current State

### Working
- Camera pipeline (Orbbec overhead + RealSense wrist)
- YOLO detection + 3D fusion
- Cartesian trajectory planning (with PTP fallback)
- Home pose with reduced joint deltas
- Reflex command parsing (expanded)
- E-stop recovery between tasks

### Known Issues
- **IK settle oscillation**: Max joint delta still 65.4° (J2), may still oscillate
- **USB stability**: Occasional camera disconnects (physical issue)
- **Grasp success rate**: Untested with new home pose + trajectory planning

## Next Steps (Tomorrow)

### Priority 1: Verify IK Settle Fix
```bash
cd /home/seeed/wrc_demo/models
sudo ip link set can0 up type can bitrate 1000000
sudo env HOME=/home/seeed PYTHONPATH=/home/seeed/wrc_demo/src:/home/seeed/reBotArm_control_py \
  /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs --llm mock --interactive
# Password: 0000

# Test commands:
# - "pick up the toy" (should use Cartesian trajectory)
# - "back to zero" (should trigger move_home reflex)
# - Watch for oscillation during descend/lift phases
```

**Success criteria**: 
- Grasp completes without "did not settle" errors
- Smooth vertical motion during descend/lift (no joint-space arcs)
- No oscillation at pregrasp/grasp poses

### Priority 2: If Oscillation Persists
Two options:
1. **Reduce max_joint_vel** in `configs/demo.yaml` (currently 1.2 rad/s → try 0.8)
2. **Further relax settle tolerance** (currently 0.08 rad → try 0.10)

### Priority 3: End-to-End Grasp Test
Once settle is stable:
```bash
# Switch from mock LLM to real planner
--llm openai  # or --llm anthropic

# Test full pick-and-place
"pick up the toy and place it in the basket"
```

## Key Files Changed Today
- `src/wrc_demo/planning/cartesian_planner.py` (new)
- `src/wrc_demo/safety/harness.py` (added move_joints_path, move_cartesian)
- `src/wrc_demo/skills/runtime.py` (upgraded 4 skills + added move_to_pose)
- `configs/arms/rebot_rs.yaml` (new home_q)
- `src/wrc_demo/apps/demo.py` (e-stop reset + KeyboardInterrupt handling)
- `src/wrc_demo/agent/reflex.py` (expanded home intent)

## Diagnostic Tools
```bash
# Test Cartesian planner (no hardware)
PYTHONPATH=src:/home/seeed/reBotArm_control_py \
  /home/seeed/miniforge3/envs/DM/bin/python scripts/test_cartesian_planner.py

# Analyze joint deltas for a grasp
PYTHONPATH=src:/home/seeed/reBotArm_control_py \
  /home/seeed/miniforge3/envs/DM/bin/python scripts/diag_grasp_planning.py

# Check IK reachability
PYTHONPATH=src:/home/seeed/reBotArm_control_py \
  /home/seeed/miniforge3/envs/DM/bin/python scripts/diag_ik_settle.py
```

## Git Log
```
b70f3a0 fix: e-stop auto-reset + 'back to zero' reflex + clean shutdown
f6fca0d fix: tune home_q to reduce joint deltas during grasps
e97998c feat: cartesian trajectory planning + IK settle fix
```

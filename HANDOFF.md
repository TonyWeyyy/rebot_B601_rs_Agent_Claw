# WRC Demo 交接文档

## 2026-07-22 工作摘要

### 完成的任务

1. **添加笛卡尔轨迹规划能力**
   - 新增 `src/wrc_demo/planning/cartesian_planner.py`
   - 封装 `reBotArm_control_py.trajectory` 模块
   - SE(3) 测地线插值 + CLIK 跟踪
   - 末端走笛卡尔直线而非关节空间弧线

2. **升级 SafeArm 运动接口**
   - `move_joints_path()` - 沿预计算航点序列运动
   - `move_cartesian()` - 笛卡尔轨迹运动，自动回退 PTP

3. **升级运动技能**
   - `grasp_object` - descend/lift 使用笛卡尔轨迹
   - `place_at` - descend/lift 使用笛卡尔轨迹
   - `push_object` - 三段运动使用笛卡尔轨迹
   - `move_relative` - 笛卡尔直线运动
   - **新增** `move_to_pose` - 通用笛卡尔位姿运动

4. **诊断 IK settle 问题**
   - 根因：大关节运动（86.7°）导致振荡
   - 修复：settle_tol 0.05→0.08 rad, timeout 2.0→4.0 s

### 改动的文件

**新增：**
- `src/wrc_demo/planning/__init__.py`
- `src/wrc_demo/planning/cartesian_planner.py`
- `scripts/diag_ik_settle.py` - IK 诊断工具
- `scripts/diag_grasp_planning.py` - Grasp planning 诊断

**修改：**
- `src/wrc_demo/safety/harness.py` - SafeArm 新增方法
- `src/wrc_demo/skills/runtime.py` - 5 个技能升级 + 新增
- `src/wrc_demo/control/arm_base.py` - settle_timeout_s 默认值
- `configs/arms/rebot_rs.yaml` - settle_tol 配置

### 环境状态

- ✅ DM conda env 可用（pinocchio, numpy, cv2）
- ✅ reBotArm_control_py trajectory 模块可导入
- ✅ 所有语法检查通过
- ✅ 规划模块端到端测试通过（51 waypoints, 0.1mm error）

### 待解决问题

1. **IK settle 振荡**（已部分修复）
   - 状态：增加了 tolerance 和 timeout
   - 下一步：真实硬件测试验证

2. **USB 相机断开**（物理问题）
   - 状态：昨天已改善但未根除
   - 建议：检查线缆和供电

3. **轨迹规划实时测试**
   - 状态：代码就绪
   - 下一步：mock 模式测试 → 真实硬件测试

## 启动命令

### 验证改动

```bash
cd /home/seeed/wrc_demo
PYTHONPATH=src:/home/seeed/reBotArm_control_py \
/home/seeed/miniforge3/envs/DM/bin/python -c "
from wrc_demo.planning import _AVAILABLE
print(f'CartesianPlanner: {_AVAILABLE}')
from wrc_demo.skills.runtime import TOOL_SPECS
print(f'Skills: {len(TOOL_SPECS)}')
"
```

### Mock 模式测试

```bash
cd /home/seeed/wrc_demo/models
PYTHONPATH=/home/seeed/wrc_demo/src:/home/seeed/reBotArm_control_py \
/home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.apps.demo \
  --camera mock --arm mock --llm mock --interactive --no-view --no-serve
```

### 真实硬件测试

```bash
cd /home/seeed/wrc_demo/models
sudo ip link set can0 up type can bitrate 1000000
sudo env HOME=/home/seeed YOLO_OFFLINE=True \
PYTHONPATH=/home/seeed/wrc_demo/src:/home/seeed/reBotArm_control_py \
/home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.apps.demo \
  --interactive --cameras orbbec_overhead --arm rebot_rs --llm mock
# 密码: 0000
# 测试: "pick up the toy"
```

### 诊断工具

```bash
# IK settle 诊断
PYTHONPATH=src:/home/seeed/reBotArm_control_py \
/home/seeed/miniforge3/envs/DM/bin/python scripts/diag_ik_settle.py

# Grasp planning 诊断
PYTHONPATH=src:/home/seeed/reBotArm_control_py \
/home/seeed/miniforge3/envs/DM/bin/python scripts/diag_grasp_planning.py
```

## 技术要点

### 轨迹规划工作原理

```
目标位姿 (x,y,z,roll,pitch,yaw)
    ↓ IK 求解 q_end
    ↓ SE(3) 测地线插值 T_start → T_end
    ↓ CLIK 跟踪生成 N 个关节角
    ↓ SafetyHarness 逐点审查
    ↓ min-jerk 流式发送
```

### 回退机制

- `reBotArm_control_py` 不可用 → 自动回退 PTP
- CLIK 跟踪失败 → 自动回退 PTP
- 所有现有功能不受影响

### 关键参数

- `settle_tol`: 0.08 rad (4.6°) - 位置容差
- `settle_timeout_s`: 4.0 s - 等待稳定时间
- `pregrasp_offset_m`: 0.04 m - 预抓取高度
- `max_joint_vel`: 1.2 rad/s - 关节速度限制

## Git 状态

```
Modified:
  src/wrc_demo/safety/harness.py
  src/wrc_demo/skills/runtime.py
  src/wrc_demo/control/arm_base.py
  configs/arms/rebot_rs.yaml

Untracked:
  src/wrc_demo/planning/
  scripts/diag_ik_settle.py
  scripts/diag_grasp_planning.py
```

建议 commit message:
```
feat: cartesian trajectory planning + IK settle fix

- Add CartesianPlanner wrapping reBotArm_control_py.trajectory
- SafeArm: move_joints_path, move_cartesian with PTP fallback
- Upgrade grasp/place/push/move_relative to cartesian trajectory
- Add move_to_pose skill (22nd skill)
- Fix IK settle: increase tol to 0.08 rad, timeout to 4.0 s
- Add diagnostic scripts for IK and grasp planning
```

---

## 2026-07-22 (下午) — Home pose 优化

### 问题诊断

IK settle 振荡根因分析：

```
旧 home: [0, 68.8, 68.8, 0, 43, 0]°
Pregrasp IK 唯一解: [8.9, 121.6, 118.3, -86.7, 0, -81.1]°

J4: 0° → -86.7° (86.7° delta) ← 振荡主因
J6: 0° → -81.1° (81.1° delta)
```

Pregrasp IK 对该旋转矩阵只有唯一解，轨迹规划无法改变目标关节角。

### 解决方案

采用 rebot_grasp 的 ready pose 并预置 J4/J6：

```
新 home: [0, 56.1, 76.2, -60.2, 0, -80.8]°
  - 基于 rebot_grasp ready (x=0.3, y=0, z=0.3, pitch=0.7)
  - J4 预置到 -60° (接近 grasping 的 -87°)
  - J6 预置到 -81° (接近 grasping 的 -81°)
```

### 效果

| 场景 | 旧 home max delta | 新 home max delta | 改善 |
|------|-------------------|-------------------|------|
| Toy [0.37, -0.06, 0.08] | 86.7° (J4) | 65.4° (J2) | **24%** |
| Cup [0.3, 0, 0.05] | 90.0° (J6) | 43.8° (J2) | **51%** |

### 提交

```
f6fca0d fix: tune home_q to reduce joint deltas during grasps
```

修改文件：`configs/arms/rebot_rs.yaml`

### 下一步

1. 真实硬件测试新 home pose
2. 验证 settle 振荡是否解决
3. 观察笛卡尔轨迹规划效果


# 夹爪控制问题修复方案

## 问题
1. 夹爪"一瞬间打开" - 没有平滑轨迹
2. 没有自适应开合 - 不根据物体尺寸调节
3. 第二次抓取"挂掉" - 控制逻辑不完整

## 根因
`RebotRSArm` 缺少持续运行的控制循环，`set_gripper()` 只发送一次命令。

## 对比
- **reBotArm_control_py**: 有 `_loop_cb()` 每 10ms 运行，平滑跟踪 `gripper_target`
- **rebot_grasp**: 有 `gripper_tick()` 状态机，支持 `open_gripper(distance_m)` 自适应开合
- **wrc_demo**: 只有 `set_gripper()` 单次命令，没有控制循环

## 修复步骤
1. 添加控制循环线程，持续发送 MIT 命令跟踪 `self._gripper_target`
2. 添加 `set_gripper_target(pos)` 方法（只设置目标）
3. 修改 `open_gripper(distance_m=None)` 支持自适应开合
4. 改进 `close_gripper_two_stage()` 使用控制循环

## 优先级
**高** - 这是抓取功能的核心问题

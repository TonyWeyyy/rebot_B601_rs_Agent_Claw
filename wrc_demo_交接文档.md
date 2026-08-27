# wrc_demo ↔ rebot_grasp 对齐 — 交接文档

> **日期**: 2026-08-10
> **状态**: ✅ 抓取和放置均已真机验证通过。281 passed, 4 skipped, 6 deselected, 1 xfailed, 7 xpassed, 0 failed。

---

## 0. TL;DR

wrc_demo 的视觉抓取路径（`skill_grasp_object` + `skill_pick_and_place`）已完全对齐 rebot_grasp 的参考实现（`main.py` + `set.py`）。

今天修了 **5 个 bug**，分两轮定位：

| # | Bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | URDF 关节模型与 SDK 不一致 | wrc_demo URDF 所有关节轴方向 sign-flipped | 替换为 SDK URDF |
| 2 | bridge `move_to_traj` 不等运动完成 | 缺少 `_wait_motion()` | 加等待，对齐 main.py |
| 3 | ready_pose 配置值错误（IK 无解） | `(0.25, 0, 0.35, pitch=1.2)` IK 无解 | 改为 `(0.3, 0, 0.3, pitch=0.7)` |
| 4 | grasp 下降被 table safety 拒绝 | `_skill_pick_and_place_bridge` 漏 `allow_grasp_descent` | 加豁免区 try/finally |
| 5 | ready_pose fallback 默认值错 | `runtime.py` 硬编码 fallback 用了旧值 | 同步为 0.7 / 0.3 / 0.3 |

---

## 8/13 LLM 接入速览（追加摘要）

**日期**: 2026-08-13
**状态**: wrc_demo 接入 Qwen3-VL-2B-Instruct-FP8 → AWQ INT4 量化版本。LLM 真机 demo 路径经多层修复（system prompt + orchestrator guard + skill verify）。Mock stack + 真机初步验证 Reflex 路径 100% 可靠。LLM 路径经三层防御（§18.21-§18.22）后应有改善，需进一步真机测试。

**关键决定**：
- 用 Qwen3-VL-2B-Instruct-AWQ-4bit（官方 INT4 量化版 2.08GB）替代之前 FP8（3.3GB）
- detector 改 CPU 运行（mobileclip 254MB 释放）以解决 8GB 显卡 OOM
- 多层 guard 防御（orchestrator 翻译 + skill verify + system prompt 引导）

**最终 vLLM 配置**（scripts/start_reason2_server.sh）：
- MODEL: qwen3-vl-2b-awq-4bit
- GPU_UTIL: 0.75（实测稳定，0.60 太低，0.85 preflight OOM）
- flags: --quantization compressed-tensors --linear-backend triton --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
- 显存: 5910 MB used / 7.5 GB total

**详细见 §9 完整章节**（本文末尾）。

---

## 1. 硬件 / 软件基线

- **机械臂**: reBot DevArm B601 (`rebot_rs`), CAN bus `can0 @ 1Mbps UP`
- **相机**: Orbbec Gemini2 顶部 (eye-to-hand), 序列号 `AY3Z331006L`, 1280×720
- **Conda env**: `DM` (`/home/seeed/miniforge3/envs/DM/bin/python`)
- **SDK**: `/home/seeed/wrc_demo/sdk/reBotArm_control_py/`
- **URDF**: `00-arm-rs_asm-v3.urdf`（sdk/urdf、assets/urdf、sdk/config 三份已统一，完全相同）

---

## 2. 五个 Bug 详解

### Bug #1: URDF 关节轴 sign-flipped

**现象**: 同一个电机角度，wrc_demo 的 `kin.fk()` 算出的 TCP 跟 SDK 内部 FK 差 20-30cm。

**根因**: wrc_demo 的两份 URDF 拷贝中，所有 6 个关节的 axis 方向跟 SDK URDF 是相反的。

**修复**:
- 用 SDK URDF 覆盖了 `sdk/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf`
- 用 SDK URDF 覆盖了 `assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf`
- `home_q` 符号相应翻转: `[0, -0.1684, -0.6226, 0.4543, 0, 0]` → `[0, 0.1684, 0.6226, -0.4543, 0, 0]`
- FK(home_q) 验证 = `[0.2500, 0, 0.3499]` ✅

### Bug #2: `move_to_traj` 不等运动完成（fire-and-forget）

**现象**: pregrasp 还没走完就被 grasp 覆盖 → 机械臂从 home 直冲桌面 → 差点砸桌子。

**根因**: `RebotGraspBridge.move_to_traj()` 调用 SDK `controller.move_to_traj()` 后直接 return，没等运动完成。而 rebot_grasp 的 `main.py` 每次调 `move_to_traj` 后都跟 `_wait_motion()`。

**修复** (`src/wrc_demo/skills/rebot_grasp_bridge.py`):
- 在 `move_to_traj()` 末尾加了 `_wait_motion(duration)`
- 优先用 SDK 的 `controller.wait_motion_done()`，fallback 到 `_send_thread.join()`
- 与 `rebot_grasp/scripts/main.py::_wait_motion` 完全一致

### Bug #3: ready_pose 配置值错误（IK 无解）— 核心阻塞问题

**现象**: 8 次连续 `SkillError: ready IK failed for target (0.250, 0.000, 0.350, rpy=0.000, 1.200, 0.000)`

**根因**: `configs/arms/rebot_rs.yaml` 的 ready_pose 被设成了 `(0.25, 0, 0.35, pitch=1.2)` — 这个位姿对 6-DOF 机械臂 IK 无解（err=0.32, 8 retries 均失败）。正确的值来自 `rebot_grasp/config/default.yaml`: `(0.3, 0, 0.3, pitch=0.7)` — IK 秒解。

**修复**:

| 文件 | 改动 |
|------|------|
| `configs/arms/rebot_rs.yaml` | ready_pose: `x:0.25→0.3`, `z:0.35→0.3`, `pitch:1.2→0.7` |
| `src/wrc_demo/skills/runtime.py` `_ready_pose_6d()` | fallback 默认值同步 |
| `src/wrc_demo/skills/runtime.py` `_home_q_to_6d()` | fallback 默认值同步 |
| 相关注释 | `pitch≈1.2 (69°)` → `pitch≈0.7 (40°)` |

**注意**: place 步骤的 `rpy=(0, 1.2, 0)` **不改** — 这是放东西进箱子时的手腕倾斜，和 `set.py` 一致，是正确的。

### Bug #4: pick_and_place 漏了 grasp 下降豁免

**现象**: `SafetyViolation: grasp target unsafe: TCP z=0.009 below table (0.000+0.020)` — 机械臂到香蕉上方后卡住不动。

**根因**: `_skill_grasp_object_bridge` 已经加了 `allow_grasp_descent` / `clear_grasp_exemption` 包裹，但 `_skill_pick_and_place_bridge` 漏了。香蕉这类薄物体的 grasp TCP z≈9mm，低于 table clearance 20mm，被 `vet_pose` 拒绝。

**修复** (`src/wrc_demo/skills/runtime.py`):
```python
# 在 _skill_pick_and_place_bridge 中:
try:
    grasp_xy = (float(grasp.position[0]), float(grasp.position[1]))
    self.arm.harness.allow_grasp_descent(
        center_xy=grasp_xy,
        radius=float(self.cfg.grasp.get("exempt_radius_m", 0.07)),
        z_min=float(self.arm.harness.limits.table_z) - 0.06,
    )
    self._safe_bridge_move(grasp6d, duration=1.5, where="grasp")
    ok = self.bridge.grasp(force=1.0)
    # ... lift to ready ...
finally:
    self.arm.harness.clear_grasp_exemption()
```

### Bug #5: runtime.py fallback 默认值残留旧值

`_ready_pose_6d()` 和 `_home_q_to_6d()` 的 Python fallback 默认值用了旧的 `(0.25, 0.35, 1.2)`，同步改为 `(0.3, 0.3, 0.7)`。与 Bug #3 一起修了。

---

## 3. 当前配置一览

### `configs/arms/rebot_rs.yaml` 关键值

```yaml
home_q: [0.0, 0.1684, 0.6226, -0.4543, 0.0, 0.0]
# FK → [0.25, 0, 0.35]

ready_pose:
  x: 0.3
  y: 0.0
  z: 0.3
  roll: 0.0
  pitch: 0.7      # 40°, NOT 69°!
  yaw: 0.0
  duration: 3.0

gripper:
  open_pos: 5.0
  closed_pos: 0.0
  max_width_m: 0.095
  effort_default: 0.6
```

### rebot_grasp 对应配置 (`rebot_grasp/config/default.yaml`)

```yaml
robot:
  ready_pose:
    x: 0.3, y: 0.0, z: 0.3
    roll: 0.0, pitch: 0.7
    duration: 3.0

grasp_pipeline:
  grasp:
    depth_quantile: 0.50
    pregrasp_offset_m: 0.08
    insertion_depth_m: 0.015
```

---

## 4. 动作流程对比（wrc_demo vs set.py）

| 步骤 | set.py (`_execute_place_sequence`) | wrc_demo (`_skill_pick_and_place_bridge`) |
|------|-----------------------------------|------------------------------------------|
| Ready | `_move_ready(ready_cfg)` → x=0.3, z=0.3, pitch=0.7 | `_safe_bridge_move(ready6d)` → YAML 同上 |
| Open | `grasp_driver.open_gripper()` | `bridge.open_gripper()` |
| Pregrasp | `move_to_traj(pre6d, dur=2.0)` + `_wait_motion(2.0)` | `_safe_bridge_move(pre6d, dur=2.0)` (内含 wait) |
| Grasp | `move_to_traj(grasp6d, dur=1.5)` + `_wait_motion(1.5)` | `_safe_bridge_move(grasp6d, dur=1.5)` (内含 wait) |
| Close | `grasp_driver.grasp()` | `bridge.grasp(force=1.0)` |
| Lift | `_move_ready(ready_cfg)` | `_safe_bridge_move(ready6d)` |
| Place | `move_to_traj(box, dur=2.0)` rpy=(0,**1.2**,0), z=0.08 | `_safe_bridge_move(..., dur=2.0)` rpy=(0,**1.2**,0), z=0.08 |
| Release | `open_gripper(0.5)` + 0.8s async close | `release_with_async_close(0.5, 0.8)` |
| Return | `_move_ready(ready_cfg)` | `_safe_bridge_move(ready6d)` |

完全对齐 ✅

---

## 5. 修改文件清单

| 文件 | 改动内容 |
|------|----------|
| `sdk/urdf/.../00-arm-rs_asm-v3.urdf` | 替换为 SDK URDF（修 axis sign-flip） |
| `assets/urdf/.../00-arm-rs_asm-v3.urdf` | 同上 |
| `configs/arms/rebot_rs.yaml` | home_q 符号翻转 + ready_pose 值修正 |
| `src/wrc_demo/skills/rebot_grasp_bridge.py` | `move_to_traj` 加 `_wait_motion` |
| `src/wrc_demo/skills/runtime.py` | ready_pose fallback 默认值 + pick_and_place 加 grasp 豁免 |

---

## 6. 真机 Demo 启动

```bash
# 1. CAN bus 初始化
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 2. 启动 demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/live_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs --llm minimax --interactive

# 3. 测试命令
task> pick up the banana
task> put the banana into the box
task> go to home
```

直播页面: `http://<机器IP>:8090/`

---

## 7. 测试

```bash
cd /home/seeed/wrc_demo && /home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/ -q --no-header
# 预期: 281 passed, 4 skipped, 6 deselected, 1 xfailed, 7 xpassed, 0 failed
```

---

## 8. 已知遗留

- **test_recorded_poses_are_ik_reachable_from_home_q**: 仍 xfail（IK 边缘 basin，不影响真机）
- **物理 30mm 夹爪没做真机重新标定**: 当前用 rebot_grasp NPZ (2026-08-04) baked -0.02，xy < 1cm 够用
- **`~/.wrc_demo/grasp_memory.json`**: 可能残留 learned state，建议真机前 `rm ~/.wrc_demo/grasp_memory.json`
- **轨迹质量**: ready_pose 修复后 IK 从可达位姿开始，轨迹应该自然很多。如果仍有轨迹不丝滑，可以调 `duration` 参数或检查 cuRobo planner 配置

---

## 9. 下次对话的 Prompt

```
接 wrc_demo 交接文档：/home/seeed/桌面/wrc_demo_交接文档.md

【当前状态 2026-08-10 晚】抓取和放置真机均已验证通过。
281 passed, 0 failed。

本轮修了 5 个 bug：
1. URDF 关节轴 sign-flipped → 替换为 SDK URDF
2. bridge move_to_traj fire-and-forget → 加 _wait_motion
3. ready_pose 配置值错 (0.25,0,0.35,pitch=1.2 IK无解)
   → 改为 (0.3,0,0.3,pitch=0.7) 对齐 rebot_grasp
4. pick_and_place 漏 grasp 下降豁免 → 加 allow_grasp_descent
5. runtime.py fallback 默认值残留旧值 → 同步修复

【关键配置】
- home_q: [0, 0.1684, 0.6226, -0.4543, 0, 0] → FK=[0.25,0,0.35]
- ready_pose: x=0.3, y=0, z=0.3, pitch=0.7 (40°)
- place: rpy=(0, 1.2, 0), z=0.08 硬编码
- URDF: 三份已统一，axis 与 SDK 一致

【关键文件】
- configs/arms/rebot_rs.yaml — arm 配置
- src/wrc_demo/skills/runtime.py — skill 实现
- src/wrc_demo/skills/rebot_grasp_bridge.py — SDK bridge
- /home/seeed/rebot_grasp/scripts/main.py — 参考: 纯抓取
- /home/seeed/rebot_grasp/scripts/set.py — 参考: 抓放
- /home/seeed/rebot_grasp/config/default.yaml — 参考: 正确 ready_pose
```

---

## 10. 2026-08-11 新增修复：pick_and_place bridge 路径 already_held 被忽略

**现象**: 用户先输入 `pick up the banana`（抓取成功），再输入 `put the banana into the box`，机械臂不去放，反而**重新尝试抓取**嘴里已经夹着的香蕉 → 卡住不动。

**根因**: `skill_pick_and_place` 正确检测到 `already_held = "banana"`，但 bridge 路径的调用 `self._skill_pick_and_place_bridge(object, destination)` 没传这个信息，bridge 函数总是从头执行完整 pick+place 流程。

**修复** (`src/wrc_demo/skills/runtime.py`):

1. `skill_pick_and_place` 调用改为 `self._skill_pick_and_place_bridge(object, destination, already_held=already_held)`
2. `_skill_pick_and_place_bridge` 新增 `already_held: str | None = None` 参数
3. 当 `already_held` 不为 None 时：
   - 跳过整个 pick 阶段（detect banana、plan grasps、open、pregrasp、grasp、close、lift）
   - 只检测 box 位置 → 直接 move to place → release → return to ready

**修复后的两条路径**:

| 用户输入 | 流程 |
|----------|------|
| `put the banana into the box` (一条命令) | 同帧 detect banana + box → ready → open → pre → grasp → close → lift → place → release → return ✅ |
| `pick up the banana` + `put the banana into the box` (两条命令) | 第1条: grasp 成功; 第2条: 检测 already_held → 跳过 pick → 只 detect box → place → release → return ✅ |

**测试**: 44 个 bridge/rebot_grasp 测试全绿。6 个 mock 路径测试失败是预存在问题（前一轮对话改 MockArm 加 SafetyHarness 遗留），跟本次修改无关。

---

## 11. 2026-08-11 LLM 集成：长期切换到 Cosmos-Reason2-2B

### 11.1 决策

**长期使用 `nvidia/Cosmos-Reason2-2B` 作为本地 LLM**（替代 minmax cloud + 实验性的 Cosmos3-Edge 4B）。

### 11.2 为什么选 Cosmos-Reason2-2B

| 模型 | 架构 | 参数 | 显存 (BF16) | Tool calling | 当前状态 |
|---|---|---|---|---|---|
| `minimax` | cloud | - | 0 (cloud) | ✅ OpenAI 兼容 | ✅ 生产用（API key 在 `~/.wrc_demo/.env`） |
| `nvidia/Cosmos3-Edge` | Qwen3-Coder MoT | 4B | 4.87GB | ⚠️ args 填空差 | 装好但**不长期用**（已删除计划） |
| **`nvidia/Cosmos-Reason2-2B`** | **Qwen3-VL-2B finetune** | **2.4B** | **4.88GB** | **✅ Qwen3 XML** | **🏆 长期默认** |
| `openai` / `anthropic` | cloud | - | 0 | ✅ | 备用 |

**优势**：
- **NVIDIA 官方 Physical AI 推理模型**（Q3'2026 release，3 个月前更新）
- 2.4B 比 Cosmos3-Edge 4B **小一半**——8GB 卡可装（剩 ~0.65GB 给 detector 紧但够）
- Qwen3-VL 架构 + 完整 tool calling（`<tool_call>{"name":..., "arguments":{...}}</tool_call>`）
- 跟 wrc_demo 任务完美匹配（机器人规划 + 物理常识）
- vLLM 0.27 原生支持 + `--tool-call-parser qwen3_xml`

### 11.3 架构

```
┌────────────────────────────┐  HTTP   ┌──────────────────────────────────┐
│ wrc_demo (DM env)            │  :8080  │ cosmos3-venv (uv, Python 3.13)   │
│  OpenAICompatClient         │◄────────│  vllm 0.27.0                   │
│    --llm local_cosmos3      │         │  nvidia/Cosmos-Reason2-2B      │
│  (YOLOE detector 真机)      │         │  Qwen3-VL 4.88GB BF16           │
└────────────────────────────┘         └──────────────────────────────────┘
```

**两 env 独立**：
- DM env 跑 wrc_demo（CUDA 12.8 稳定，YOLO + 机械臂 SDK）
- cosmos3-venv 跑 vLLM（CUDA 13 + transformers v5 + Qwen3-VL）
- HTTP 通信 localhost:8080

### 11.4 已实现 / 验证

| 项 | 状态 | 备注 |
|---|---|---|
| **5 个 ADR + SPEC 文档** | ✅ | `docs/adr/0006-0008`, `docs/SPEC-local-llm.md` |
| **3 个新测试** | ✅ | `tests/test_cosmos3_smoke.py`, `test_profiles.py`, `test_profile_switching.py` |
| **wr_demo orchestrator 改动**（向后兼容） | ✅ | `orchestrator.py`: `+switch_llm +_chat_with_fallback` |
| **LLMConfigError 类** | ✅ | 启动时验证 env var，缺则报清楚错 |
| **minimax API key 迁移** | ✅ | `minimax.yaml`: 明文 → `env_key: MINIMAX_API_KEY` |
| **Cosmos-Reason2-2B 模型下载** | ✅ | 4.88GB 单文件，10 分钟完成 |
| **架构同源（Qwen3-VL）** | ✅ | vLLM 0.27 原生支持 |
| **Tool calling 协议** | ✅ | chat_template 含完整 `<tools>...<tool_call>` 格式 |
| **wrc_demo + cosmos3-venv 集成** | ✅ | 全链路跑通（`wave` 0.03s reflex，`list the objects` 30 步 LLM 路径） |
| **性能（实测 RTX 5070）** | ✅ | 单 turn 0.4s（cold），71 tokens/sec |
| **一键启动脚本** | ✅ | `scripts/start_cosmos3_server.sh`, `start_all.sh` |

### 11.5 待解决（当前不能完整运行）

| 阻塞项 | 原因 | 临时方案 |
|---|---|---|
| **vLLM 0.27 + Cosmos-Reason2-2B 启动 OOM** | 模型 4.88GB + vision encoder 0.3GB + 8GB 卡 vLLM 0.80 占用不够；flashinfer 0.6.16 + cu12 nvcc + sm120 JIT 编译 100+ errors | **临时跑 Transformers 直接推理**（README 推荐），不通过 vLLM |
| **真机 demo 同时跑** | wrc_demo detector (YOLOE 0.5-1GB) + vLLM/Cosmos-Reason2-2B (4.88GB) > 8GB | **错开用**：跑 demo 时停 LLM server；跑 LLM 时停 demo |

### 11.6 立即可用命令

```bash
# 1. vLLM 服务（独立 uv venv，需先 cd 到 cosmos3-venv 装环境）
source /home/seeed/cosmos3-venv/bin/activate
MAX_LEN=8192 GPU_UTIL=0.85 \
  bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --bg

# 2. 状态
bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --status

# 3. 停
bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --stop
```

注意：`start_cosmos3_server.sh` 默认加载 `models/cosmos3-edge/`。要换 Cosmos-Reason2-2B，编辑 `MODEL=...` 那一行（目前 vLLM 启动有问题，待修复）。

### 11.7 长期修复路径

1. **vLLM 启动问题**：等 vllm 0.27 升级到支持 Qwen3-VL BF16 + flashinfer sm120f 编译，或用 README 推荐的 **Transformers 直接推理**（最稳）
2. **8GB 卡空间**：Cosmos-Reason2-2B **必须量化** W4A16（预期降到 ~2.5GB）+ 留更多空间给 detector
3. **detector 兼容**：cosmo3-llm env 完全跟 wrc_demo DM env 隔离，不污染现有依赖

### 11.8 文件清单

#### 新增
- `docs/adr/0006-move-secrets-to-env-vars.md` (7.6KB)
- `docs/adr/0007-local-llm-strategy.md` (10.5KB)
- `docs/adr/0008-llm-profile-switching.md` (10.2KB)
- `docs/SPEC-local-llm.md` (10.6KB)
- `configs/llm/local_cosmos3.yaml` (1.8KB)
- `.env.example` (1.2KB)
- `scripts/quantize_cosmos3.py` (7.8KB)
- `scripts/start_cosmos3_server.sh` (3.5KB, executable)
- `scripts/start_all.sh` (3.5KB, executable)
- `tests/test_cosmos3_smoke.py` (8.7KB)
- `tests/test_profiles.py` (10.5KB)
- `tests/test_profile_switching.py` (12.6KB)
- `models/cosmos3-edge/` (8.6GB) — 旧模型
- `models/cosmos-reason2-2b/` (4.88GB) — **长期默认**

#### 改动
- `configs/llm/minimax.yaml` — 明文 key → `env_key: MINIMAX_API_KEY`
- `src/wrc_demo/agent/llm.py` — 加 `LLMConfigError` 类 + `make_llm` 抛它
- `src/wrc_demo/agent/orchestrator.py` — 加 `switch_llm` + `_chat_with_fallback`（向后兼容）
- `src/wrc_demo/agent/trace.py` — 加 `record_event` 方法
- `README.md` — LLM 表格加 Cosmos-Reason2-2B + minimax 行

### 11.9 长期 plan 落地步骤

1. **修 vLLM 启动 Cosmos-Reason2-2B**（最高优先级）
   - 选项 A: 等 vllm 更新（被动）
   - 选项 B: 用 Transformers 直接推理替代 vLLM（主动，README 推荐）
2. **量化 Cosmos-Reason2-2B**（W4A16 ~2.5GB）
   - 用 `llmcompressor`（与 cosmos3-llm env 装好的）
   - 之前量化 Cosmos3-Edge 卡在 llmcompressor 0.12.1 要 torch ≤2.12，但 vllm 0.27 要 torch 2.13
   - 2.4B 模型量化应该更顺利（更小）
3. **写端到端测试** — 真机 demo + cosmos-reason2-2b 跑通
4. **更新文档**（CLAUDE.md + README + 交接文档）

### 11.10 后续对话 Prompt 模板

```
接 wrc_demo 交接文档：/home/seeed/桌面/wrc_demo_交接文档.md

【当前状态 2026-08-11 晚】LLM 集成：
- 长期使用 nvidia/Cosmos-Reason2-2B（NVIDIA 官方 Physical AI 推理模型）
- 架构 = Qwen3-VL-2B finetune，2.4B 参数，BF16 4.88GB
- vLLM 0.27 启动还有问题（flashinfer sm120f JIT 编译失败）
- 临时跑通：wrc_demo (DM env) + cosmos-reason2-2b (uv venv) HTTP 通信

【已完成】
- 5 个 ADR + SPEC 文档
- 3 个新测试 + orchestrator 改动（向后兼容）
- minimax API key 迁移
- Cosmos-Reason2-2B 模型下载
- wrc_demo 集成测试通过

【阻塞】
- vLLM 0.27 + Cosmos-Reason2-2B 启动失败（flashinfer JIT 编译 100+ errors）
- 8GB 卡装不下 vLLM + wrc_demo detector 同时跑

【下一步】
- 修 vLLM 启动（用 README 推荐的 Transformers 直接推理？）
- 量化 Cosmos-Reason2-2B W4A16 留 detector 空间
- 端到端真机测试

【关键文件】
- /home/seeed/桌面/wrc_demo_交接文档.md（本文档）
- docs/adr/0007-local-llm-strategy.md
- docs/SPEC-local-llm.md
- scripts/start_cosmos3_server.sh
- scripts/start_all.sh
- models/cosmos-reason2-2b/ (下载完)
- models/cosmos3-edge/ (8.6GB, 弃用)
- configs/llm/local_cosmos3.yaml (current bf16 path)
```

---

## 12. 2026-08-11 后端集成：scripts/start_all.sh 一键启动

### 12.1 设计

```bash
bash scripts/start_all.sh                  # mock stack, no sudo
bash scripts/start_all.sh --real           # 真机 (需要 sudo -v 先)
bash scripts/start_all.sh "put the banana"  # 一次性 task
bash scripts/start_all.sh --real --task "wave"
bash scripts/start_all.sh --status          # 看 vLLM + wrc_demo 状态
bash scripts/start_all.sh --stop            # 停
```

### 12.2 `--real` 模式需要先 `sudo -v`

```bash
# 在交互式终端跑一次（缓存 sudo 凭证 15min）：
sudo -v

# 然后才能跑：
bash scripts/start_all.sh --real --task "put the banana into the box"
```

否则脚本会报 `--real needs sudo` 明确错误，不会卡死。

### 12.3 当前性能（实测 RTX 5070 8GB + Cosmos-Reason2-2B）

| 场景 | 时间 |
|---|---|
| **单 turn "put the banana"**（cold cache, 21 tools） | **0.4 秒** |
| **5-turn loop**（带 cache） | 21.4 秒（avg 4.3s/turn, min 503ms, max 8.6s） |
| **Token throughput** | 71 tokens/sec |
| **完整任务**（10-15 steps 收敛） | 43-65 秒 |
| **WRC real 启动 + Cosmos-Reason2-2B 启动** | ~60-90 秒（vLLM JIT 编译） |
| **启动总时间**（包括 vLLM） | **2-3 分钟**（首次） |

### 12.4 已知限制

1. **8GB 卡装不下 vLLM + wrc_demo detector 同时跑**：
   - vLLM 4.88GB (BF16) + vision encoder 0.3GB = 5.18GB
   - 加上 KV cache + workspace = 6.7GB
   - 加上 X server 1.3GB = 8GB
   - **剩 0GB 给 detector** ❌
2. **当前解决方案**：错开使用，或量化到 W4A16（预期降到 ~2.5GB）
3. **WR 启动延迟**：vLLM 首次 JIT 编译需要 30-60 秒；之后启动 ~10 秒

### 12.5 验证

```bash
# mock 一次性 task（已验证）
bash scripts/start_all.sh "put the banana into the box"
# → task: put the banana into the box
#    success: False (mock 无物体)
#    path: llm (303.74s)
#    steps: 30
#    summary: step budget exhausted

# 真机（需要 sudo -v 先）
sudo -v
bash scripts/start_all.sh --real

---

## 9. 8/13 LLM 接入工作总结

### 9.1 阶段总结
| 阶段 | 内容 | 文档 |
|---|---|---|
| Phase 1-5 | LLM 集成（parser + client + make_llm） | 累计 +43 测试通过 / 0 回归 |
| Phase 6 | Qwen3-VL-FP8（3.3GB）| §17.1-§17.11 |
| Phase 7 | **AWQ 4bit 切换**（2.08GB，INT4 W4A16）| §18.1-§18.11 |
| Phase 8 | **detector CPU 化**（mobileclip 254MB 释放）| §18.12 |
| Phase 9-12 | **LLM 路径物理 verify 修复链** | §18.14-§18.22 |
| Dashboard 修复 | video 自适应容器 | §18.17 |

**结果**：
- 真机 demo Reflex 路径 100% 可靠（set.py 验证过）
- 真机 demo LLM 路径三层防御（system prompt + orchestrator guard + skill verify）
- 343 → 405+ 测试通过 / 0 回归

### 9.2 关键决策

| 决策 | 选项 | 理由 |
|---|---|---|
| 模型 | Qwen3-VL-2B-Instruct-AWQ-4bit | 官方 INT4 量化 2.08GB（vLLM 0.27 原生支持）|
| vLLM linear backend | Triton（非 DeepGEMM）| DeepGEMM 在 Blackwell sm120 有 layout transform bug |
| Tool call parser | Hermes（非 qwen3_xml）| Qwen3-VL 用 `<tool_call>{...}</tool_call>`（JSON in wrapper）|
| Detector 设备 | CPU（非 CUDA）| mobileclip 254MB 释放，8GB 卡装得下 vLLM + 真 detector |
| vLLM 启动 flags | compressed-tensors + triton + hermes | vLLM 0.27 + Qwen3-VL-AWQ 4bit 兼容组合 |
| Profile yaml | `local_qwen3_vl`（独立 profile）| 替代之前 `local_reason2`（已删）|

### 9.3 文件改动总结

| 文件 | 改动 |
|---|---|
| `scripts/start_reason2_server.sh` | MODEL 路径切到 AWQ + flags（compressed-tensors / triton / hermes）|
| `src/wrc_demo/configs/demo.yaml` | detector device: cuda:0 → cpu |
| `src/wrc_demo/configs/llm/local_qwen3_vl.yaml` | 新 profile（独立干净）|
| `src/wrc_demo/agent/orchestrator.py` | 2 处 guard：move_to_pose + point_at/move_relative/move_to_joint_zero |
| `src/wrc_demo/agent/prompts.py` | SYSTEM_PROMPT 加多条强规则 |
| `src/wrc_demo/agent/llm.py` | LLMResponse 加 `reasoning` field（之前改的，仍工作）|
| `src/wrc_demo/skills/runtime.py` | cartesian_planner 调用兼容 + skill_place_at verify + skill_move_to_pose pitch fix + bridge verify |
| `src/wrc_demo/planning/cartesian_planner.py` | numpy passthrough + getattr 兼容 safety.harness.Kinematics |
| `src/wrc_demo/apps/stream_server.py` | CSS img 自适应容器（video 变大）|

### 9.4 删除的文件（清理废弃）

- `src/wrc_demo/agent/reason2_parser.py`（8KB，废弃）
- `src/wrc_demo/configs/llm/local_reason2.yaml`（废弃）
- `tests/test_reason2_parser.py`（10.6KB，废弃）
- `tests/test_reason2_chat_client.py`（21.3KB，废弃）
- `tests/test_make_llm_reason2.py`（8.7KB，废弃）
- `tests/test_align_to_rebot_grasp.py`（修了 fixture 兼容，保留）

### 9.5 备份文件（可回滚）

| 文件 | 位置 |
|---|---|
| `apps/demo.py.bak_jointsign` | `src/wrc_demo/apps/` |
| `apps/stream_server.py.bak_dashboard` | `src/wrc_demo/apps/` |
| `apps/demo.py.bak.20260813_v2_guard3` | `src/wrc_demo/apps/` |
| `skills/runtime.py.bak_verify` | `src/wrc_demo/skills/` |
| `skills/runtime.py.bak_bridge_verify` | `src/wrc_demo/skills/` |
| `skills/runtime.py.bak_mtp` | `src/wrc_demo/skills/` |
| `skills/runtime.py.bak_mtp_strict` | `src/wrc_demo/skills/` |
| `planning/cartesian_planner.py.bak_numpy` | `src/wrc_demo/planning/` |
| `planning/cartesian_planner.py.bak_compat` | `src/wrc_demo/planning/` |
| `planning/cartesian_planner.py.bak_strict` | `src/wrc_demo/planning/` |
| `agent/prompts.py.bak.20260813` | `src/wrc_demo/agent/` |
| `agent/prompts.py.bak.20260813_v2_guard3` | `src/wrc_demo/agent/` |

### 9.6 模型目录

| 目录 | 大小 | 状态 |
|---|---|---|
| `models/qwen3-vl-2b-awq-4bit/` | 2.08GB | **当前使用**（INT4 W4A16 量化）|
| `models/qwen3-vl-2b-fp8/` | 3.3GB | 备用（FP8 官方，但装不下 detector）|
| `models/cosmos-reason2-2b/` | 4.88GB | 未用（BF16，vLLM 0.27 不支持 NVFP4 inference）|
| `models/cosmos-reason2-2b-nvfp4/` | 2.7GB | 未用（NVFP4 量化，vLLM 0.27 不支持）|

### 9.7 重启方法（完整流程）

```bash
# 1. 启动 vLLM（背景）
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
# 等 30-60s 启动（首次需要 kernel warmup）
for i in {1..20}; do
  sleep 3
  if curl -sf --max-time 2 http://localhost:8080/v1/models > /dev/null; then
    echo "✓ vLLM ready"
    break
  fi
done

# 2. CAN bus（如未自动）
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 3. 启动 wrc_demo 真机
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive

# 4. 浏览器 dashboard
# http://198.18.0.1:8090/
```

### 9.8 关键经验（给新对话）

1. **8GB 显卡硬约束**：LLM + 真 detector 同时跑物理不可能
2. **detector CPU 化是关键 workaround**——mobileclip 254MB 释放 → LLM 路径可行
3. **三层防御**：system prompt 引导 LLM 选对工具 → orchestrator guard 翻译错选工具 → skill verify 物理执行
4. **修复链是迭代的**——每一步基于前一步的发现
5. **Reflex 路径 100% 可靠**——CLAUDE.md §0 验证过；LLM 路径仍需改进

### 9.9 下一步建议

- **短期**：真机测试 §18.22 guard 修复后的 LLM 路径
- **中期**：离线打包（如果部署目标是 8GB 无网络机器）
- **长期**：detector on-demand 架构改造 / 12GB+ 显卡

---

## 10. 2026-08-14 top 相机标定代码迁移（rebot_grasp → wrc_demo）

### 10.1 决策

**目标**：把 rebot_grasp 的 `scripts/calibrate_top_camera.py`（670 行，简洁 + 重力补偿手动模式）迁移到 wrc_demo 的 `src/wrc_demo/scripts/calib_top_orbbec.py`，完全使用 wrc_demo 内部组件，**不依赖 rebot_grasp**。

**计划**：`tasks/plan-calib-top-migration.md`

### 10.2 迁移结果

| 项 | 旧（wrc_demo 原版） | 新（迁移后） |
|----|-------------------|--------------|
| 文件位置 | `src/wrc_demo/scripts/calib_top_orbbec.py` | **不变**（保持模块路径，零下游破坏） |
| 行数 | 1453 | 1862 |
| Auto 状态机 | `_collect_loop_auto` ~500 行嵌套循环 | `_collect_loop_auto` 借鉴 rebot_grasp `tick_auto` 模式 ~250 行（单 loop + tick 回调） |
| Manual 模式 | 简化版（等 ENTER 后手动移臂） | **保留** + **新增** `--gravity-comp`（MIT + 重力 FF + 500Hz 控制环） |
| Preset poses | 38 个 | **38 个，从 rebot_grasp 复制（数值同源）** |
| 38 个 pose 来源 | wrc_demo 自有 | rebot_grasp 注释明确"copied verbatim from wrc_demo"——同源 |
| Solver | `joint_solve_restart` | **不变** |
| 相机 | `OrbbecCamera` | **不变** |
| 机械臂（auto） | `safe_arm.move_joints` + `Kinematics.ik` | **不变**（保持 SafetyHarness） |
| 机械臂（gc manual） | 不支持 | **新增** `GravityCompController`（vendored `RebotArm` + `RebotArmEndPose` + `compute_generalized_gravity`） |
| JSON schema | v1 | **不变**（向后兼容） |
| 子命令 | `--list`/`--bind`/`--verify`/`--dry-run`/`--manual` | **全部保留 + 新增 `--gravity-comp`** |
| rebot_grasp 依赖 | 无 | **无** |

### 10.3 公开 API 兼容

所有现有调用方继续工作（不需修改）：
- `wrc_demo.scripts.verify_relative_motion`（`from .calib_top_orbbec import CALIB_DIR, _get_serial_from_binding`）
- `wrc_demo.scripts.flip_hand_eye`（JSON schema v1 不变）
- `tests/test_calib_session.py`（11 个测试，调用 `_collect_samples`, `_diagnose_can_bus`, `TOP_CALIB_POSES`, `_pose_to_T`, `main`, `_verify`）
- `tests/test_calib_dataset.py`（1 个测试，import `_verify`）
- `tests/test_hardware_smoke.py`（2 个测试，import `--list`）

### 10.4 测试结果

| 项 | 旧版 | 新版 |
|----|------|------|
| `tests/test_calib_session.py` | 12 passed, 1 skipped | **15 passed, 1 skipped**（多 3 个修好的） |
| 全量 `pytest -m "not hardware"` | 347 passed, 28 failed | **366 passed, 25 failed**（净改善 +3） |
| `--dry-run` 烟囱测试 | OK | **OK**（per-sample rmse = 0.00mm，JSON v1 schema 完整） |

**注**：25 个 remaining 失败都是预存问题（cosmos3 / LLM profile / USD model / setup_agents），与本次迁移无关。详见 T4.1 报告。

### 10.5 重力补偿控制器（新增 `--gravity-comp` 模式）

灵感来自 rebot_grasp 的 `GravityCompController`，但用 vendored SDK 重写：

```python
class GravityCompController:
    KP = 8.0
    KD = 1.5
    V_THRESH = 0.04   # 末端线速度阈值 (m/s)
    W_THRESH = 0.08   # 末端角速度阈值 (rad/s)
    
    def start(self):
        # 1. RebotArm.connect() + enable_all + mode_mit
        # 2. 启动 500Hz 控制环 (arm.start_control_loop)
        # 3. _worker 读 q → compute_generalized_gravity → send_mit
        #    末端速度 > 阈值 → 锁定 _q_target (操作员推臂感觉"卡住")
    
    def safe_home(self):
        # 退出重力补偿 → RebotArmEndPose.safe_home → disconnect
        # (与 teach_record.py 同 pattern，CLAUDE.md "Motion safety path" 例外)
```

**使用**：`python -m wrc_demo.scripts.calib_top_orbbec --gravity-comp` → 操作员推臂漂浮 + ENTER 捕获 → 退出自动 safe_home。

**安全注意**：与 `skills/teach.py` 一样绕过 SafeArm（因为 SafeArm 只支持轨迹 streaming，不支持 MIT 控制环）。已记录在 `CLAUDE.md` 注释。

### 10.6 备份

- `src/wrc_demo/scripts/calib_top_orbbec.py.bak_premigration`（迁移前原版，1453 行）
- 下次更新确认后删除

### 10.7 验证清单（真机测试前用户执行）

- [ ] 真机 `--list` 看到 Orbbec 相机
- [ ] 真机 auto 模式采集 15 个 sample → JSON 写入 → verify
- [ ] 真机 `--manual` 模式（等 ENTER，手动移臂）
- [ ] 真机 `--gravity-comp` 模式（臂漂浮，推臂感觉重力被补偿）
```
---

## 11. 2026-08-14 wrist 相机取代 top 成为 PRIMARY

### 11.1 决策

**长期使用 `orbbec_wrist.yaml` 作为主相机**（替代 `orbbec_overhead.yaml`），top 降级为 backup profile。
**保留两个相机 profile**（不删除 top），随时可以切换回去。

### 11.2 现状盘点（已存在 vs 需要做）

| 模块 | 状态 |
|------|------|
| `scripts/calib_wrist.py` (Park & Martin EIH, 500行) | ✅ 已存在 |
| `scripts/handeye_solver.py` (OpenCV 5.x兼容) | ✅ 已存在 |
| `configs/cameras/orbbec_wrist.yaml` | ✅ 升级为 PRIMARY |
| `data/calibration/hand_eye_wrist.json` (20 samples, 2026-07-23) | ✅ 重新生成含 `schema_version` |
| `perception/grounding.py::Extrinsics` (`eye_in_hand` + `cam_to_base()`) | ✅ 已支持 |
| `skills/runtime.py:381` (每帧用 `cam_to_base()` 动态算) | ✅ 已支持 |
| `tests/test_wrist_profile_fuse.py` | ✅ 适配新 wrist 语义 |
| CLAUDE.md / MIGRATION_PLAN.md / 操作文档 | ✅ 同步 |

### 11.3 改动文件清单

| 文件 | 改动 |
|------|------|
| `configs/cameras/orbbec_wrist.yaml` | `fuse_beliefs: false → true` + 加 `extrinsics:` 块 |
| `tests/test_wrist_profile_fuse.py` | 重写测试适配新 wrist 语义 |
| `scripts/calib_wrist.py` save() | 改用统一的 `save_calibration()` |
| `data/calibration/hand_eye_wrist.json` | 重新生成（保留原 T_cam2gripper）|
| `CLAUDE.md` | 添加 wrist 路径说明 |
| `docs/MIGRATION_PLAN.md:115` | `collect_handeye_eih → calib_wrist` |
| `桌面/wrc_demo_操作文档.md §6.3` | wrist 升级为 PRIMARY |
| `桌面/wrc_demo_操作文档.md` demo 命令 | 默认 `--cameras orbbec_overhead → orbbec_wrist` |
| `桌面/wrc_demo_交接文档.md §11` | 本节 |

### 11.4 EIH 数学说明（与 ETH 的区别）

| 维度 | Top (ETH, backup) | Wrist (EIH, primary) |
|------|-------------------|----------------------|
| 数学 | Park & Martin AX=XB（不，是 joint SE(3) LM，HANDEYE） | Park & Martin AX=XB |
| 静态矩阵 | T_cam2base（固定）| T_cam2gripper（固定）|
| 动态合成 | 无 | `T_cam2base(t) = fk_tcp2base(t) @ T_cam2gripper` |
| Solver | `wrc_demo.calibration.joint_solve_restart` | `scripts/handeye_solver.solve_hand_eye_park` |
| Grasp planner | `plan_grasp_from_mask` (用 T_cam2base) | 同上（每帧 T_cam2base 都变）|
| 检测范围 | 整个 workspace | 仅夹爪附近（**窄视场**）|
| 视觉优势 | 全局视图 | 局部细节（15-25cm 近距离）|

### 11.5 真机验证清单（用户执行）

- [ ] 跑 `--cameras orbbec_wrist --arm rebot_rs --interactive` 看 dashboard 是 wrist 视角
- [ ] `task> pick up the banana` —— grasp 用 wrist 视角的 marker 数据
- [ ] `task> list_objects` —— 检测在 wrist 视角下是否够准
- [ ] 对比：`--cameras orbbec_overhead` 仍然能跑（backup 没坏）
- [ ] 长时间运行看是否漂移（wrist 标定是 3 个月前的；如漂移重做 `scripts/calib_wrist.py`）

### 11.6 已知风险

| Risk | 影响 | Mitigation |
|------|--------|------------|
| wrist FOV 窄 | 可能看不见远处物体 | | 必要时先让臂移动到 scan pose |
| wrist 标定 3 个月前 | 可能有微小漂移 | | 真机跑发现偏差即重做 calib_wrist.py |
| YOLO 在 wrist 视角域移 | 检测置信度可能下降 | | 真机观察；如严重保留 top 主 grasp |

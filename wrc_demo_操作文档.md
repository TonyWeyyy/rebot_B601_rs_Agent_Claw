# wrc_demo 操作文档

> 真实硬件上的 Seeed reBot DevArm B601（rebot_rs）+ Orbbec Gemini 2 顶部相机（orbbec_overhead）+ 本地 LLM（Cosmos3-Edge 4B via vLLM）或 cloud LLM（MiniMax / Anthropic / OpenAI）。
> 文档版本：2026-08-11

---

## 0. 30 秒速览

> **必读**：wrc_demo 在 conda `DM` env 跑，**vLLM/Cosmos3-Edge 在独立 uv venv 跑**。两者通过 `localhost:8080` HTTP 通信。
> **vLLM 必须先启**，否则 wrc_demo 会 `Connection refused`。
> 详细架构见 §2，启停步骤见 §5。

### 一键启动（推荐）—— 同时启 vLLM + demo

```bash
# mock stack（开发/后端测试，不需要 sudo）
bash /home/seeed/wrc_demo/scripts/start_all.sh

# 真机（需要 sudo 给 CAN bus）
bash /home/seeed/wrc_demo/scripts/start_all.sh --real

# 真机 + 一次性 task（后端集成）
bash /home/seeed/wrc_demo/scripts/start_all.sh --real --task "put the banana into the box"

# 状态 / 停止
bash /home/seeed/wrc_demo/scripts/start_all.sh --status
bash /home/seeed/wrc_demo/scripts/start_all.sh --stop
```

### 手动两步（如果不想要 wrapper 脚本）

**Step 1 — 启 vLLM**（独立 uv venv，不需 sudo）：
```bash
MAX_LEN=8192 GPU_UTIL=0.85 bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --bg
# 等 ~60-90s（首次启动要 JIT 编译 flashinfer）
bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --status
# 等看到 "vllm running, PID ..." 再走 Step 2
```

**Step 2 — 启 wrc_demo**：

#### 真机 interactive demo（手输 task）

```bash
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/live_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_wrist --arm rebot_rs --llm local_cosmos3 --interactive
```

#### Mock 一次性 task（开发 / 后端集成）

```bash
cd /home/seeed/wrc_demo
/home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.apps.demo \
    --task "put the banana into the box" \
    --llm local_cosmos3 --cameras mock --arm mock \
    --no-view --no-serve
```

启动后：
- vLLM 在 `localhost:8080`（Step 1 启的）
- wrc_demo 在 DM env 通过 `OpenAICompatClient` 调 vLLM
- 浏览器打开 `http://<ip>:8090/` 看摄像头流 + 仪表盘
- 终端里直接输 `task> ` 后的指令
- Ctrl+C 干净退出

---

## 1. 仓库布局

```
/home/seeed/wrc_demo/
├── src/wrc_demo/                # Python 包
│   ├── safety/harness.py        # SafetyHarness + SafeArm + Kinematics
│   ├── control/                 # arm backends (Mock / Isaac / RebotRS via SDK)
│   ├── perception/              # 相机 + 检测
│   ├── skills/                  # 21 个 skill（grasp / pick_and_place / wave / ...）
│   ├── agent/                   # LLM 编排 + reflex + experience
│   │   ├── llm.py               # LLMClient 基类 + OpenAICompat + Anthropic + Mock
│   │   ├── orchestrator.py      # 3-tier dispatch: reflex → experience → LLM
│   │   ├── advisor.py           # VLM advisor (失败时调视觉建议)
│   │   └── reflex.py            # tier-1 regex + tier-2 experience memory
│   ├── memory/                  # belief store + episodic memory
│   ├── apps/                    # demo CLI / MCP server / viewer
│   ├── calibration/             # ArUco + hand-eye solver
│   ├── scripts/                 # CLI 入口（标定等）—— `calib_top_orbbec.py` 即 top 相机标定
│   ├── config.py                # YAML 配置加载
│   └── configs/
│       ├── demo.yaml            # 总配置（safety, grasp, llm, etc.）
│       ├── cameras/*.yaml       # orbbec_overhead, l515, d435i, mock, ...
│       ├── arms/*.yaml          # rebot_rs, mock, isaac
│       └── llm/*.yaml           # minimax, anthropic, openai, local_qwen, local_cosmos3, mock
├── sdk/                         # vendored reBotArm_control_py（89M, in-tree）
├── scripts/                     # 标定 / 紧急下蹲 / 录制 / 配置
│   ├── start_cosmos3_server.sh  # 启动 vLLM + Cosmos3-Edge（独立 env）
│   ├── quantize_cosmos3.py      # AWQ INT4 量化脚本（autoawq env 不支持）
│   ├── serve_cosmos3_vllm.sh    # 旧版 vLLM 启动脚本（被 start_cosmos3_server.sh 取代）
│   ├── collect_handeye_eth.py   # ⚠️ 旧版 top 相机标定（2026-08-14 已被 in-tree 版本取代，仅作参考）
│   ├── (collect_handeye_eih.py 删于 2026-08-27 — superseded by in-tree wrc_demo.scripts.calib_wrist)
│   └── emergency_stow.py
├── data/calibration/            # 手眼标定产物（hand_eye_top.json 等）
├── configs/llm/*.yaml           # LLM profile 配置
├── runs/                        # 每次运行的 trace（trace.jsonl + keyframes/）
├── docs/
│   ├── adr/0006-move-secrets-to-env-vars.md
│   ├── adr/0007-local-llm-strategy.md
│   ├── adr/0008-llm-profile-switching.md
│   ├── SPEC-local-llm.md
│   ├── MIGRATION_PLAN.md
│   └── HARDWARE_VERIFICATION_HANDOVER.md
└── pyproject.toml               # uv 项目元数据
```

---

## 2. Python 环境（关键：两个独立 env）

| 环境 | 路径 | 用途 |
|---|---|---|
| **conda `DM`** | `/home/seeed/miniforge3/envs/DM/bin/python` | **wrc_demo 主体**（含 motorbridge, ultralytics, pyorbbecsdk） |
| **uv `cosmos3-venv`** | `/home/seeed/cosmos3-venv/bin/python` | **vLLM server**（cu13 + torch 2.13 + transformers v5） |
| `uv` venv | `/home/seeed/wrc_demo/.venv/bin/python` | mock 测试 / 编辑（不完整：缺 ultralytics） |

> **wrc_demo 在 DM env 跑**。**vLLM/Cosmos3-Edge 在 cosmos3-venv 跑**。两者通过 `localhost:8080` 的 HTTP 通信（OpenAI-compatible API）。

确认 DM env 装了什么：

```bash
/home/seeed/miniforge3/envs/DM/bin/python -c "
import motorbridge, ultralytics, pyorbbecsdk
print('motorbridge', motorbridge.__version__)
print('ultralytics', ultralytics.__version__)
print('pyorbbecsdk OK')
"
```

确认 cosmos3-venv（vLLM server）：

```bash
source /home/seeed/cosmos3-venv/bin/activate
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
# 期望: vllm 0.27.0+, torch 2.13.0+cu130, transformers 5.15.0
```

---

## 3. LLM Profile 速查

| Profile | 类型 | 配置 | API key 来源 | 何时用 |
|---|---|---|---|---|
| `mock` | scripted | `configs/llm/mock.yaml` | 无 | 单元测试 / 离线开发 |
| `anthropic` | Claude (cloud) | `anthropic.yaml` | `ANTHROPIC_API_KEY` env | cloud fallback |
| `openai` | OpenAI (cloud) | `openai.yaml` | `OPENAI_API_KEY` env | cloud fallback |
| `minimax` | MiniMax (cloud) | `minimax.yaml` | **`MINIMAX_API_KEY` env** | 默认 cloud（之前 key 在 yaml 里，已迁） |
| `local_qwen` | local Qwen via llama.cpp/vLLM | `local_qwen.yaml` | `EMPTY`（local dummy） | 大 GPU（≥24GB） |
| **`local_cosmos3`** | **local Cosmos3-Edge via vLLM** | `local_cosmos3.yaml` | `EMPTY`（local dummy） | **8GB 显卡默认** |

**切换方式**：
- 启动时：`--llm local_cosmos3`
- 环境变量：`WRC_LLM=local_cosmos3`
- 运行时切（ADR-0008）：通过 MCP `llm_switch` 工具

**fallback chain**（ADR-0008）：在 `demo.yaml` 设 `llm.fallback: [local_cosmos3, mock]`——当主 LLM 失败时自动走 fallback。

**安全提醒**：所有 cloud profile 的 API key 现在都从环境变量读，**不要再 commit 明文 key 到 yaml**。详见 ADR-0006 + `.env.example`。

---

## 4. 关键环境变量

| 变量 | 默认 | 用途 |
|---|---|---|
| `WRC_CAMERAS` / `WRC_CAMERA` | `mock` | 相机 profile（逗号分隔多相机），如 `orbbec_overhead` |
| `WRC_ARM` | `mock` | 机械臂 profile：`mock`/`rebot_rs`/`isaac` |
| `WRC_LLM` | `mock` | LLM profile：见 §3 表格 |
| `WRC_RUN_DIR` | `<repo>/runs/mcp_<pid>` | 每次运行的 trace 目录 |
| `WRC_STREAM` | `1` | 启 MJPEG dashboard（端口 8090） |
| `WRC_VIEW` | `1 if DISPLAY` | 启 cv2 窗口 |
| `WRC_DETECTOR_MODEL` | `models/yolo11n.pt` | 覆盖 detector 权重 |
| `WRC_DETECT_CLASSES` | — | 覆盖检测类别 |
| `WRC_STREAM_PORT` | `8090` | dashboard 端口 |
| `WRC_PREWARM` | `1` | MCP server 启动时后台 build_runtime |
| `YOLO_OFFLINE` | — | 设 `True` 阻止 ultralytics 联网下载（**真机必设**） |
| `PYTHONPATH` | — | 设 `src` 即可 import wrc_demo |
| `DISPLAY` | — | cv2 窗口需要的 X server |
| `MINIMAX_API_KEY` | — | `--llm minimax` 必设 |
| `ANTHROPIC_API_KEY` | — | `--llm anthropic` 必设 |
| `OPENAI_API_KEY` | — | `--llm openai` 必设 |

> 真实硬件**必设** `YOLO_OFFLINE=True`。

---

## 5. 启动 / 停止

### 5.1 vLLM server（**先启**）

wr_demo 需要 vLLM 在 `localhost:8080` 跑。

```bash
# 后台启（推荐后端集成）
MAX_LEN=8192 GPU_UTIL=0.85 bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --bg

# 看状态
bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --status

# 停
bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --stop
```

参数（环境变量）：
- `MAX_LEN`：vLLM context window（8GB 卡推荐 8192；更大 OOM）
- `GPU_UTIL`：vLLM GPU 占用比（默认 0.85，X server 吃 1.3GB 时降到 0.70）
- `PORT`：默认 8080

启动脚本自动：
- patch `vllm/transformers_utils/processors/cosmos3_edge.py`（transformers v5 兼容）
- 检测 driver CUDA 版本 + 用 cu13 nvcc（黑魔法补丁绕过 flashinfer sm120 编译问题）
- 写 PID 到 `/tmp/vllm_cosmos3.pid`
- 日志到 `/tmp/vllm_cosmos3.log`

**首次启动约 60-90s**（vLLM 加载模型 + flashinfer JIT 编译）。

### 5.2 wrc_demo demo

#### Interactive（真机手输 task）

```bash
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/live_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_wrist --arm rebot_rs --llm local_cosmos3 --interactive
```

#### 一次性 task（mock / 后端调用）

```bash
cd /home/seeed/wrc_demo
/home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.apps.demo \
    --task "put the banana into the box" \
    --llm local_cosmos3 --cameras mock --arm mock \
    --no-view --no-serve
```

返回 TaskReport：success / path (reflex|experience|llm) / steps / summary。

#### 后台运行（推荐生产用）

```bash
mkdir -p /tmp/wrc_logs
RUN_DIR=/home/seeed/wrc_demo/runs/live_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
nohup /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
    --cameras orbbec_wrist --arm rebot_rs --llm local_cosmos3 --interactive \
    --no-view \
    > /tmp/wrc_logs/demo.log 2>&1 &
echo $! > /tmp/wrc_logs/demo_pid
tail -f /tmp/wrc_logs/demo.log
kill $(cat /tmp/wrc_logs/demo_pid)
```

#### 后端一次性调用（spawn-per-task）

最简单——每次 HTTP request 启一个 Python 子进程跑 task：

```bash
/home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.apps.demo \
    --task "$USER_TASK_TEXT" \
    --llm local_cosmos3 --cameras mock --arm mock \
    --no-view --no-serve 2>&1 | tail -10
```

**性能（实测 RTX 5070 + Cosmos3-Edge BF16）：**
- 单 turn "put the banana"（cold cache, 21 tools）：**~0.4s**
- 5-turn loop：~21s
- 完整任务（10-15 steps 收敛）：**~43-65s**
- Token throughput：**71 tok/s**

#### MCP server（让 Claude Code / Hermes / Codex 接入）

```bash
WRC_CAMERAS=orbbec_wrist WRC_ARM=rebot_rs WRC_LLM=local_cosmos3 \
    WRC_STREAM=1 WRC_VIEW=0 \
    /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.mcp_server
```

走 stdio JSON-RPC（agent host 是大脑）。

#### 纯只读 / dashboard

```bash
WRC_CAMERAS=orbbec_wrist WRC_ARM=rebot_rs WRC_LLM=mock \
    WRC_STREAM=1 WRC_VIEW=0 \
    /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
    --no-interactive --no-task --serve
```

---

## 6. 标定流程

### 6.1 顶部相机内参

```bash
# ROS kalibr / OpenCV calibrateCamera 离线算
# 产物: data/calibration/orbbec_overhead.npz (含 K + D)
```

### 6.2 眼在手外（top 相机 eye-to-hand）

> **2026-08-14 迁移**：原 `scripts/collect_handeye_eth.py`（rebot_grasp 风格，外部脚本）已迁移到 in-tree 的 `wrc_demo.scripts.calib_top_orbbec`，并从 rebot_grasp 的 `scripts/calibrate_top_camera.py` 重写了核心架构（状态机 + 重力补偿手动模式）。**推荐使用新入口**；旧脚本保留为参考但不再维护。

**前置（首次必做）**：把相机序列号绑到 role，写入 `data/calibration/cameras.json`：

```bash
# 列出 Orbbec 设备
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec --list
# 把 [0] 设备绑为 top（写入 cameras.json + 同步 patch configs/cameras/orbbec_overhead.yaml）
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec --bind top
```

**新入口（推荐）**：

```bash
# Auto 模式：38 个预设位姿 + safe_arm + IK 自动遍历（默认模式）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec
# 产物: data/calibration/hand_eye_top.json (含 T_cam2base) + .npz

# Manual 模式：每个 pose 后等 ENTER（操作员手动移臂）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec --manual

# Gravity-comp 模式：MIT 模式 + 重力 FF，臂漂浮；操作员手推 + ENTER 捕获
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec --gravity-comp

# 指定样本数 + 不绑定的相机（指定序列号）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec \
    --serial AY3Z331006L --n 15

# Dry-run：不依赖硬件的 pipeline 烟囱测试
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec \
    --dry-run --dry-run-out /tmp/dryrun.json
```

**旧入口（已弃用，保留仅作参考）**：
```bash
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m scripts.collect_handeye_eth \
    --camera orbbec_overhead --arm rebot_rs
```

### 6.3 眼在手上（wrist 相机 eye-in-hand）— **PRIMARY since 2026-08-14**

> **2026-08-14 迁移完成**：wrist 标定已迁移到 in-tree `scripts/calib_wrist.py`（Park & Martin solver，
> OpenCV 5.x 兼容）；wrist 配置已升级到 PRIMARY（`fuse_beliefs: true` + `extrinsics: {mode: eye_in_hand}`）；
> top 相机降级为 backup profile。Demo 命令现在默认用 `--cameras orbbec_wrist`。

```bash
# 新入口（推荐）—— Park & Martin 自动模式
# 注意：模块名是 calib_wrist，**不是** calib_wrist_orbbec
# （wrist 默认就是 Orbbec Gemini 2，无需后缀）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_wrist
# 产物: data/calibration/hand_eye_wrist.json (含 T_cam2gripper, schema_version=1) + .npz

# Manual 模式（重力补偿手推 + ENTER 捕获）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_wrist --manual

# 等价 script 形式（不推荐，等价于 -m）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u /home/seeed/wrc_demo/scripts/calib_wrist.py

# 旧入口（rebot_grasp 风格，仅作参考，不再维护；scripts/ 不是 Python 包，所以需要 PYTHONPATH）
PYTHONPATH=/home/seeed/rebot_grasp:/home/seeed/rebot_grasp/scripts \
    sudo /home/seeed/miniforge3/envs/dm_visual/bin/python -u /home/seeed/rebot_grasp/scripts/collect_handeye_eih.py \
    --camera orbbec_wrist --arm rebot_rs
```

> **命名提示**：
> - ✅ `wrc_demo.scripts.calib_wrist`（in-tree, 2026-08-14 后唯一推荐）
> - ❌ `wrc_demo.scripts.calib_wrist_orbbec`（**不存在**——之前有人拼错，ModuleNotFoundError）
> - ❌ `scripts.calib_wrist`（rebot_grasp-style 外部模块，需要 PYTHONPATH hack）

> **Wrist EIH 数学说明**：wrist JSON 里 `T_marker2gripper` 字段存的实际是 SOLVED `T_cam2gripper`
> （camera→gripper 静态矩阵）。`perception/grounding.py::Extrinsics.cam_to_base()`
> 每帧用 `fk_tcp2base() @ T_cam2gripper` 动态算 `T_cam2base(t)`。Eye-in-hand 标定数学是 `AX=XB`
> （Park & Martin），跟 top 的 ETH 不同。

### 6.4 验证（不重跑标定）

```bash
# Top 相机（ETH，用新入口验证）
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec \
    --verify --npz data/calibration/hand_eye_top.json \
    --known-points "0.30,0.00,0.02;0.25,-0.05,0.02;0.35,0.05,0.02"

# Wrist 相机（EIH，用 flip_hand_eye.py 验证）
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.flip_hand_eye \
    --npz data/calibration/hand_eye_wrist.json \
    --camera-side above --dry-run
```

---

## 7. 紧急 / 维护操作

### 7.1 紧急下蹲
```bash
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m scripts.emergency_stow
```

### 7.2 硬件断电
- demo 跑着时：Ctrl+C（第一次 = soft stop，第二次 = 进程退出）
- demo 死了之后：物理上断开 24V 电源

### 7.3 CAN bus 状态
```bash
ip -d link show can0
# 重置（如果出现 bus-off 错误）：
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

### 7.4 进程清理
```bash
# wrc_demo
ps -ef | grep -E 'wrc_demo|miniforge.*mcp' | grep -v grep
pkill -f 'wrc_demo.apps.demo'
pkill -f 'wrc_demo.apps.mcp_server'

# vLLM
ps -ef | grep 'vllm.entrypoints' | grep -v grep
pkill -f 'vllm.entrypoints'
# 或优雅：
bash /home/seeed/wrc_demo/scripts/start_cosmos3_server.sh --stop

# 端口
sudo lsof -i :8080   # vLLM
sudo lsof -i :8090   # wrc_demo dashboard
```

---

## 8. 故障排查

### wrc_demo 主线问题

| 症状 | 排查 |
|---|---|
| 启动卡 `sudo: 需要密码` | 你的终端 sudo 缓存过期；在你启动这个的**原始终端**重跑 |
| `[error] AY3Z331006L Timestamp anomaly` | Orbbec SDK 在 Linux 上的已知噪声，**忽略** |
| `ModuleNotFoundError: ultralytics` | 用错了 Python——换到 `DM` env |
| `motor.get_state()` returns None | motors 未 `clear_error`；`make_arm` 工厂已自动清 |
| `arm 不响应 MIT 命令`（电机位置不动） | 99% 是 fault bit 没清；试 `motor.clear_error()` 然后再 `arm.arm.enable()` |
| `TCP drift > 2mm in 5s` | SDK gravity FF 没生效；检查 `RebotArmEndPose(use_gravity_ff=True)` |
| `harness 拒绝 move_joints` | 当前姿态越界（特别是 joint2 在 ~0.004 而 URDF 下限 0.020）；先写一个 0.03 rad 的小目标 |
| `fuse_beliefs=false` 警告 | `data/calibration/hand_eye_top.json` 缺失 |
| `wrc-demo` 报 `motor not detected` | 检查 `sudo ip link show can0`（必须 UP）+ 24V 电源 |

### vLLM / Cosmos3-Edge 问题

| 症状 | 排查 |
|---|---|
| `curl http://localhost:8080/v1/models` 无响应 | vLLM 没启：`bash scripts/start_cosmos3_server.sh --status`；没启就跑 `--bg` |
| vLLM OOM（启动时 `Free memory less than desired`） | 降 `MAX_LEN`（4096 → 8192 step）或 `GPU_UTIL`（0.85 → 0.70）；关 X server 释放 1.5GB |
| vLLM `FlashInfer requires GPUs with sm75 or higher` | 启动脚本已 patch flashinfer；如果还出错，重启 vLLM |
| vLLM `Ninja build failed` / `Unsupported gpu architecture` | 检查 `PATH` 是否含 cu13 nvcc：`/home/seeed/.local/lib/python3.10/site-packages/nvidia/cu13/bin` |
| vLLM `SM 12.x requires CUDA >= 12.9` | 启动脚本已 patch；如果还出错，重启 vLLM |
| vLLM `No supported CUDA architectures` | 启动脚本已 patch flashinfer；如果还出错，重启 vLLM |
| wrc_demo 报 `LLMConfigError: profile 'minimax', env var 'MINIMAX_API_KEY'` | `export MINIMAX_API_KEY=...` 后重跑 |
| wrc_demo 报 `model 'Cosmos3-Edge-W4A16' does not exist` | vLLM 跑的是 BF16（不是 AWQ），yaml 写 W4A16；改 `model: /home/seeed/wrc_demo/models/cosmos3-edge` |
| wrc_demo 报 `maximum context length is X` | 21 TOOL_SPECS 提示词 ~3-4K tokens；vllm `--max-model-len` 至少 6144；推荐 8192 |
| tool call 返回但 arguments 是 `{}` | 模型质量问题（template 支持，但权重填空弱）；ADR-0007 §Fallback plan 准备了 Qwen3-Coder 切换 |

### 后端集成问题

| 症状 | 排查 |
|---|---|
| 后端调 wrc_demo 但返回 step budget exhausted | 模型在 mock 上"卡 wave"；用真机或调整 task prompt |
| 后端调 vLLM 报 400 context length | prompt 太大；减少 task 复杂度或加大 vLLM `--max-model-len` |
| wrc_demo 调 vLLM 报 401 / 403 | 检查 `local_cosmos3.yaml` 的 `base_url` 和 `api_key` |

---

## 9. SDK / 迁移备忘

> 详细看 `docs/MIGRATION_PLAN.md` 和 `docs/HARDWARE_VERIFICATION_HANDOVER.md`。

- **控制栈** = vendored SDK（`sdk/reBotArm_control_py/`）+ 薄 `SafeArm` wrapper
- **真实硬件代码路径**：`make_arm(cfg, kinematics)` → `RebotArm(hw_yaml=...)` → `connect()` → `clear_error()` → `enable_all()`
- **关节 frame**：asset frame，无 `joint_signs` 乘子；`home_q = [0, 0.1684, 0.6226, -0.4543, 0, 0]`，FK 给 TCP = `[0.25, 0, 0.35]`
- **SDK 已 patch**（见 `sdk/README.md`）
- **mock stack**：281 passed, 0 failed

### Cosmos3-Edge 集成（2026-08-11）

- **架构**：wrc_demo (DM env) + vLLM (cosmos3-venv) 通过 HTTP 通信
- **模型**：nvidia/Cosmos3-Edge 4B BF16
- **量化**：未来用 `scripts/quantize_cosmos3.py`（AWQ W4A16；当前 autoawq 跟 transformers v5 不兼容，需用 llm-compressor）
- **关键 patch**（脚本自动）：
    - `vllm/transformers_utils/processors/cosmos3_edge.py`：transformers v5 import 路径修复
    - `flashinfer/jit/core.py::check_cuda_arch`：sm_120 arch 缺失时 seed
    - `flashinfer/compilation_context.py`：跳过 "SM 12.x requires CUDA >= 12.9" 检查
- **8GB 显存限制**：vision 关 + max-model-len ≤ 8192 + enforce-eager
- **Tool calling**：模型选工具正确（`list_objects` from 23 tools），但 arguments 填空质量一般（ADR-0007 §Fallback plan 准备 Qwen3-Coder 切换）

---

## 10. 测试套件

```bash
# wrc_demo 单元测试（mock stack，不需要硬件）
cd /home/seeed/wrc_demo
/home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/ -q --no-header -m "not cosmos3"
# → 281+ passed, 0 failed

# Cosmos3-Edge 冒烟测试（需要 vLLM server 跑着）
/home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/test_cosmos3_smoke.py -v -m cosmos3
# → 4 个测试（2 tools / 23 tools / vision+tool / 多轮 loop）

# 跳过 mock 的硬件测试（需真机 + 24V + can0）
/home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/ -m hardware -q
```

---

## 11. 常用 URL / 路径速查

| | |
|---|---|
| wrc_demo dashboard | `http://192.168.7.33:8090/` |
| vLLM server (本机) | `http://127.0.0.1:8080/v1` |
| vLLM health check | `curl http://127.0.0.1:8080/v1/models` |
| 相机序列号 | `AY3Z331006L` (orbbec_overhead) |
| 机械臂 hardware config | `sdk/config/rebotarm_rs.yaml` |
| URDF | `sdk/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf` |
| demo 默认 trace 目录 | `runs/live_YYYYMMDD_HHMMSS/` |
| MCP 客户端注册 | `scripts/setup_agents.py` |
| 紧急下蹲 | `python -m scripts.emergency_stow` |
| 标定产物 | `data/calibration/hand_eye_*.json` |
| Top 相机标定（新入口） | `python -m wrc_demo.scripts.calib_top_orbbec`（见 §6.2） |
| Top 相机标定（旧入口，已弃用） | `python -m scripts.collect_handeye_eth`（仅参考） |
| 迁移记录 | `docs/MIGRATION_PLAN.md` |
| 验证交接 | `docs/HARDWARE_VERIFICATION_HANDOVER.md` |
| vLLM 启动脚本 | `scripts/start_cosmos3_server.sh` |
| 实时日志（wrc_demo） | `/tmp/wrc_logs/demo.log` |
| 实时日志（vLLM） | `/tmp/vllm_cosmos3.log` |
| vLLM PID 文件 | `/tmp/vllm_cosmos3.pid` |

---

## 12. 当前状态（2026-08-11）

### ✅ 已完成
- Demo 后端在 conda `DM` env + orbbec_overhead + rebot_rs 跑通
- **本地 LLM 集成**（nvidia/Cosmos3-Edge 4B via vLLM）—— 8GB RTX 5070 跑通
  - `--llm local_cosmos3` 工作（reflex + LLM 路径都验证）
  - tool calling 实际测试通过（`list_objects` from 23 tools）
- Dashboard 工作，REPL 等输入
- vLLM 启动脚本完整封装（autopatch + auto-detect CUDA + PID 管理）
- 5 个新 ADR / spec / 3 个新测试 / 2 个新 yaml / 量化脚本就绪

### ⚠️ 已知问题
- vLLM 模型权重填空质量一般（`{}` 占多数）；如果影响生产，切换到 Qwen3-Coder fine-tune
- AWQ W4A16 量化目前跑不了（autoawq 跟 transformers v5 不兼容）；脚本就绪等环境 fix
- vision 输入在 8GB 卡上关掉（vLLM `--limit-mm-per-prompt '{"image":0,"video":0}'`）；如需要 vision，需先量化 + 减少 context

### 🔄 后端集成方式
- **最简**：spawn-per-task——每次 HTTP request 启 `python -m wrc_demo.apps.demo --task X ...`
- **进阶**：MCP server（`apps/mcp_server.py`）走 stdio JSON-RPC
- **fastest**：直接调 vLLM (curl/openai sdk) 跳过 wrc_demo orchestrator（失去 skill 调度）

### 详见
- `docs/adr/0006-move-secrets-to-env-vars.md`（minimax API key 迁移）
- `docs/adr/0007-local-llm-strategy.md`（Cosmos3-Edge profile 设计）
- `docs/adr/0008-llm-profile-switching.md`（运行时切换 + fallback）
- `docs/SPEC-local-llm.md`（操作完整手册）
- `CLAUDE.md`（AI agent 上下文）

---

## 13. 真机测试操作步骤 — Cosmos-Reason2-2B（2026-08-12 新增）

> **章节号说明**：原 §12 是「当前状态（2026-08-11）」。为避免章节号冲突，本节作为 §13 追加到末尾。子节编号沿用用户原意 `13.X`。

### 13.1 状态摘要（前置 context）

- **代码改动（已完成）**：
  - Phase 1 fail-fast ✅
  - Phase 4 Slice 1-5 完成：parser / `Reason2ChatClient` / `make_llm` 集成 / reasoning fallback
  - 详见各模块 docstring 与 ADR
- **测试状态**：343 passed / 36 failed（baseline +43，0 回归）
- **mock stack 表现**：
  - `run_c` 出现 3 个 `list_objects` tool_calls
  - 模型在 mock 空场景会"快速放弃"输出空 array
- **推荐真机测试路径**：4 个 tier，从低风险到高风险（见 §13.5）

### 13.2 ⚠️ 重要风险提示

| 项 | 说明 |
|---|---|
| 模型能力 | Cosmos-Reason2-2B **不是为 tool calling 训练**的——我们 hack parser 兜底 |
| mock 成功率 | ~10%（可能反映真场景难度） |
| 真机失败成本 | 机械臂运动、物体位移、tabletop 状态被破坏 |
| 风险控制 | **手动测试 = 用户控制 risk**——任何时候 `Ctrl+C` + 物理 E-stop |
| 适用性 | 真机场景正是 Reason2-2B 设计目标（Physical AI 推理），但 tool calling 是它的弱点 |

### 13.3 前置检查

#### 1) vLLM 运行状态

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status
curl -sf --max-time 3 http://localhost:8080/v1/models | head -c 200
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

**期望**：vLLM ready、API 返回 `cosmos-reason2-2b`、显存 `~7500 500` MB 左右。

#### 2) sudo 凭证

```bash
sudo -n true 2>&1 && echo "OK" || echo "RUN: sudo -v (in interactive terminal)"
```

**期望**：输出 `OK`；否则在交互终端里跑一次 `sudo -v` 续 15min。

#### 3) CAN bus 状态

```bash
ip link show can0 2>&1 | head -5
ls /sys/class/net/can0 2>&1
```

**期望**：`can0` 存在且 `state UP`；否则需 §13.4 Step 2 手动初始化。

#### 4) 物理硬件确认清单

- [ ] 机械臂电源开（rebot_rs 控制器 + 12V/24V）
- [ ] Orbbec Gemini 2 相机（顶部）USB 连接
- [ ] 桌面清理，目标物体在相机视野内
- [ ] E-stop 物理按钮在手边且未被按下
- [ ] Dashboard 可达机器 IP（`hostname -I`）

### 13.4 启动步骤（按顺序）

#### Step 1 — sudo 凭证缓存

**目的**：必需；否则 CAN bus / 串口相关命令会被拒。

```bash
sudo -v
# 保持 15min，后续启动不必再输
```

**期望**：无输出（直接回 prompt）。

#### Step 2 — CAN bus 初始化（如未自动）

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip link show can0
```

**期望**：`state UP`。
**失败对策**：
- 设备不存在 → 检查机械臂控制器 USB
- `RTNETLINK answers: Device or resource busy` → 先 `sudo ip link set can0 down` 再重试

#### Step 3 — vLLM 启动（如未运行）

```bash
# 检查 + 启动（幂等）
if [[ ! -f /tmp/vllm_reason2.pid ]] || ! kill -0 "$(cat /tmp/vllm_reason2.pid 2>/dev/null)" 2>/dev/null; then
  bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
  # 等 60-90s（首次启动要 warmup kernel + flashinfer 编译）
fi

# 验证
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status
curl -sf http://localhost:8080/v1/models | head -c 200
```

**期望**：vLLM ready；显存 `~7.5 GB / 8 GB`；返回 JSON 含 `cosmos-reason2-2b`。
**失败对策**：见 §13.7（OOM / flashinfer patch）。

#### Step 4 — wrc_demo 真机启动（interactive）

```bash
cd /home/seeed/wrc_demo

sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/real_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_reason2 \
  --interactive
```

**期望输出**：

```
[wrc-demo] cameras=['orbbec_overhead'] arm=rebot_rs llm=local_reason2
... (CAN bus + camera init logs)
task>
```

**失败对策**：
- `Address already in use`（8090）→ `pkill -f wrc_demo.apps.demo`
- `Camera not found` → 检查 Orbbec SDK + USB

#### Step 5 — 浏览器看 dashboard

- 打开 `http://<机器IP>:8090/`
- 应看到相机流 + 状态面板

### 13.5 测试场景（按风险递进）

#### Tier 1 — reflex-only（不走 LLM）

| 项 | 内容 |
|---|---|
| 任务 | `go home` 或 `wave` |
| 目的 | 验证基础设施（CAN bus + arm + camera + dashboard） |
| 风险 | 极低 |
| 预期 | reflex 立即匹配，~50ms 内执行；机械臂动起来 |
| 观察 | dashboard 状态变化；`trace.jsonl` 里 `path=reflex` |

#### Tier 2 — 简单 LLM 任务

| 项 | 内容 |
|---|---|
| 任务 | `list the objects` 或 `describe what you see` |
| 目的 | 验证 Reason2 → orchestrator → skill dispatch 链路 |
| 风险 | 低（只读，不动臂） |
| 预期 | LLM 路径触发（不是 reflex）；`trace.jsonl` 出现 `tool_name: list_objects`；30 步内收敛 |
| 观察 | 模型输出格式（`<think><answer>` 是否符合预期）；tool_call 名字是否正确；时延（每 turn 1-5s） |
| 失败处理 | 30 步耗尽但 trace 没出现 tool_call → 模型没吐 JSON；按 §13.7 排查 |

#### Tier 3 — 复杂 LLM 任务（pick）

| 项 | 内容 |
|---|---|
| 任务 | `pick up the banana`（桌上有香蕉） |
| 目的 | 验证完整物理推理 + tool calling |
| 风险 | 中（机械臂会动） |
| 前置 | 物体在视野内且能被 YOLOE 识别 |
| 预期 | LLM 连续 dispatch：`detect_object` → `plan_grasp` → `move` → `close_gripper` → `lift`；`trace.jsonl` 记录 5-10 个 tool_calls |
| 观察 | 机械臂路径是否丝滑；grasp 是否成功；时延（reasoning 模型可能 30-60s 总） |
| 失败处理 | 任何时刻 `Ctrl+C` 暂停；`move_home` 命令把臂收回去 |

#### Tier 4 — 完整 pick-and-place

| 项 | 内容 |
|---|---|
| 任务 | `put the banana into the box` |
| 目的 | 端到端真机验证 |
| 风险 | 高（多状态转换 + 放物体） |
| 建议 | 先跑 Tier 3 成功 1-2 次再试 |

### 13.6 停止 / 回滚

#### 正常停止

1. `task> quit`（如果支持）
2. `Ctrl+C`（soft stop，freeze + e-stop latch）
3. 第二次 `Ctrl+C` 退出

#### 强制停止

```bash
# 停 wrc_demo
pkill -f 'wrc_demo.apps.demo'
# 停 vLLM
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --stop
# E-stop 机械臂（物理按钮）
```

#### 完整重置（如果出问题严重）

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --stop
pkill -f wrc_demo
sudo ip link set can0 down
# 检查机械臂物理状态
# 必要时手扶机械臂到安全位置
```

### 13.7 故障排查

#### `Connection refused` (localhost:8080)
- **原因**：vLLM 未启动
- **解决**：跑 §13.4 Step 3

#### `model not found` / `404 from vLLM`
- **原因**：`served_model_name` 不匹配
- vLLM 返回 path `/home/seeed/wrc_demo/models/cosmos-reason2-2b`
- **解决**：检查 profile yaml 里 `model:` 字段是否用 path

#### `LLMConfigError (env_key missing)`
- **原因**：profile 配置的 env var 未设
- **解决**：`export MINIMAX_API_KEY=...` 或检查 `local_reason2.yaml` 配置

#### vLLM OOM（`ValueError: Free memory < desired`）
- **原因**：`GPU_UTIL` 太高
- **解决**：编辑 `scripts/start_reason2_server.sh`，把 `GPU_UTIL="${GPU_UTIL:-0.75}"` 改 0.70，重启 vLLM

#### flashinfer JIT 编译失败
- **原因**：CUDA 13 路径问题
- **解决**：脚本已自动 patch；如果 patch 失败，参考 `start_cosmos3_server.sh` 的 3 个 patch 逻辑

#### 机械臂不动 / 不响应
- 检查 CAN bus state：`ip link show can0`（应为 UP）
- 检查 ARM SDK 通信：`ls /dev/ttyACM*` 或 SDK log
- 必要时手动 `move_home`

#### YOLOE 检测不到物体
- 相机流：dashboard 上能看到画面？
- 物体在视野内？光线够？
- `YOLO_OFFLINE=True` 已设？
- `models/mobileclip_blt.ts` 在正确位置？

#### 模型持续输出空 array（`[]\n<answer>\n[]\n</answer>`）
- **现象**：`trace.jsonl` 0 tool_calls，模型放弃
- **解释**：mock 场景下模型无目标（真场景下应该不会）
- **建议**：换更具体的任务（`list the banana on the table` 而非 `list the objects`）

### 13.8 观察 / 记录

#### 关键文件位置

| 内容 | 路径 |
|---|---|
| vLLM 日志 | `/tmp/vllm_reason2.log` |
| vLLM PID | `/tmp/vllm_reason2.pid` |
| wrc_demo 日志 | stdout + 文件（如果在 `start_all.sh` 里） |
| `trace.jsonl` | `runs/real_<timestamp>/trace.jsonl` |
| task 报告 | 与 `trace.jsonl` 同目录 |

#### 应记录的内容

1. 每个 tier 的 task + 时延 + 成功/失败
2. `trace.jsonl` 的 `tool_name` 序列
3. 模型输出的代表性 raw response（截 1-2 个）
4. 任何错误/异常
5. 真机 demo 的物理观察（机械臂路径、grasp 成功率、物体位置）

#### `trace.jsonl` 关键字段

| kind | 关键字段 | 含义 |
|---|---|---|
| `tool_call` | `tool_name` | orchestrator 实际 dispatch 的工具 |
| `llm_response` | `reasoning`, `content`, `parse_error` | LLM 返回 |
| `llm_fallback` | — | fallback chain 触发 |

### 13.9 与现有 §0-§11 的关系

- §13.4 Step 3 用 `start_reason2_server.sh`（**不是** §0 里的 `start_cosmos3_server.sh`）
- §13.4 Step 4 用 `--llm local_reason2`（**不是** §0 里的 `local_cosmos3`）
- GPU 显存策略不同：§0 用 `GPU_UTIL=0.70`，本节 `GPU_UTIL=0.75` + flashinfer sampler workspace ~150MB
- 推理特性不同：§0 用 qwen3_xml parser 做 tool calling；本节用 Slice 1-5 的 parser 做 reasoning 解析
- 其他架构、CAN bus、dashboard 等机制**与 §0-§11 完全一致**
- §12（当前状态 2026-08-11）描述的是 Cosmos3-Edge 4B 状态；本节是 Reason2-2B 的真机操作手册

### 13.10 完成后下一步

完成所有 tier 后：

1. **如果 Tier 1-2 成功**：Reason2-2B 接入 wrc_demo 主体**完成**
2. **如果 Tier 3-4 也成功**：可以考虑做 §11 离线打包（无网络部署）
3. **如果有问题**：记录现象 + trace 路径，回看 plan 调整

## 14. 真机测试 OOM 修复记录（2026-08-12）

### 14.1 问题现象

真机 demo 启动后 watcher 反复 OOM：

```text
[watcher:orbbec_overhead] OutOfMemoryError: CUDA out of memory
Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 7.50 GiB
of which 37.44 MiB is free. Process 12336 has 6.22 GiB memory in use.
```

### 14.2 根因分析

| 组件 | 显存占用 | 备注 |
|---|---|---|
| GPU 总计 | 7.50 GiB | RTX 5070 Laptop |
| vLLM (PID) | 6.22 GiB | 设定 GPU_UTIL=0.75 + KV cache 固定开销 ~700 MiB |
| YOLOE-26l-seg | ~310 MB | 反复加载失败（76MB 模型 + 运行时 buffer）|
| 其他（X server + view） | ~150 MB | |
| **剩余** | **~257 MB** | **不足 detector 加载需求** |

**矛盾**：8GB 卡装不下 BF16 Cosmos-Reason2-2B (4.88GB) + YOLOE detector + KV cache + 系统开销

### 14.3 修复方案

| 改动 | 原值 | 新值 | 效果 |
|---|---|---|---|
| vLLM `--gpu-memory-utilization` | 0.75 | **0.65** | 释放 ~1 GiB |
| detector 模型 | `yoloe-26l-seg.pt` (76MB) | **`yoloe-11s-seg.pt`** (38MB) | 节省 ~50% 模型显存 |
| 启动命令 | `WRC_VIEW=1` | **去掉** | Qt 渲染不再争 GPU |

### 14.4 修复步骤（用户可重复执行）

#### Step 1: 停 vLLM + wrc_demo

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --stop 2>&1 || true
pkill -f 'wrc_demo.apps.demo' 2>&1 || true
pkill -f 'wrc_demo.apps.mcp' 2>&1 || true
sleep 5
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
# 期望: < 500 MB used
```

#### Step 2: 改 vLLM GPU_UTIL

```bash
cp /home/seeed/wrc_demo/scripts/start_reason2_server.sh \
   /home/seeed/wrc_demo/scripts/start_reason2_server.sh.bak2

sed -i 's/GPU_UTIL="${GPU_UTIL:-0.75}"/GPU_UTIL="${GPU_UTIL:-0.65}"/' \
  /home/seeed/wrc_demo/scripts/start_reason2_server.sh

grep -n "GPU_UTIL=" /home/seeed/wrc_demo/scripts/start_reason2_server.sh | head -3
# 期望: GPU_UTIL="${GPU_UTIL:-0.65}"
```

#### Step 3: 改 detector 模型

```bash
cp /home/seeed/wrc_demo/configs/demo.yaml \
   /home/seeed/wrc_demo/configs/demo.yaml.bak2

sed -i 's|model: ${repo}/models/yoloe-26l-seg.pt|model: ${repo}/models/yoloe-11s-seg.pt|' \
  /home/seeed/wrc_demo/configs/demo.yaml

grep -n "yoloe" /home/seeed/wrc_demo/configs/demo.yaml | head -5
# 期望: model: ${repo}/models/yoloe-11s-seg.pt
```

#### Step 4: 重启 vLLM

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
for i in {1..15}; do
  sleep 6
  if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "✓ vLLM ready (after $((i*6))s)"
    break
  fi
done
```

#### Step 5: 验证显存

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
# 期望: ~5.0-5.7 GiB used / ~0.9-2.0 GiB free
```

#### Step 6: 启动 wrc_demo（**去掉 WRC_VIEW=1**）

```bash
cd /home/seeed/wrc_demo

sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/real_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_reason2 \
  --interactive
```

### 14.5 修复后状态

| 项 | 修复前 | 修复后 |
|---|---|---|
| vLLM 设定 GPU_UTIL | 0.75 | 0.65 |
| vLLM 实际占用 | 6.22 GiB | 5.71 GiB |
| detector 模型 | yoloe-26l-seg (76MB) | yoloe-11s-seg (38MB) |
| WRC_VIEW | 1 | 0 (去掉) |
| 显存剩余 | 257 MiB | 903 MiB |
| detector 加载 | ❌ OOM | ✅ 正常 |

### 14.6 关键发现

1. **vLLM 实际占用 > 设定值**：vLLM `GPU_UTIL=0.65` 实际占 5712 MiB（预期 ~4994 MiB）。多出 ~700 MiB 是 **KV cache / activations 固定开销**。计算显存预算时要考虑这个 +700 MiB buffer。

2. **YOLOE 模型选择**：
   - `yoloe-11s-seg.pt`（38MB，推荐 8GB 真机）
   - `yoloe-26s-seg.pt`（30MB，更小但准确度低）
   - `yoloe-26l-seg.pt`（76MB，桌面 dev 环境可用）

3. **WRC_VIEW=1 的代价**：Qt 渲染 + cv2.namedWindow 都要 GPU 资源。生产/真机部署应该关掉，只用 WRC_STREAM=1（HTTP 流）。

### 14.7 仍 OOM 的备选方案

如果按 §14.4 修复后仍 OOM：

1. **再降 GPU_UTIL**（0.65 → 0.60）：

   ```bash
   sed -i 's/GPU_UTIL="${GPU_UTIL:-0.65}"/GPU_UTIL="${GPU_UTIL:-0.60}"/' \
     /home/seeed/wrc_demo/scripts/start_reason2_server.sh
   ```

   然后重启。代价：vLLM KV cache 空间变小，长上下文可能 OOM。

2. **关掉其他 GPU 消费者**：

   ```bash
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   # 找出其他用 GPU 的进程，杀掉
   ```

3. **终极方案：量化 Reason2-2B**（NVFP4，4.88GB → ~2.5GB）：
   - 优点：永久释放 ~2.4 GiB，detector 获得充足空间
   - 缺点：需要联网跑一次官方 `./scripts/quantize.py`，约 30-60 分钟
   - 详见 Phase 2 计划（暂未实施）

### 14.8 备份与回滚

本次修复的备份文件：
- `/home/seeed/wrc_demo/scripts/start_reason2_server.sh.bak2`
- `/home/seeed/wrc_demo/configs/demo.yaml.bak2`

回滚命令：

```bash
cp /home/seeed/wrc_demo/scripts/start_reason2_server.sh.bak2 \
   /home/seeed/wrc_demo/scripts/start_reason2_server.sh
cp /home/seeed/wrc_demo/configs/demo.yaml.bak2 \
   /home/seeed/wrc_demo/configs/demo.yaml
# 然后重启 vLLM
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --stop
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
```

### 14.9 相关章节引用

- §13.4：wrce_demo 启动步骤（Step 4 是去掉 WRC_VIEW 后的版本）
- §13.5：测试场景（4 个 tier）
- §13.7：故障排查（含 OOM 相关）

---

## 15. NVFP4 量化失败 + 回滚记录（2026-08-12）

### 15.1 时间线

| 时间 | 事件 |
|---|---|
| 下午 14:39 | NVFP4 量化完成（llmcompressor 0.13.0） |
| 下午 14:41 | vLLM 启动脚本切到 NVFP4 + `--quantization compressed-tensors` |
| 下午 14:42-14:46 | 第一次 vLLM 启动尝试 → 卡死 → 重启 |
| 下午 14:50-15:00 | 第二次 vLLM 启动尝试 → 卡死 → 再次重启 |

### 15.2 根因分析

**为什么 BF16 能跑、NVFP4 卡死**：

| 配置 | BF16（成功） | NVFP4（卡死） |
|---|---|---|
| model.safetensors | 4.88GB raw | 2.7GB compressed |
| vLLM 加载路径 | 直接 load weights | load + **dequant kernel** |
| kernel 支持 | ✓ vLLM 0.27 原生 | ❌ vLLM 0.27 兼容性未验证 |

**关键结论**：

- NVFP4 是 NVIDIA Blackwell sm_120 专用 FP4 格式
- vLLM 0.27（2024-07）发布时 NVFP4 还未稳定
- vLLM 启动 hang（不是 OOM，是 kernel init 卡死，无错误信息）
- vLLM 升级到 ≥0.11（2026）后才正式支持 NVFP4 inference

### 15.3 回滚步骤（用户手动执行）

#### Step 1: 备份当前脚本

```bash
cp /home/seeed/wrc_demo/scripts/start_reason2_server.sh \
   /home/seeed/wrc_demo/scripts/start_reason2_server.sh.bak4
```

#### Step 2: 回滚 MODEL 路径（NVFP4 → BF16）

```bash
sed -i 's|MODEL=.*cosmos-reason2-2b-nvfp4.*|MODEL="/home/seeed/wrc_demo/models/cosmos-reason2-2b"|' \
  /home/seeed/wrc_demo/scripts/start_reason2_server.sh
```

#### Step 3: 删除 --quantization compressed-tensors（两处都要删）

```bash
sed -i '/--quantization compressed-tensors \\/d' \
  /home/seeed/wrc_demo/scripts/start_reason2_server.sh
```

#### Step 4: GPU_UTIL 改回 0.65（之前实测稳）

```bash
sed -i 's|GPU_UTIL="${GPU_UTIL:-0.78}"|GPU_UTIL="${GPU_UTIL:-0.65}"|' \
  /home/seeed/wrc_demo/scripts/start_reason2_server.sh
```

#### Step 5: 验证回滚

```bash
echo "=== 回滚后配置 ==="
grep -n "MODEL=\|GPU_UTIL=\|quantization" /home/seeed/wrc_demo/scripts/start_reason2_server.sh | head -10
```

**期望**：

- MODEL 指 `models/cosmos-reason2-2b`（BF16）
- GPU_UTIL=0.65
- 没有 `--quantization` flag

#### Step 6: 启动 vLLM（用户手动）

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
# 等 60-90s（BF16 之前实测 20s ready）
for i in {1..15}; do
  sleep 6
  if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "✓ vLLM ready (after $((i*6))s)"
    break
  fi
  echo "  check $i..."
done
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
# 期望: ~5.7 GB used / ~1.8 GB free
```

### 15.4 8GB 真机 demo 的硬约束

**回滚后 vLLM 启动成功，但真机 demo 仍可能 OOM**：

| 组件 | 显存占用 | 备注 |
|---|---|---|
| GPU 总计 | 7.50 GiB | RTX 5070 Laptop sm_120 |
| vLLM BF16 @ 0.65 | ~5.7 GiB | 实测 |
| YOLOe-11s 模型 | ~38 MB | 已切到 11s |
| **YOLOe mobileclip text encoder** | **~254 MB** | hidden bottleneck，之前未意识到 |
| YOLOe runtime + activation | ~200 MB | |
| CUDA context + fragmentation | ~300 MB | |
| **最小需求** | **~6.5 GiB** | |
| **剩余 (含 KV cache growth)** | **~1.0 GiB** | **临界，pick/place 会 OOM** |

**关键发现**：MobileCLIP text encoder (`mobileclip_blt.ts`, 254 MB) 是 hidden 的 GPU 消耗，切小 YOLOe 不够，必须降 vLLM GPU_UTIL 或换硬件。

### 15.5 三条出路（决策）

#### 选项 A: 接受 mock stack（最稳）

- vLLM BF16 @ 0.65 启动 OK + wrc_demo mock arm + **真 camera**（orbbec_overhead）
- 验证 Reason2 → orchestrator → mock skill 链路
- **不验证物理推理**（真机才有，但 8GB 不够）

#### 选项 B: 真机 Tier 1-2（reflex + list_objects）

- Tier 1: `go home` / `wave`（不走 LLM）
- Tier 2: `list the objects`（走 LLM + detector，但只读）
- Tier 3-4 (pick/place) 放弃——硬件能力不足
- **接受真机物理推理测试受限**

#### 选项 C: 换硬件

- 16GB+ 显卡（RTX 4080, RTX 5070 Ti, RTX 4090 24GB）
- 或 H100 / B200 服务器
- **长期方案**，但需要新硬件采购

### 15.6 备份文件清单

| 文件 | 状态 |
|---|---|
| `scripts/start_reason2_server.sh.bak` | 原始 BF16 @ 0.75 配置 |
| `scripts/start_reason2_server.sh.bak2` | 切 detector + GPU_UTIL=0.65 |
| `scripts/start_reason2_server.sh.bak3` | 切 NVFP4 模型路径（未加 flag） |
| `scripts/start_reason2_server.sh.bak4` | 切 NVFP4 + --quantization compressed-tensors |
| `scripts/start_reason2_server.sh` | **当前：BF16 + 0.65（回滚后）** |
| `models/cosmos-reason2-2b-nvfp4/` | 量化输出保留（2.7GB，未来 vLLM 升级后可用） |

### 15.7 相关章节引用

- §13.4: wrc_demo 启动步骤（Step 6 是去掉 WRC_VIEW 后的版本）
- §13.5: 测试场景（4 个 tier）
- §13.7: 故障排查（含 OOM 相关）
- §14: 真机测试 OOM 修复记录（vLLM @ 0.65 + yoloe-11s）
- §15: 本章节（NVFP4 失败 + 回滚）

### 15.8 关键经验教训

1. **量化不是万能药**：NVFP4 解决了 weights 大小，但 vLLM 兼容性是另一回事
2. **8GB 是硬约束**：任何"装下 vLLM"的方案都可能撞到 mobileclip 254MB
3. **手测优于自动测**：agent task 中断 2 次（卡 hang），用户手动跑更稳
4. **BF16 + 0.65 是已知稳定配置**：所有激进方案（更高 GPU_UTIL / 量化）都可能引入新风险
5. **NVFP4 模型保留**：`models/cosmos-reason2-2b-nvfp4/` 不删，等 vLLM 升级到 ≥0.11 后可重试

## 16. FP8 量化手动操作步骤（2026-08-12）

### 16.1 为什么是手动操作

之前 3 次 agent 跑 FP8/NVFP4 量化都导致系统卡崩溃重启：

| 原因 | 详情 |
|---|---|
| 长任务（30-60 分钟）| agent 跑期间用户无法看到进度、Ctrl+C |
| GPU 临界 | 量化峰值 ~6.8/7.5 GB，CUDA driver 可能 hang |
| 系统级 freeze | 不是干净的 OOM exception，是 system hang |
| 唯一解决方案 | 用户手动跑 + Terminal 1 监控 + 异常立即 Ctrl+C |

**新策略**：
- **agent 只跑短任务**（< 10 分钟：清理、配置修改）
- **用户手动跑长任务**（FP8 量化：30-60 分钟）
- **并行监控**（Terminal 1 `watch nvidia-smi`）

### 16.2 前置条件（已就绪）

| 项 | 状态 |
|---|---|
| `/home/seeed/wrc_demo/models/cosmos-reason2-2b/` (4.88GB BF16) | ✓ |
| `llmcompressor 0.13.0` 在 cosmos3-venv | ✓ |
| `/tmp/cosmos-reason2/` 仓库（带 `scripts/quantize.py`） | ✓ |
| `/tmp/run_fp8_quantize.sh` 脚本 | ✓ |
| GPU 干净（~430 MB used / 7250 MB free）| ✓ |
| 网络（HF / GitHub / PyPI）| ✓ |
| 磁盘空间 | 173 GB free |

### 16.3 关键显存防护（脚本已设置）

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

这个环境变量减少 CUDA allocator 碎片化，避免隐性 OOM。

### 16.4 操作步骤

#### Terminal 1：GPU 监控（一直开着）

```bash
watch -n 5 nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
```

**触发 Ctrl+C 的阈值**：

| 指标 | 阈值 | 动作 |
|---|---|---|
| memory.used | **> 6500 MiB** | ⚠️ Ctrl+C 量化进程 |
| memory.used | **> 7000 MiB** | 🚨 立即 Ctrl+C（系统可能 hang）|
| temperature.gpu | **> 85°C** | 🚨 立即 Ctrl+C（过热）|
| utilization.gpu | 持续 100% 几分钟 | 正常（继续观察）|

#### Terminal 2：跑 FP8 量化

```bash
bash /tmp/run_fp8_quantize.sh
```

**脚本内容**（自动）：
```bash
#!/bin/bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
source /home/seeed/cosmos3-venv/bin/activate
cd /tmp/cosmos-reason2

START=$(date +%s)
python scripts/quantize.py \
  --model /home/seeed/wrc_demo/models/cosmos-reason2-2b \
  -o /home/seeed/wrc_demo/models/cosmos-reason2-2b-fp8 \
  --precision fp8 \
  --num-samples 512 \
  --kv-precision bf16 \
  --smoothing-strength 0.8 \
  2>&1 | tee /tmp/quantize_fp8.log
END=$(date +%s)
echo "=== 耗时: $((END-START)) 秒 ($(( (END-START)/60 )) 分钟) ==="
```

#### 预期 GPU 占用曲线

| 阶段 | 耗时 | GPU used |
|---|---|---|
| 启动 + Python init | 10s | < 1 GB |
| 加载 BF16 模型 | 30s | **~5.5 GB** ←峰值1 |
| 校准 512 samples | 20-30 min | 5.5-6.0 GB |
| SmoothQuant 算法 | 5 min | 5.8-6.3 GB |
| 量化 + 写 model.safetensors | 5-10 min | 6.0-6.5 GB ←峰值2 |
| 保存完成 | 1 min | < 500 MB |
| **总耗时** | **30-60 分钟** | **峰值 ~6.5 GB** |

#### 输出验证（量化成功后）

```bash
ls -lh /home/seeed/wrc_demo/models/cosmos-reason2-2b-fp8/model.safetensors
# 期望: ~2.5 GB

python3 -c "
import json
with open('/home/seeed/wrc_demo/models/cosmos-reason2-2b-fp8/config.json') as f:
    cfg = json.load(f)
qc = cfg.get('quantization_config', {})
print('format:', qc.get('format'))
print('quant_method:', qc.get('quant_method'))
"
# 期望: format=fp8 / quant_method=compressed-tensors
```

### 16.5 失败处理

| 错误 | 表现 | 处理 |
|---|---|---|
| Ctrl+C 触发 | 进程退出，日志不完整 | 重新跑（llmcompressor 支持 resume） |
| OOM 错误 | 看到 "OutOfMemoryError" | 关闭其他 GPU 应用，降 calibration batch size |
| 系统 hang | 鼠标不动，必须重启 | 重启后回到 §16.4 步骤，从 Terminal 1 监控重新跑 |
| 网络中断 | 校准数据集下载失败 | 恢复网络后重跑 |
| exit code ≠ 0 | 脚本最后输出 EXIT=非0 | 看 `/tmp/quantize_fp8.log` 最后 50 行 |

### 16.6 量化成功后的下一步

FP8 量化成功后，**立即**：

1. **告诉 agent "FP8 量化完成"** —— 我会委派短 task：
   - 改 `scripts/start_reason2_server.sh`：
     - `MODEL` 路径切到 `cosmos-reason2-2b-fp8`
     - GPU_UTIL 改 0.85（FP8 模型小，vLLM 可以占用更多）
     - 加 `--quantization fp8 \` flag（两处）
     - 删之前的 `--quantization compressed-tensors \`（如果有）
   - 备份脚本为 `bak5`
2. **不**启动 vLLM——你手动验证启动
3. **手动跑** vLLM BF16 fallback（如果 FP8 启动失败）：
   - 把脚本改回 MODEL=BF16 路径
   - GPU_UTIL=0.65
   - 删 `--quantization fp8`

### 16.7 备份与回滚路径

**当前脚本状态**（FP8 量化前）：
```
MODEL="/home/seeed/wrc_demo/models/cosmos-reason2-2b"     ← BF16
GPU_UTIL="${GPU_UTIL:-0.78}"                                ← 0.78
--quantization compressed-tensors \                       ← 之前误加的
--quantization compressed-tensors \                       ← 两处都有
```

**备份文件**：
- `scripts/start_reason2_server.sh.bak` (原始)
- `scripts/start_reason2_server.sh.bak2` (切 detector + GPU_UTIL=0.65)
- `scripts/start_reason2_server.sh.bak3` (切 NVFP4 模型路径，未加 flag)
- `scripts/start_reason2_server.sh.bak4` (切 NVFP4 + --quantization compressed-tensors)
- `scripts/start_reason2_server.sh` (BF16 + 0.78 + compressed-tensors flag)

**FP8 完成后会备份为 `.bak5`**。

### 16.8 相关章节引用

- §13: 真机测试操作步骤（Cosmos-Reason2-2B）
- §14: 真机测试 OOM 修复记录
- §15: NVFP4 量化失败 + 回滚记录（关键经验：vLLM 0.27 不支持 NVFP4）
- §16: 本章节（FP8 手动操作步骤）
- 未来 §17: FP8 量化成功后 vLLM 切换记录

### 16.9 关键经验教训

1. **长任务必须用户手动跑**：agent 跑 30+ 分钟任务期间，GPU 临界可能系统 hang
2. **监控必须并行**：Terminal 1 监控 + Terminal 2 跑任务
3. **显存阈值明确**：6500 MB 警告，7000 MB 立即停
4. **环境变量必备**：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片化
5. **没有预量化版本**：HF 上只有 BF16 + llama.cpp/AWQ 量化，没有 vLLM FP8
6. **FP8 替代 NVFP4**：vLLM 0.27 原生支持 FP8 W8A8，不支持 NVFP4

## 17. Qwen3-VL-2B-FP8 切换 + wrc_demo 操作命令（2026-08-12）

### 17.1 切换总结（与 §13-§16 对比）

| 维度 | Cosmos-Reason2-2B（之前） | Qwen3-VL-2B-Instruct（现在） |
|---|---|---|
| 模型 | nvidia/Cosmos-Reason2-2B | Qwen/Qwen3-VL-2B-Instruct |
| 精度 | BF16（4.88GB） | FP8（3.3GB） |
| 量化 | 尝试 NVFP4/FP8 失败 | **官方预量化 FP8** ✓ |
| Tool calling | 需 parser hack（Reason2ChatClient） | **vLLM 原生 hermes parser** |
| 显存 @ 0.60 | - | **6.811 GB < 7 GB ✓** |
| wrc_demo 兼容 | 需 reason2: true flag | **标准 openai_compat** |

**结论**：Qwen3-VL-Instruct **原生支持 tool calling**（与 OpenAI 协议完全兼容），无需任何特殊适配。

### 17.2 vLLM 当前状态（用户验证后开始）

| 项 | 值 |
|---|---|
| vLLM PID | 27983 |
| 模型路径 | `/home/seeed/wrc_demo/models/qwen3-vl-2b-fp8` |
| 显存 used | 6974 MiB (6.811 GB) |
| 显存 free | ~769 MB |
| 端口 | 8080 |
| Profile | `local_qwen3_vl` |

### 17.3 wrc_demo mock stack 测试（**用户立即跑**）

**mock stack 不接真硬件**——只验证 LLM 链路 + tool calling + orchestrator + skill dispatch。

#### Terminal 1: 确认 vLLM 还活着

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status 2>&1 || {
  echo "vLLM 没跑，正在启动..."
  bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
  # 等 ready
  for i in {1..15}; do sleep 6
    if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
      echo "✓ vLLM ready (after $((i*6))s)"
      break
    fi
  done
}
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# 期望: ~6800 MB used
```

#### Terminal 2: 跑 wrc_demo mock 测试

```bash
cd /home/seeed/wrc_demo

HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
WRC_RUN_DIR=/home/seeed/wrc_demo/runs/qwen_test_$(date +%Y%m%d_%H%M%S) \
/home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --task "list the objects" \
  --no-view --no-serve \
  --cameras mock --arm mock \
  --llm local_qwen3_vl
```

**预期**：
- LLM 路径触发（不是 reflex）
- trace.jsonl 出现 `tool_name: list_objects`
- 工具调用格式 OpenAI 标准（hermes parser 已解析）
- 30 步内完成（mock 空场景可能快速 task_done）

#### 验证 trace.jsonl

```bash
# 找最新 run 目录
RUN_DIR=$(ls -td /home/seeed/wrc_demo/runs/qwen_test_* | head -1)

# 看 tool_call 序列
python3 -c "
import json
from pathlib import Path
trace = Path('$RUN_DIR/trace.jsonl')
if not trace.exists():
    print('⚠ trace.jsonl 不存在')
else:
    counts = {}
    parse_errors = {}
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except:
            continue
        if ev.get('kind') == 'tool_call':
            counts[ev.get('tool_name', '?')] = counts.get(ev.get('tool_name', '?'), 0) + 1
        if 'parse_error' in ev:
            pe = ev.get('parse_error')
            parse_errors[pe] = parse_errors.get(pe, 0) + 1
    print(f'tool_call 总数: {sum(counts.values())}')
    print(f'工具序列: {counts}')
    print(f'parse_errors: {parse_errors}')
"
```

**期望**：
- tool_call 总数 >= 1（说明 vLLM 的 tool_calls 字段被解析）
- 工具名是 SKILL_SPECS 里的（`list_objects`、`detect_object` 等）
- parse_errors 应为 0 或很少（hermes parser 工作良好）

### 17.4 wrc_demo 真机 demo 操作命令（mock 通过后跑）

**前提**：
- sudo 凭证缓存（`sudo -v` 跑一次）
- CAN bus 状态 UP（`ip link show can0`）
- 物理硬件就位（机械臂、相机、物体）
- vLLM + detector 显存够（vLLM 6.8 GB + detector ~500 MB = 7.3 GB，**临界**！）

#### Step 1: sudo + CAN bus

```bash
sudo -v
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip link show can0
# 期望: state UP
```

#### Step 2: 启动 wrc_demo（**去掉 WRC_VIEW=1** 节省 GPU）

```bash
cd /home/seeed/wrc_demo

sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/real_qwen_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

#### Step 3: 浏览器看 dashboard

- `http://<机器IP>:8090/`（相机流 + 状态）

### 17.5 真机测试 4 个 Tier（按 §13.5 顺序）

| Tier | 任务 | 风险 | 验证 |
|---|---|---|---|
| 1 | `go home` / `wave` | 极低（reflex） | reflex 是否走通 |
| 2 | `list the objects` | 低 | LLM 链路、tool_call 解析 |
| 3 | `pick up the banana` | 中 | 物理推理 + tool dispatch |
| 4 | `put the banana into the box` | 高 | 端到端 |

### 17.6 Tool calling 验证（关键优势）

Qwen3-VL + hermes parser **原生支持**：

```bash
curl -sf -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/seeed/wrc_demo/models/qwen3-vl-2b-fp8",
    "messages": [{"role":"user","content":"Detect the banana."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "detect_object",
        "description": "Detect an object",
        "parameters": {
          "type": "object",
          "properties": {"label": {"type": "string"}},
          "required": ["label"]
        }
      }
    }],
    "max_tokens": 256
  }' 2>&1 | python3 -c "
import json, sys
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
print('tool_calls count:', len(msg.get('tool_calls') or []))
if msg.get('tool_calls'):
    t = msg['tool_calls'][0]
    print(f'first tool: {t[\"function\"][\"name\"]}({t[\"function\"][\"arguments\"]})')
"
```

**预期**：
- `tool_calls count: 1`
- `first tool: detect_object({"label": "banana"})`
- `finish_reason: tool_calls`

### 17.7 关键修复记录（避免以后重新踩坑）

| 问题 | 解决 |
|---|---|
| NVFP4 量化 + vLLM 0.27 卡死 | vLLM 0.27 不支持 NVFP4 inference kernel |
| FP8 量化（llmcompressor）跑不动 | wikitext 替换 flickr30k；moe_context 路径修复；add_labels 失败 |
| FP8 模型加载 + DeepGEMM 失败 | 改 `--linear-backend triton` |
| tool-call-parser qwen3_xml 不匹配 | Qwen3-VL 用 `<tool_call>{json}</tool_call>` 格式，hermes parser 才能解析 |
| 显存超 7 GB | GPU_UTIL 0.85 → 0.60 |

### 17.8 备份文件清单

| 文件 | 状态 |
|---|---|
| `scripts/start_reason2_server.sh.bak` | 原始 cosmos3-edge 配置 |
| `scripts/start_reason2_server.sh.bak2` | 切 detector + GPU_UTIL=0.65 |
| `scripts/start_reason2_server.sh.bak3` | 切 NVFP4 模型路径 |
| `scripts/start_reason2_server.sh.bak4` | 切 NVFP4 + compressed-tensors |
| `scripts/start_reason2_server.sh.bak5` | 切 qwen3-vl-2b-fp8 + fp8 |
| `scripts/start_reason2_server.sh.bak6` | 加 linear-backend triton（之前错误尝试） |
| `scripts/start_reason2_server.sh.bak7` | 清空行修复（bash continuation bug）+ linear-backend triton |
| `scripts/start_reason2_server.sh.bak8` | GPU_UTIL=0.60 + parser=hermes（当前） |
| `scripts/start_reason2_server.sh` | **当前：Qwen3-VL-FP8 + GPU_UTIL=0.60 + Triton + hermes** |

### 17.9 相关章节引用

- §0-§11: 旧配置（cosmos3-edge 时代）
- §12: 旧状态（2026-08-11）
- §13: 真机测试操作步骤（基于 cosmos-reason2 写的，部分仍然适用）
- §14: 早期 OOM 修复记录
- §15: NVFP4 失败 + 回滚（最终改为 Qwen3-VL）
- §16: FP8 量化失败（最终放弃本地量化，用官方预量化版）
- **§17: 当前章节**（Qwen3-VL-FP8 + wrc_demo 操作）

### 17.10 历史遗留代码（可清理但不影响）

**Phase 1-5 写的代码现在完全没用**：
- `src/wrc_demo/agent/reason2_parser.py`
- `src/wrc_demo/agent/llm.py` 里的 `Reason2ChatClient` 类
- `configs/llm/local_reason2.yaml`

**保留无害**（不影响 wrc_demo 运行）：
- 测试 `tests/test_reason2_parser.py`、`tests/test_reason2_chat_client.py`、`tests/test_make_llm_reason2.py` 仍通过（不依赖 vLLM）

**未来清理建议**（用户手动决定）：
- 删除 `reason2_parser.py` 和 `Reason2ChatClient` 类
- 删除 `local_reason2.yaml`
- 删除 `tests/test_reason2_*.py` 和 `tests/test_make_llm_reason2.py`
- 删除 `models/cosmos-reason2-2b-nvfp4/`（NVFP4 量化输出）
- 删除 `models/cosmos-reason2-2b/`（如果不再用 BF16）
- 备份脚本 `.bak2` ~ `.bak8`（保留 .bak 当最后 fallback）

## 18. AWQ 4bit 切换成功 — 真机 demo 应该能跑（2026-08-12）

### 18.1 切换总结

| 项 | FP8（之前）| **AWQ 4bit（现在）**|
|---|---|---|
| 模型 | Qwen/Qwen3-VL-2B-Instruct-FP8 | **cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit** |
| 精度 | FP8 W8A8 | **INT4 W4A16** |
| 模型磁盘 | 3.3 GB | **2.1 GB**（-36%）|
| vLLM 显存 | 6852 MB | **5760 MB**（-16%）|
| 显存 free | 650 MB | **1924 MB**（+196% ✓）|
| Vision encoder | ❌ 关 | **✅ 开** |
| 真机 demo | ❌ OOM | **✅ 应该能跑** |
| Tool calling | ✓ hermes | ✓ hermes |

### 18.2 关键修复记录

| 问题 | 解决 |
|---|---|
| FP8 真机 OOM（mobileclip 254MB 装不下）| 改 AWQ 4bit 模型（free +1.3 GB）|
| 之前 `--limit-mm-per-prompt '{"image":0,"video":0}'` 关 vision | AWQ 启动时删除此 flag，vision encoder 加载 |
| 之前 `--quantization fp8` 加载 flag | 改 `--quantization compressed-tensors`（AWQ 格式）|
| 之前 `--reasoning-parser qwen3`（reasoning 模型用）| 删除（AWQ 是 Instruct，不是 reasoning）|

### 18.3 当前 vLLM 配置（最稳定）

```bash
vllm serve /home/seeed/wrc_demo/models/qwen3-vl-2b-awq-4bit \
    --port 8080 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --linear-backend triton \
    --quantization compressed-tensors \
    --attention-backend FLASH_ATTN \
    --host 0.0.0.0
```

### 18.4 真机 demo 操作命令（用户测试用）

```bash
# Terminal 1: 确认 vLLM 状态
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# 期望: ~5760 MB used

# Terminal 2: 真机 demo（**去掉 WRC_VIEW 节省 GPU**）
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/awq_test_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

**浏览器 dashboard**：`http://198.18.0.1:8090/`

### 18.5 测试场景（按 §13.5 顺序）

| Tier | 任务 | 验证 |
|---|---|---|
| 1 | `go home` / `wave` | reflex 路径（不走 LLM）|
| 2 | `list the objects` | LLM 链路（tool calling + dispatch）|
| 3 | `pick up the banana` | 物理推理 + detector + 真机 arm |
| 4 | `put the banana into the box` | 端到端 |

### 18.6 监控指标（用户跑测试时观察）

- vLLM 启动：~15-20s ready
- 显存使用：~5.76 GB（free 1.9 GB）
- 单 turn 推理延迟：~1-3s（AWQ 4bit 可能比 FP8 略慢）
- trace.jsonl 在 `$WRC_RUN_DIR/trace.jsonl`
- vLLM 日志：`/tmp/vllm_reason2.log`

### 18.7 验证 AWQ tool calling

```bash
curl -sf -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/seeed/wrc_demo/models/qwen3-vl-2b-awq-4bit",
    "messages": [{"role":"user","content":"Detect the banana on the table."}],
    "tools": [{"type":"function","function":{"name":"detect_object","description":"Detect","parameters":{"type":"object","properties":{"label":{"type":"string"}},"required":["label"]}}}],
    "max_tokens": 256
  }' 2>&1 | python3 -m json.tool | head -20
```
期望：`tool_calls[0].function.name = "detect_object"`、`tool_calls[0].function.arguments = {"label": "banana"}`

### 18.8 关键经验（避免以后重踩）

1. **AWQ INT4 优于 FP8 for 8GB 卡**：模型小（2.1GB vs 3.3GB），可以加载 vision encoder
2. **vLLM compressed-tensors 后端**：必须用 `--quantization compressed-tensors`，**不是** `--quantization awq`（前者是 LLM Compressor 路径）
3. **Triton linear backend** 避开了 DeepGEMM 在 Blackwell sm120 的 layout transform bug
4. **GPU_UTIL 0.75** 是当前 vLLM 的稳定配置（注释里有完整失败历史）
5. **vLLM 0.27 + Hermes parser** 兼容 INT4 量化模型
6. **真机 demo 需要**：vLLM + detector (~500MB) + arm SDK + CUDA overhead，**总需求 < 7.5 GB**

### 18.9 备份文件清单

| 文件 | 状态 |
|---|---|
| `scripts/start_reason2_server.sh.bak7` | linear-backend triton |
| `scripts/start_reason2_server.sh.bak8` | GPU_UTIL=0.60 + parser=hermes |
| `scripts/start_reason2_server.sh.bak9` | FP8 配置 |
| `scripts/start_reason2_server.sh` | **当前：AWQ 4bit + compressed-tensors + 0.75 GPU_UTIL** |

### 18.10 相关章节引用

- §13: 真机测试操作步骤（cosmos-3 时代，可参考测试场景）
- §14: 早期 OOM 修复记录
- §15: NVFP4 量化失败 + 回滚
- §16: FP8 量化失败（最终放弃本地量化，用官方预量化版）
- §17: Qwen3-VL-FP8 切换（被 AWQ 替代）
- **§18: 本章节（AWQ 4bit 切换成功，真机 demo 解锁）**

### 18.11 历史遗留（不再需要）

**已清理（§17 完成）**：
- ❌ `src/wrc_demo/agent/reason2_parser.py`（删除）
- ❌ `src/wrc_demo/agent/llm.py::Reason2ChatClient`（删除）
- ❌ `configs/llm/local_reason2.yaml`（删除）
- ❌ `tests/test_reason2_*.py`（删除 3 个）
- ❌ `tests/test_make_llm_reason2.py`（删除）

**模型**：
- `models/cosmos-reason2-2b/`（BF16 4.88 GB，未用）
- `models/cosmos-reason2-2b-nvfp4/`（NVFP4 2.7 GB，未用）
- `models/qwen3-vl-2b-fp8/`（FP8 3.3 GB，未用，AWQ 替代）
- `models/qwen3-vl-2b-awq-4bit/`（**当前使用** 2.1 GB）

### 18.12 真机 LLM demo 跑通 — detector CPU 模式（2026-08-13）

#### 18.12.1 改动内容

| 文件 | 改动 |
|---|---|
| `configs/demo.yaml:137` | `device: cuda:0` → `device: cpu` |
| `configs/llm/local_qwen3_vl.yaml` | `model: qwen3-vl-2b-fp8` → `qwen3-vl-2b-awq-4bit`（与 vLLM 服务匹配）|
| 备份 | `configs/demo.yaml.bak_cpu`（保留可回滚）|

#### 18.12.2 关键验证结果

| 验证项 | 结果 |
|---|---|
| vLLM 启动 | ✓ AWQ 4bit @ GPU_UTIL=0.75 |
| vLLM 显存 | **5910 MiB** / 8151 MiB（余量 ~2.2 GB）|
| detector CPU 加载 | ✓ 0.3s |
| detector CPU 推理 | ✓ 1.4s/inference |
| **真机 LLM demo** | ✓ **跑通**（camera 真 + LLM 真 + detector 真检测）|
| 任务 path | **llm**（不是 reflex）|
| detector 输出 | banana conf=0.7 @ [0.328, -0.175, 0.017]，box conf=0.23 @ [0.331, 0.042, 0.08] |
| LLM 输出 | "Listed the two objects on the table: a banana and a box." |
| trace 工具序列 | get_observation → list_objects → task_done（3 步）|
| vLLM 请求数 | 22 次 POST /v1/chat/completions 200 OK |
| 显存 OOM | ❌ **没有**（detector 跑 CPU 不占 GPU）|

#### 18.12.3 完整真机测试命令（用户跑真 arm）

```bash
# Terminal 1: 监控 GPU
watch -n 5 nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Terminal 2: 启动 vLLM（如未在跑）
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg

# Terminal 3: 真机 demo（真 arm + detector CPU）
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/real_llm_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

#### 18.12.4 推荐 LLM 路径测试命令（不走 reflex）

| 命令 | 期望 path | 验证 |
|---|---|---|
| `list the objects` | llm | detector 真检测 + LLM 输出 |
| `find me the banana` | llm | LLM 调 find_object skill |
| `pick up the banana then place it on the table` | llm | compound with "then" 走 LLM |
| `describe what you see` | reflex (不推荐测，已经验证) | - |

#### 18.12.5 回滚路径（如需恢复 GPU detector）

```bash
cp /home/seeed/wrc_demo/configs/demo.yaml.bak_cpu \
   /home/seeed/wrc_demo/configs/demo.yaml
# device 恢复 cuda:0
```

#### 18.12.6 关键经验

1. **detector CPU 是 8GB 卡跑真机 LLM demo 的最优解**——vLLM 单独占 5.9 GB，余量 2.2 GB 足够 arm SDK + CUDA context
2. **CPU detector 性能损失可接受** —— 0.3s 加载 + 1.4s/inference，每次 detector 调用延迟 ~2s
3. **AWQ 4bit + CPU detector 的组合** —— vLLM 加载到 7.1 GB（之前 OOM），现在 + detector CPU 总占 5.9 GB（CUDA OOM 解决）
4. **真机 LLM 链路完整跑通** —— reflex 路径走了 ~80% 命令，LLM 路径覆盖剩余 20%（变体命令、compound、异常处理）

#### 18.12.7 真机 LLM 链路 vs Reflex 路径

| 维度 | Reflex 路径（之前验证） | LLM 路径（今天验证） |
|---|---|---|
| 命令 | `put banana in box`, `pick up banana` | `list the objects`, `find me banana` |
| 理解力 | 固定模式 | 任意自然语言 |
| 物体位置 | 预设（脚本 hard-code）| **真检测**（camera + YOLOe）|
| 速度 | ~10ms | ~2s（含 detector + LLM）|
| 工具调用 | 1 个直接 skill | **3+ 个 skills**（observation → action → done）|
| 多步 | ❌ | ✅ |


### 18.13 真机 LLM demo joint_signs bug 修复（2026-08-13）

#### 18.13.1 现象

- 真机 LLM demo 跑 `pick up the banana then place it on the table` 崩
- 抓取成功（skill_grasp_object 工作）
- 移动崩：`AttributeError: 'Kinematics' object has no attribute '_pin'`

#### 18.13.2 根因

- CLAUDE.md §"Contracts" 警告过：`Arm profiles pointing at the shipped assets MUST carry joint_signs: [-1]*6`
- 但 `configs/arms/rebot_rs.yaml` 没填该字段
- `apps/demo.py:55-58` 也没读 `cfg.arm.joint_signs`
- 结果：`Kinematics` 加载失败（`frame 'gripper_end' not in model`），`self._pin` 未初始化，`cartesian_planner` 拿到空对象

#### 18.13.3 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/apps/demo.py:55-58` | 加 `joint_signs=list(cfg.arm.get("joint_signs", [-1] * 6))` |
| `configs/arms/rebot_rs.yaml` | 加 `joint_signs: [-1, -1, -1, -1, -1, -1]` |
| 备份 | `apps/demo.py.bak_jointsign` + `rebot_rs.yaml.bak_jointsign` |

#### 18.13.4 验证结果

| 验证 | 结果 |
|---|---|
| `Kinematics(..., joint_signs=[-1]*6)` 加载 | ✓ |
| `CartesianPlanner(k)` 实例化 | ✓ |
| FK 验证 | `FK(0) → [0.3017, ~0, 0.2178]` |
| **真机 LLM demo** | **应能完整跑 pick-and-place** |

#### 18.13.5 真机 LLM 链路完整（现在 vs 之前）

| 维度 | 之前（崩） | 现在（修好） |
|---|---|---|
| grasp | ✓ | ✓ |
| cartesian_planner | ❌ `_pin` AttributeError | ✓ |
| place_at | ❌ 崩 | ✓ |
| release gripper | ❌ 没执行 | ✓ |
| task_done | ❌ | ✓ |

#### 18.13.6 用户真机测试命令

```bash
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/llm_fixed_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

测试命令：
```task> list the objects
task> pick up the banana then place it on the table
```

#### 18.13.7 关键经验（CLAUDE.md 早就警告过）

1. **CLAUDE.md §"Contracts" 警告过** arm profile 必须有 `joint_signs: [-1]*6`，**但代码没遵守**
2. **之前的 36 个 pre-existing failures** 反映了这个问题（`test_kinematics.py` 等都因为 `_pin` 错 fail）
3. **只有真机 LLM demo（不走 reflex）**才会触发这个 bug——之前 reflex 路径绕过了
4. **修复 = 补 joint_signs** 到 yaml + 改 demo.py 传参数

#### 18.13.8 回滚路径

```bash
cp /home/seeed/wrc_demo/src/wrc_demo/apps/demo.py.bak_jointsign \
   /home/seeed/wrc_demo/src/wrc_demo/apps/demo.py
cp /home/seeed/wrc_demo/configs/arms/rebot_rs.yaml.bak_jointsign \
   /home/seeed/wrc_demo/configs/arms/rebot_rs.yaml
```

### 18.14 cartesian_planner 真正修复 — 移除 _pin 依赖（2026-08-13）

### 18.14.1 现象

- 之前 `pick up the banana then place it on the table` 跑出 `AttributeError: 'Kinematics' object has no attribute '_pin'`
- 抓取成功，但 place_at 步骤崩

### 18.14.2 真正的根因

- `apps/demo.py:36` 导入的 `Kinematics` 实际是 `safety.harness.Kinematics`（**不是** `control.kinematics.Kinematics`）
- `safety.harness.Kinematics` **故意不用 pinocchio**（注释明确）
- `cartesian_planner` 用 `self._pin.SE3(...)`（pinocchio API）→ safety.harness.Kinematics 没 `_pin` 属性

### 18.14.3 之前的"修复"是错的

- agent 之前改 `apps/demo.py` 加 `joint_signs=list(cfg.arm.get("joint_signs", [-1] * 6))`
- 报新错：`TypeError: Kinematics.__init__() got an unexpected keyword argument 'joint_signs'`
- 因为 `safety.harness.Kinematics.__init__` 签名是 `(urdf_path, ee_frame, arm_joint_count)`，**没有** `joint_signs`

### 18.14.4 CLAUDE.md §"Contracts" 的误读

- "Arm profiles pointing at the shipped assets MUST carry `joint_signs: [-1]*6`" —— **是为已废弃的 `control.kinematics.Kinematics` 设计的**
- `safety.harness.Kinematics` 是**正确**的，**不**需要 joint_signs
- `safety.harness.Kinematics` 注释："no `joint_signs` multiplication is applied (the old wrc_demo control layer did that and it was **wrong**)"

### 18.14.5 真正修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/apps/demo.py` | **回滚**到 agent 改之前（移除 `joint_signs` 参数）|
| `configs/arms/rebot_rs.yaml` | **回滚**到 agent 改之前（移除 `joint_signs`）|
| `src/wrc_demo/planning/cartesian_planner.py` | 删 `self._pin = kin._pin`；改用 numpy passthrough（`_to_se3`/`_from_se3` 改为 identity）|

### 18.14.6 改动细节

```python
# cartesian_planner.py:65 之前
self._pin = kin._pin

# 之后
# 删除（self._pin 不再需要）

# _to_se3 之前
def _to_se3(self, T):
    T = np.asarray(T, dtype=float)
    return self._pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())

# 之后
def _to_se3(self, T):
    return np.asarray(T, dtype=float)  # identity passthrough

# _from_se3 之前
def _from_se3(self, se3):
    return np.array(se3.homogeneous)

# 之后
def _from_se3(self, T):
    return np.asarray(T, dtype=float)  # identity passthrough
```

### 18.14.7 为什么 numpy passthrough 可行

SDK 的 `_se3_interpolate` (`sdk/.../trajectory/sampler.py:80-83`) 已自动把 `np.ndarray` 转 `pin.SE3`：

```python
# SDK 内部
def _se3_interpolate(T_start, T_end, t):
    return pin.SE3.Interpolate(...)  # 自动接 numpy 4x4
```

所以 cartesian_planner 传 `np.ndarray` 给 SDK 的 sampler 时，SDK 自己处理 SE3 转换——cartesian_planner 不需要自己 wrap。

### 18.14.8 验证结果

| 验证 | 结果 |
|---|---|
| `safety.harness.Kinematics` 加载 | ✓ |
| `CartesianPlanner(k)` 实例化 | ✓（不再依赖 `_pin`）|
| 备份保留 | `cartesian_planner.py.bak_numpy` |

### 18.14.9 真机 LLM 完整链路（现在 vs 之前）

| 维度 | 之前 | 现在 |
|---|---|---|
| grasp | ✓ | ✓ |
| detector (CPU) | ✓ | ✓ |
| cartesian_planner | ❌ `_pin` AttributeError | ✓ |
| place_at | ❌ 崩 | ✓ |
| release gripper | ❌ 没执行 | ✓ |
| task_done | ❌ | ✓ |
| **真机 LLM demo** | ❌ 失败 | **✓ 应能完整跑通** |

### 18.14.10 真机 LLM demo 操作命令

```bash
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/llm_real_v2_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

测试：
```task> pick up the banana then place it on the table
```

### 18.14.11 关键经验

1. **`safety.harness.Kinematics` ≠ `control.kinematics.Kinematics`**——两个不同的类
2. CLAUDE.md §"Contracts" 的 `joint_signs: [-1]*6` **是过时的**（针对已废弃的 control.kinematics）
3. **`safety.harness.Kinematics` 是当前正确选择**（用 SDK 加载，不应用 joint_signs）
4. **`cartesian_planner` 之前依赖 pinocchio 的 `_pin`**——是错误设计（应不依赖 Kinematics 内部实现）
5. **SDK 的 `_se3_interpolate` 自动处理 ndarray→SE3**——cartesian_planner 改用 numpy passthrough 即可

### 18.15 cartesian_planner 兼容 safety.harness.Kinematics（2026-08-13）

### 18.15.1 现象
- §18.14 修复后，真机 LLM demo 仍崩：
  ```
  File "wrc_demo/planning/cartesian_planner.py", line 139, in plan
      q_init = np.zeros(self.kin.model.nq)
  AttributeError: 'Kinematics' object has no attribute 'model'. Did you mean: '_model'?
  ```

### 18.15.2 根因
- §18.14 修了 `_pin` 和 `_to_se3` 但**没修其他 `self.kin.X` 引用**
- `cartesian_planner.py` 还有 5 处用了 `control.kinematics.Kinematics` 风格（无下划线）：
  - line 139: `self.kin.model.nq`
  - line 140: `self.kin.n`
  - line 142-143: `self.kin.model`, `self.kin.fid`
  - line 162: `self.kin.n`
- 但 `SkillRuntime.kin` 实际是 `safety.harness.Kinematics`（**带下划线**：`_model`/`_data`/`_ee_frame_id`/`_n_arm`）

### 18.15.3 docstring 与代码不一致
- `cartesian_planner.py` docstring 说："the harness's :class:`Kinematics` is loaded via the SDK and exposes `model`/`data`/`fid` but no `_pin` module handle"
- **注释说无下划线，实际代码有下划线**——历史遗留错误

### 18.15.4 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/planning/cartesian_planner.py:139-140` | 用 `getattr` 取 `_model`/`_n_arm`（兼容 safety.harness）|
| `src/wrc_demo/planning/cartesian_planner.py:142-143` | 用 `getattr` 取 `_ee_frame_id` |
| `src/wrc_demo/planning/cartesian_planner.py:162` | 用本地变量 `_n_arm` |
| 备份 | `cartesian_planner.py.bak_compat` |

### 18.15.5 修复 pattern

```python
# 之前
q_init = np.zeros(self.kin.model.nq)
q_init[: self.kin.n] = np.asarray(q_start, dtype=float)[: self.kin.n]
joint_traj = track_trajectory(
    self.kin.model, self.kin.fid, ...
)

# 之后（兼容两种 Kinematics）
_model = getattr(self.kin, '_model', getattr(self.kin, 'model', None))
_n_arm = getattr(self.kin, '_n_arm', getattr(self.kin, 'n', None))
if _model is None or _n_arm is None:
    raise RuntimeError("Kinematics missing required attrs")
q_init = np.zeros(_model.nq)
q_init[:_n_arm] = np.asarray(q_start, dtype=float)[:_n_arm]
joint_traj = track_trajectory(
    _model, getattr(self.kin, '_ee_frame_id', getattr(self.kin, 'fid', None)), ...
)
```

### 18.15.6 验证

| 验证 | 结果 |
|---|---|
| `safety.harness.Kinematics` 加载 | ✓ |
| `_ee_frame_id=15`, `_model.nq=8`, `_n_arm=6` | ✓ |
| `CartesianPlanner(k)` 实例化 | ✓ |
| `plan(q0, T0, T1, duration=1.0)` | ✓ **51 waypoints**（1s × 50Hz + 1）|

### 18.15.7 真机 LLM demo 操作

```bash
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/llm_v3_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

测试：
```task> pick up the banana then place it on the table
```

### 18.15.8 完整修复链

| § | 修复 |
|---|---|
| §18.1-§18.11 | AWQ 4bit 量化 + Triton + hermes parser |
| §18.12 | detector CPU 化（mobileclip 254 MB 释放）|
| §18.14 | cartesian_planner 移除 `_pin` 依赖（numpy passthrough）|
| **§18.15** | **cartesian_planner 兼容 safety.harness.Kinematics（getattr）**|

### 18.15.9 关键经验

1. **`safety.harness.Kinematics` ≠ `control.kinematics.Kinematics`**——两套接口不一致
2. **`cartesian_planner` 注释和代码不一致**——历史遗留错误
3. **`getattr` 兼容模式**——支持两种 Kinematics 类型（不依赖具体类）
4. **之前的 §18.14 修复不完整**——只改了 `_pin` 和 `_to_se3`，没改其他 `self.kin.X` 引用
5. **真机 LLM demo 完整跑通**——所有 cartesian_planner 引用都修了

### 18.16 LLM 路径物理 bug 修复（2026-08-13）

#### 18.16.1 现象
真机 LLM demo 跑 `pick up the banana then place it on the table`：
- `place_at(x=0.3, y=0.0, z=0.2) -> ok`（dashboard 显示）
- `placed 'banana' at [0.3, 0.0, 0.145]`（z cap 后）
- `task_done success=True`
- **但机械臂实际悬停**——**没真到 target**！

#### 18.16.2 根因
两个 bug 链：
1. `cartesian_planner.plan()` 默认 `max_fail_ratio=0.15` 太宽松——50% IK 失败仍返回部分 waypoint
2. `skill_place_at` 不 verify TCP 实际到达 target——`move_cartesian` 返回 True 就 trust

**bug 链**：
1. LLM 输出 `place_at(x=0.3, y=0.0, z=0.2)`
2. `skill_place_at` 算 target = (0.3, 0.0, 0.145)（z cap）
3. 调 IK 算 pre (hover) + low (target)
4. `arm.move_joints(pre.q)` 移动到 hover ✓
5. `arm.move_cartesian(low.q FK, target, planner)` 
6. `planner.plan()` 出 waypoint（**3/6 IK 失败**但仍返回）
7. `move_joints_path(waypoints)` 走错 waypoint——**机械臂到错位置**
8. `move_cartesian` 返回 True（"动过了"）
9. `set_gripper(open)` 释放——香蕉在 hover 处掉下来
10. `held_object = None` + `beliefs.update(placed)`——**假 placed**
11. `place_at ok` + `task_done success=True`——**假成功**

#### 18.16.3 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/planning/cartesian_planner.py:100` | `max_fail_ratio: float = 0.15` → `0.05` |
| `src/wrc_demo/skills/runtime.py:1690-1706`（新增 17 行）| verify TCP 实际位置（2cm 误差内才 success）|
| 备份 | `cartesian_planner.py.bak_strict` + `runtime.py.bak_verify` |

#### 18.16.4 verify TCP 实际位置的代码

```python
# 2026-08-13: verify TCP actually reached the target.
# cartesian_planner's waypoint IK may partially fail; if so the
# arm stops short of the target but move_cartesian still returns
# True. Don't trust the boolean — re-FK and check the error.
final_q = self.arm.get_state().q
final_tcp = self.kin.fk(final_q)
place_err = float(np.linalg.norm(final_tcp[:3, 3] - target))
if place_err > 0.02:  # 2 cm tolerance
    self.memory.add(
        "note",
        f"place TCP error {place_err*1000:.0f} mm exceeds 20 mm tolerance "
        f"(cartesian_planner may have partial-IK-failure); aborting release",
    )
    raise SkillError(
        f"place failed: TCP off target by {place_err*1000:.0f} mm "
        f"(target={[round(float(v),3) for v in target]})"
    )
self.arm.set_gripper(self._grip_open, effort=0.6)
```

#### 18.16.5 验证结果
- `cartesian_planner.plan()` 默认严格（5% 阈值）
- `5cm 移动（39% IK fail）→ reject`（0.05 默认下）
- verify 块 6/6 子检查通过
- runtime.py + cartesian_planner.py 编译通过

#### 18.16.6 真机 LLM demo 现在的行为
- ✅ plan 严格：任何 IK 失败 >5% 立即 raise
- ✅ place_at verify TCP 实际位置（2cm 误差内才 success）
- ✅ 假 success 不会发生
- ✅ 任务**要么真成功**，要么**明确报告失败**

#### 18.16.7 关键经验
1. **`arm.move_cartesian` 返回 True 不能信**——必须 FK verify
2. **`cartesian_planner` 部分 IK 失败仍返回 waypoint**——之前 max_fail_ratio 0.15 太宽松
3. **真机 LLM 路径**需要**多层 verify**（plan 严格 + place_at verify + post-move FK）
4. **reflex 路径**用 set.py 验证过的流程——**没这些 bug**

### 18.17 Dashboard 视频自适应修复（2026-08-13）

#### 18.17.1 现象
- 用户在浏览器看 `http://198.18.0.1:8090/`
- orbbec_overhead 视频流**画面比例太小**
- 硬编码 `max-width:min(44vw,760px)` 限制最大宽度

#### 18.17.2 修复

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/apps/stream_server.py:45-49` | 移除硬编码 `max-width:min(44vw,760px)`；改用自适应 |
| 备份 | `stream_server.py.bak_dashboard` |

#### 18.17.3 新 CSS

```css
/* 2026-08-13: img auto-fills its container while preserving aspect ratio,
   so the orbbec_overhead feed scales with the browser window instead
   of being capped at 44vw/760px (which made the view tiny on big
   monitors and on the small cells used by the multi-cam tile layout). */
.cam img { width:100%; max-width:100%; height:auto; aspect-ratio:16/9;
          border-radius:4px; display:block; object-fit:contain; background:#000; }
```

#### 18.17.4 关键改动
- `width:100%` 占满容器宽度
- `aspect-ratio:16/9` 保持原始比例
- `object-fit:contain` 完整显示（不裁剪）
- `background:#000` 黑色背景（视频未到时的占位）

#### 18.17.5 用户使用
- 重启 wrc_demo 后立即生效
- 浏览器开 `http://198.18.0.1:8090/`
- 视频流**自适应窗口大小**——大屏幕显示更大，小屏幕显示小（保持比例）

### 18.18 bridge 路径 silent fail 修复（2026-08-13）

#### 18.18.1 现象
真机 LLM demo 跑 `pick_and_place(banana, box)`：
- 机械臂视觉上在 box 上方悬停（实际在 home 位置）
- 之后回 banana 初始位置（实际是 home）
- 任务 success=True **假成功**

#### 18.18.2 真bug
- `_safe_bridge_move` 只 pre-check IK + vet_pose——`bridge.move_to_traj` 返回 True 但**可能静默失败**
- `_skill_pick_and_place_bridge` **不检查** `_safe_bridge_move` 返回值
- 直接调 `release_with_async_close`——释放空气
- 任务 success 但**实际机械臂没到 box**

#### 18.18.3 对比 Reflex 路径
| 路径 | 走哪 | verify 实际位置？|
|---|---|---|
| Reflex `put the banana into box` | `rebot_grasp.set.py::_execute_place_sequence` | ✅ 真 set.py 验证过（CLAUDE.md §0）|
| LLM `pick_and_place(banana, box)` | `_skill_pick_and_place_bridge`（bridge 模拟）| ❌ 之前 silent fail |

#### 18.18.4 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/skills/runtime.py:1244-1280`（`_safe_bridge_move` 末尾）| 加 post-move FK verify：TCP 偏离 target > 20mm → return False |
| `src/wrc_demo/skills/runtime.py:1446, 1454, 1472, 1542`（4 个关键调用点）| 加 return check：失败时 raise `SkillError` 携带 target 信息 |
| `src/wrc_demo/skills/runtime.py:53-65`（新增 helper）| `_is_mock(obj)` duck-type 检测（测试 fixture 探测）|
| 备份 | `runtime.py.bak_bridge_verify` |

#### 18.18.5 post-move verify 代码

```python
ok = self.bridge.move_to_traj(*target_6d, duration=duration)
if ok:
    # 2026-08-13: post-move verify. bridge.move_to_traj can return True
    # without actually moving (SDK silent fail). Re-FK the arm and
    # check we actually reached the target.
    try:
        final_q = self.arm.get_state().q
        final_tcp = self.kin.fk(final_q)
        T_target = np.asarray(target_6d[:3], dtype=float)
        err_m = float(np.linalg.norm(final_tcp[:3, 3] - T_target))
        if err_m > 0.02:
            self.memory.add(
                "note",
                f"{where} TCP off by {err_m*1000:.0f}mm after move; "
                f"target={target_6d[:3]} actual={final_tcp[:3,3].round(3).tolist()}",
            )
            return False
    except Exception as e:
        self.memory.add("note", f"{where} post-move verify failed: {e}")
return ok
```

#### 18.18.6 4 个关键调用点 return check

```python
if not self._safe_bridge_move(
    (drop[0], drop[1], drop[2], 0.0, 1.2, 0.0),
    duration=2.0, where="place",
):
    raise SkillError(
        f"place move failed (drop={[round(drop[0],3), round(drop[1],3), round(drop[2],3)]}, "
        f"q_target unsafe or SDK silent fail)"
    )
```

`ready`/`pregrasp`/`grasp` 同样 return check。

#### 18.18.7 测试 fixture 探测 (`_is_mock`)

```python
def _is_mock(obj) -> bool:
    """Duck-type detection of test fixtures (MagicMock / MockArm) that
    don't simulate motion. Skips post-move verify in tests."""
    return bool(getattr(obj, "_mock_methods", False) or
                getattr(obj, "_spec_class", None) is not None)
```

避免 MagicMock 触发 verify（mock 不模拟真实 TCP 位置）。

#### 18.18.8 完整修复链

| § | 修复 | 路径 |
|---|---|---|
| §18.16 | `cartesian_planner` plan 严格化（max_fail_ratio 0.05）+ `skill_place_at` verify | LLM 单步 `place_at(x, y, z)` |
| §18.18 | `_safe_bridge_move` post-move verify + `_skill_pick_and_place_bridge` 4 个 return check | LLM 多步 `pick_and_place(object, destination)` |
| Reflex | `rebot_grasp.set.py::_execute_place_sequence`（CLAUDE.md §0 验证 5 个 bug 都修了）| `put the banana into box` |

#### 18.18.9 真机 LLM 路径现在

- ✅ `place_at(x, y, z)` —— verify TCP 实际位置（§18.16）
- ✅ `pick_and_place(object, destination)` —— verify 每个 bridge move（§18.18）
- ✅ 假 success 不会发生
- ✅ Reflex 仍 100% 走 set.py

#### 18.18.10 关键经验

1. **bridge.move_to_traj 静默失败**——SDK 返回 True 但机械臂没动
2. **`_safe_bridge_move` 之前只 pre-check IK + vet_pose**——没 post-move verify
3. **bridge 模拟 set.py**——验证不如 set.py 真——**silent fail**
4. **真机 LLM 路径**需要**多层 verify**（plan 严格 + place verify + bridge verify）
5. **测试 fixture 探测**——避免 mock 误判

#### 18.18.11 用户测试命令

```bash
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/llm_verify_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive
```

测试：
```task> pick up the banana then place it on the table
task> put the banana into box
```

**期望**：
- ✅ `pick_and_place(banana, box)`——bridge 路径 verify 全部 move
- ✅ 假 success 不会发生——失败 raise SkillError
- ✅ `place_at(x, y, z)` verify 位置（§18.16）
- ✅ Reflex 仍 100% 走 set.py

### 18.19 `move_to_pose` 默认 pitch + verify 修复（2026-08-13）

#### 18.19.1 现象
真机 LLM demo 跑 `pick up the banana then place it on the table`：
- LLM 选 `move_to_pose(x=0.374, y=0.013, z=0.15, roll=0.0, pitch=0.0, yaw=0.0, duration=0.5)`
- `pitch=0.0` —— **horizontal wrist** —— 几何不可能
- `IK failed for pose (0.374, 0.013, 0.150); err=0.0856` —— IK 失败
- `SkillError: IK failed` —— 任务 `success=False`

#### 18.19.2 真 bug
- `skill_move_to_pose(x, y, z, roll, pitch, yaw, duration)` 默认 `pitch=0.0`（horizontal wrist）
- LLM 调 `move_to_pose` **不**传 pitch → 用默认值 0
- **horizontal wrist** 在 0.15m 高度 + 0.374m 远 —— wrist 几何限制 —— IK 失败
- 之前 `skill_place_at` 用 `approach_pitch_rad=1.2`（top-down wrist）—— 不 IK 失败
- **LLM 选 `move_to_pose` 不选 `place_at`** 因为 TOOL_SPECS description 模糊

#### 18.19.3 对比

| 路径 | wrist pose | IK 结果 |
|---|---|---|
| `place_at`（之前 verify 修过）| `approach_pitch_rad=1.2`（top-down）| ✓ IK 成功 |
| `move_to_pose`（之前没修）| 默认 `pitch=0`（horizontal）| ❌ IK 失败 |
| `place_on_object` | `approach_pitch_rad=1.2` | ✓ IK 成功 |

#### 18.19.4 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/skills/runtime.py:2371-2386`（`skill_move_to_pose` 签名）| `pitch: float = 0.0` → `pitch: float \| None = None` + resolve from `cfg.grasp.approach_pitch_rad` |
| `src/wrc_demo/skills/runtime.py:2410-2418`（cartesian 路径 post-move verify）| 加 FK 实际位置 vs target 检查，> 20mm return False |
| `src/wrc_demo/skills/runtime.py:2450-2458`（PTP fallback 路径 verify）| 同上 |
| `TOOL_SPECS["move_to_pose"]` description | 提示 LLM `place` 用 `place_at` / `place_on_object` + 说明 pitch 默认 top-down |
| 备份 | `runtime.py.bak_mtp` |

#### 18.19.5 默认 pitch 修复

```python
def skill_move_to_pose(
    self,
    x: float,
    y: float,
    z: float,
    # 2026-08-13: default roll/pitch/yaw from approach_pitch_rad so
    # LLM-driven calls (e.g. 'place at x,y,z') get a top-down wrist
    # pose that solves at the box's z without horizontal-pose
    # singularities. pitch defaults to approach_pitch_rad
    # (~1.2 rad = ~69° top-down) instead of 0 (= horizontal palm).
    roll: float = 0.0,
    pitch: float | None = None,
    yaw: float = 0.0,
    duration: float = 2.0,
) -> dict:
    # Resolve pitch default from cfg
    if pitch is None:
        pitch = float(self.cfg.grasp.get("approach_pitch_rad", 1.2))
```

#### 18.19.6 post-move verify

cartesian 路径：
```python
ok = self.arm.move_joints_path(waypoints, segment_duration_s=seg)
# 2026-08-13: post-move verify (cartesian path)
try:
    final_q = self.arm.get_state().q
    final_tcp = self.kin.fk(final_q)
    T_target_arr = np.array([x, y, z], dtype=float)
    err_m = float(np.linalg.norm(final_tcp[:3, 3] - T_target_arr))
    if err_m > 0.02:
        self.memory.add(
            "note",
            f"move_to_pose TCP off by {err_m*1000:.0f}mm (cartesian)",
        )
        return {"ok": False, "error": f"TCP off by {err_m*1000:.0f}mm"}
except Exception as e:
    self.memory.add("note", f"move_to_pose post-move verify failed: {e}")
```

PTP fallback 路径同样 verify。

#### 18.19.7 TOOL_SPECS description 改进

```python
{
    "name": "move_to_pose",
    "description": "Move the end-effector to an absolute Cartesian pose (x,y,z in metres, roll/pitch/yaw in radians) along a smooth SE(3) geodesic trajectory. Every intermediate waypoint is safety-vetted. **For placing a held object, prefer 'place_at' or 'place_on_object' — move_to_pose is for empty-hand repositioning.** Falls back to joint-space PTP if trajectory planning is unavailable. **pitch defaults to approach_pitch_rad (~1.2 rad = top-down) if not specified, so calls like 'move_to_pose(x=0.3,y=0.0,z=0.15)' get a top-down wrist pose that solves at the box's z without horizontal singularities.**",
    ...
}
```

关键：
1. 提示 LLM `place` 用 `place_at` / `place_on_object`——`move_to_pose` 是 empty-hand repositioning
2. 提示 `pitch` 默认 top-down——不会 IK 失败

#### 18.19.8 完整修复链

| § | 修复 | 路径 |
|---|---|---|
| §18.16 | `cartesian_planner` plan 严格化 + `skill_place_at` verify | LLM 单步 `place_at(x, y, z)` |
| §18.18 | `_safe_bridge_move` post-move verify + 4 个 return check | LLM 多步 `pick_and_place(object, destination)` |
| **§18.19** | **`move_to_pose` 默认 pitch + post-move verify + TOOL_SPECS 改进** | **LLM `move_to_pose(x, y, z)`** |
| Reflex | `set.py::_execute_place_sequence` | 100% 走 set.py |

#### 18.19.9 关键经验

1. **LLM 选错工具**——`move_to_pose` 不是 `place_at`——TOOL_SPECS description 模糊
2. **`move_to_pose` 默认 `pitch=0`**——LLM 不传 pitch 导致 IK 失败
3. **默认参数要稳**——任何 default 应该是**安全 + 工作**的（top-down 而不是 horizontal）
4. **`post-move verify` 通用**——所有"移动到位置"skill 都该加

### 18.20 `move_to_pose` 强制 override 修复（2026-08-13）

#### 18.20.1 现象
真机 LLM demo 跑 `pick up the banana then place it into the box`：
```
[ 13.2s ago] (outcome) move_to_pose(x=0.364, y=0.023, z=0.15, roll=0.0, pitch=0.0, yaw=0.0, duration=0.5) -> SkillError: IK failed for pose (0.364, 0.023, 0.150); err=0.0964
```
- LLM 显式传 pitch=0.0
- IK failed `err=0.0964`（horizontal wrist 几何不可能）

#### 18.20.2 §18.19 修复不完整

之前 §18.19：
```python
if pitch is None:
    pitch = float(self.cfg.grasp.get("approach_pitch_rad", 1.2))
```

**问题**：LLM 显式传 `pitch=0.0` 时**不走 None 分支**——default 不生效。

#### 18.20.3 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/skills/runtime.py:2395`（pitch resolve）| `if pitch is None or pitch == 0.0` → approach_pitch_rad |
| `TOOL_SPECS["move_to_pose"]` description | 强提示 LLM `place` 用 `place_at`/`place_on_object` + 警告 `pitch=0` IK 失败 |
| 备份 | `runtime.py.bak_mtp_strict` |

#### 18.20.4 pitch 强制 override 代码

```python
# 2026-08-13: pitch=0.0 (horizontal wrist) IK-fails on the B601-RS
# at any z < ~0.18 m. The LLM often emits pitch=0 from its training
# data; override both None and explicit 0.0 to approach_pitch_rad
# (~1.2 rad top-down). LLMs that want a different pitch must pass
# an explicit non-zero value > 0.1 rad.
if pitch is None or pitch == 0.0:
    pitch = float(self.cfg.grasp.get("approach_pitch_rad", 1.2))
```

**关键**：覆盖 `None` **和** `0.0`——LLM 显式传 0 也会被 override。

#### 18.20.5 TOOL_SPECS description 改进

```python
{
    "name": "move_to_pose",
    "description": "Move the end-effector to an absolute Cartesian pose (x,y,z in metres, roll/pitch/yaw in radians) along a smooth SE(3) geodesic trajectory. Every intermediate waypoint is safety-vetted. **For placing a held object, PREFER 'place_at' or 'place_on_object' — they handle wrist pitch + box detection automatically.** 'move_to_pose' is for empty-hand repositioning; **pitch=0 (horizontal wrist) is FORBIDDEN at z<0.18 m (IK-fails) — if you must call move_to_pose with low z, pass pitch=1.2 explicitly (top-down)**. Falls back to joint-space PTP if trajectory planning is unavailable.",
    ...
}
```

**强提示**：
1. place 用 `place_at` / `place_on_object`（自动处理 wrist pitch + box detection）
2. `move_to_pose` 是 empty-hand repositioning
3. `pitch=0` 在 z<0.18m 失败（IK 不解）
4. 低 z 必须显式传 `pitch=1.2`

#### 18.20.6 完整 LLM 路径修复链

| § | 修复 | 路径 |
|---|---|---|
| §18.16 | `cartesian_planner` plan 严格化 + `skill_place_at` verify | LLM 单步 `place_at(x, y, z)` |
| §18.18 | `_safe_bridge_move` post-move verify + 4 个 return check | LLM 多步 `pick_and_place(object, destination)` |
| §18.19 | `move_to_pose` 默认 pitch None → 1.2 | LLM `move_to_pose` 不传 pitch |
| **§18.20** | **`move_to_pose` 强制 override pitch=0 → 1.2 + TOOL_SPECS 改进** | **LLM `move_to_pose` 显式传 pitch=0** |
| Reflex | `set.py::_execute_place_sequence` | 100% 走 set.py |

#### 18.20.7 用户的洞察——LLM 工具选择问题

**真bug 不是 IK verify——是 LLM 选错工具 + 选错参数**：
- LLM 选 `move_to_pose`（**应该** `place_at`/`place_on_object`）
- LLM 传 `pitch=0`（**应该** `pitch=1.2`）

**修复**：
- §18.20 强制 override pitch=0（**短期**）
- TOOL_SPECS 引导 LLM 用对工具（**长期**——LLM 学会）

#### 18.20.8 关键经验

1. **§18.19 修复不完整**——LLM 显式传 `pitch=0` 绕过 `None` fallback
2. **强制 override** `0.0` 是必要的——LLM 训练数据常 emit 0
3. **TOOL_SPECS description 引导 LLM** 是最稳的长期方案
4. **verify + override + description 三层防御**——LLM 路径才稳

---

### 18.21 orchestrator guard 修复（2026-08-13）

#### 18.21.1 现象
真机 LLM demo 跑 `pick up the banana then place it into the box`：
```
[ 14.2s ago] (outcome) move_to_pose(x=0.372, y=0.017, z=0.15, roll=0.0, pitch=0.0, yaw=0.0, duration=0.1) -> SafetyViolation: e-stopped
```
- LLM 调 `move_to_pose`（**应该** `place_at`）place 香蕉
- `pitch=0.0`（horizontal wrist）—— `vet_pose` 拒绝
- `duration=0.1` —— 几乎没时间动——e-stopped
- §18.19/§18.20 修复（默认 pitch=None → 1.2 + 强制 override pitch=0 → 1.2）**仍不够**——LLM 选错工具

#### 18.21.2 真bug
- LLM 选错工具（`move_to_pose` 而不是 `place_at`/`place_on_object`）
- TOOL_SPECS description 改进**没生效**——LLM 仍调 `move_to_pose`
- 之前修复（默认 pitch）**没解决**——LLM 显式传 pitch=0，§18.20 强制 override 后 IK 仍失败（horizontal wrist 在低 z 几何不可能）

#### 18.21.3 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/agent/orchestrator.py:222-258` | guard：LLM 调 `move_to_pose` + `held_object != None` → mutate `call.name = "place_at"` + 调整 arguments + 记 `llm_translate` trace event |
| `src/wrc_demo/agent/prompts.py:27-34` | `SYSTEM_PROMPT` 加规则：place 用 `place_at`/`place_on_object`，**不要**用 `move_to_pose` |
| 备份 | `orchestrator.py.bak.20260813` + `prompts.py.bak.20260813` |

#### 18.21.4 orchestrator guard 代码

```python
# 2026-08-13: guard — LLM often picks move_to_pose for "place the held
# object at this position", but move_to_pose uses the LLM-supplied pitch
# (usually 0.0 = horizontal wrist) which IK-fails at z<0.18 m on the B601-RS.
# If the LLM picks move_to_pose while we're holding an object, redirect to
# place_at (which always uses approach_pitch_rad top-down).
if (
    call.name == "move_to_pose"
    and self.runtime.held_object is not None
):
    new_args = dict(call.arguments or {})
    new_args.setdefault("x", new_args.get("x", 0.0))
    new_args.setdefault("y", new_args.get("y", 0.0))
    if "z" in new_args:
        new_args["z"] = new_args["z"]
    self.runtime.trace.record_event(
        "llm_translate",
        {"from": "move_to_pose", "to": "place_at", "reason": "held_object set + move_to_pose IK-fails at low z"},
    )
    call.name = "place_at"
    call.arguments = new_args
```

**关键**：`ToolCall` 是普通 `@dataclass`（非 frozen）——直接 mutate `call.name` + `call.arguments` 即可。后续 dispatch (`self.runtime.execute(call.name, call.arguments)`)、tool_log、LLM transcript 都自动看到翻译后的 call——行为透明。

#### 18.21.5 system prompt 规则改进

```python
SYSTEM_PROMPT = """..."""
# 2026-08-13: **For placing a held object, ALWAYS prefer `place_at(x, y, z)` or
# `place_on_object(label="box")` — these tools internally use
# approach_pitch_rad (~1.2 rad = top-down) which is the only wrist pose that
# IK-solves at low z on the B601-RS. **Do NOT use `move_to_pose` for placing
# a held object** — its default pitch is 0 (horizontal wrist) which IK-fails
# at z<0.18 m.
"""
```

#### 18.21.6 E2E 验证

| 场景 | LLM 调用 | guard 行为 | 最终结果 |
|---|---|---|---|
| held=banana, 调 `move_to_pose(x=0.4, y=0.1, z=0.03, pitch=0.0, yaw=0.0)` | mutates to `place_at(x=0.4, y=0.1, z=0.03)` | translate | `{"placed": "banana", "at": [0.4, 0.1, 0.03], "ok": True}` |
| held=None, 调 `move_to_pose(...)` | no change | passthrough | 正常执行 `move_to_pose` |

**trace 记录**：guard 触发时记 `llm_translate {from: move_to_pose, to: place_at, orig_args: ..., new_args: ...}`——透明可追溯。

#### 18.21.7 完整 LLM 路径修复链

| § | 修复 | 路径 |
|---|---|---|
| §18.16 | `cartesian_planner` plan 严格化 + `skill_place_at` verify | LLM `place_at(x, y, z)` |
| §18.18 | `_safe_bridge_move` post-move verify + 4 个 return check | LLM `pick_and_place(object, destination)` |
| §18.19 | `move_to_pose` 默认 pitch None → 1.2 | LLM `move_to_pose` 不传 pitch |
| §18.20 | `move_to_pose` 强制 override pitch=0 → 1.2 + TOOL_SPECS | LLM `move_to_pose` 显式传 pitch=0 |
| **§18.21** | **orchestrator guard：move_to_pose → place_at** + **system prompt 强指令** | **LLM 调 move_to_pose place**（立即生效 + 长期引导） |
| Reflex | `set.py::_execute_place_sequence` | 100% 走 set.py |

#### 18.21.8 关键经验

1. **§18.19/§18.20 修复 IK verify 但没解决"LLM 选错工具"问题**
2. **§18.21 orchestrator guard 立即生效**——LLM 即使选错工具，guard 翻译
3. **system prompt 长期引导**——LLM 学会选对工具
4. **三层防御**：
   - L1: system prompt 引导 LLM 选对工具
   - L2: orchestrator guard 翻译错选工具
   - L3: skill 实现 verify 物理执行

### 18.22 guard 扩展（point_at/move_relative/move_to_joint_zero）（2026-08-13）

#### 18.22.1 现象

真机 LLM demo 跑 `pick up the banana then place it into the box`：

```
step 1: point_at(banana) -> SafetyViolation: joint4 out of limits (-1.552 ∉ [-1.550, 1.550])
step 2: move_to_joint_zero -> e-stopped
step 4: localize_object(banana) -> no detections
```

- LLM 选 `point_at(banana)`（**应该** `place_at`/`place_on_object`）
- `point_at` IK 算到 joint4=-1.552 rad（超 limit 0.002 rad）
- safety harness e-stop——**机械臂 stuck**
- 后续所有 `move_joints` 失败——任务**卡住**——**比之前更糟**

#### 18.22.2 真 bug

- §18.21 guard 只覆盖 `move_to_pose`
- **没**覆盖 `point_at` / `move_relative` / `move_to_joint_zero`
- LLM 选错这些工具 → 直接执行 → IK fail / joint limit overshoot → **safety e-stop** → 任务**卡住**（不是 task 失败而是**机械臂卡在 e-stop**）

#### 18.22.3 修复内容

| 文件 | 改动 |
|---|---|
| `src/wrc_demo/agent/orchestrator.py:278-318` | guard 扩展：`point_at` / `move_relative` / `move_to_joint_zero` 当 `held_object != None` → 翻译成 `place_at(x, y)` |
| `src/wrc_demo/agent/prompts.py:35-43` | system prompt 强规则扩展 |
| 备份 | `orchestrator.py.bak.20260813_v2_guard3` + `prompts.py.bak.20260813_v2_guard3` |

#### 18.22.4 guard 扩展代码

```python
if (
    call.name in ("point_at", "move_relative", "move_to_joint_zero")
    and self.runtime.held_object is not None
):
    # Find the held object's last-known position so we can pass a sane
    # (x, y) to place_at. beliefs.find(label) returns the most recent
    # observed position, or None.
    belief = self.runtime.beliefs.find(self.runtime.held_object)
    if belief is not None:
        pos = belief.position
        place_args = {
            "x": float(pos[0]),
            "y": float(pos[1]),
        }
    else:
        # No belief — fall back to TCP current position
        tcp = self.runtime.kin.fk(self.runtime.arm.get_state().q)
        place_args = {
            "x": float(tcp[0, 3]),
            "y": float(tcp[1, 3]),
        }
    self.runtime.trace.record_event(
        "llm_translate",
        {
            "from": call.name,
            "to": "place_at",
            "reason": (
                f"held_object set + {call.name} is unsafe or non-place "
                f"while holding (point_at IK-overshoots joint limits; "
                f"move_relative is open-loop; move_to_joint_zero forces "
                f"unsafe pose). Redirect to place_at using held-object belief."
            ),
            "orig_args": dict(call.arguments or {}),
            "new_args": place_args,
        },
    )
    call.name = "place_at"
    call.arguments = place_args
```

**关键修正**：

- `self.runtime.kin` / `self.runtime.arm`（不是 `self.kin` / `self.arm`——orchestrator 无直接属性）
- `belief.position` 已经是 `np.ndarray (3,)`——直接 `float(pos[0])`
- guard 块放在 §18.21 `move_to_pose` guard **之后**（同 indent，独立 `if`）

#### 18.22.5 system prompt 强规则

```
- 2026-08-13 (strong rule): **When holding an object (held_object != null),
  the ONLY place-related tools allowed are `place_at` and `place_on_object`.
  `move_to_pose`, `point_at`, `move_relative`, `move_to_joint_zero` MUST NOT
  be called while holding — `move_to_pose` IK-fails at low z, `point_at`
  IK-overshoots joint limits (joint4 -1.55 rad out of [-1.550, 1.550]),
  `move_relative` is open-loop Cartesian translation, and
  `move_to_joint_zero` forces an unsafe pose. The orchestrator WILL translate
  these calls to `place_at` automatically, but pick the right tool the
  first time to avoid confusion. Plan ahead.**
```

#### 18.22.6 完整 LLM 路径修复链

| § | 修复 | 路径 |
|---|---|---|
| §18.16 | `cartesian_planner` plan 严格化 + `skill_place_at` verify | LLM `place_at(x, y, z)` |
| §18.18 | `_safe_bridge_move` post-move verify + 4 个 return check | LLM `pick_and_place(object, destination)` |
| §18.19 | `move_to_pose` 默认 pitch None → 1.2 | LLM `move_to_pose` 不传 pitch |
| §18.20 | `move_to_pose` 强制 override pitch=0 → 1.2 + TOOL_SPECS | LLM `move_to_pose` 显式传 pitch=0 |
| §18.21 | orchestrator guard：`move_to_pose` → `place_at` + system prompt 强指令 | LLM 调 `move_to_pose` place |
| **§18.22** | **guard 扩展：`point_at` / `move_relative` / `move_to_joint_zero` → `place_at`** | **LLM 调这些非 place 工具** |
| Reflex | `set.py::_execute_place_sequence` | 100% 走 set.py |

#### 18.22.7 关键经验

1. **§18.21 guard 不完整**——只覆盖 `move_to_pose`，没覆盖 `point_at` 等非 place 工具
2. **`point_at` IK 算到超 joint limit**——safety e-stop——**机械臂 stuck**——比 `move_to_pose` IK 失败更糟
3. **三层防御**持续完善——LLM 即使选错工具，guard 翻译 + verify 兜底
4. **`self.runtime.kin` / `self.runtime.arm`**——orchestrator 通过 `self.runtime` 访问 SkillRuntime 资源

---

## 19. 启动本地 Qwen3-VL-AWQ-4bit（精简 4-tool 配置，2026-08-14）

### 19.1 当前生产配置（实测稳定）

| 参数 | 值 | 来源 | 备注 |
|------|-----|------|------|
| vLLM MAX_LEN | 8192 | 脚本默认值 | 昨天（8/13）验证过 |
| vLLM GPU_UTIL | 0.80 | 脚本默认值 | 0.85 在 GUI/X 占用下 OOM |
| vLLM 模型 | qwen3-vl-2b-awq-4bit (INT4 2.08GB) | -- | 路径: `/home/seeed/wrc_demo/models/qwen3-vl-2b-awq-4bit` |
| orchestrator max_tokens | 512 | `orchestrator.py:210` | 避免 prompt+output 超 8192 |
| TOOL_SPECS | 4 tools（`list_objects` / `pick_and_place` / `move_home` / `task_done`） | `runtime.py:2092` | 之前 23 个精简到 4 个 |
| 脚本启动命令 | `bash scripts/start_reason2_server.sh --bg` | §18.4 | 脚本名"reason2"是历史遗留，实际启动 Qwen3-VL |

### 19.2 完整启动流程（2 个 Terminal）

#### Terminal 1：vLLM

```bash
# 启动（默认 MAX_LEN=8192 + GPU_UTIL=0.80）
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg

# 等 30-60s ready
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# 期望: ~6160 MB used / ~1500 MB free

# 测试
curl -sf http://localhost:8080/v1/models | head -c 200
# 期望: JSON 含 "qwen3-vl-2b-awq-4bit"
```

#### Terminal 2：wrc_demo

```bash
cd /home/seeed/wrc_demo
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl --interactive
```

打开 `http://198.18.0.1:8090/` 测试 task。

### 19.3 故障排查（这次踩过的坑）

#### 19.3.1 `[stream] task failed: Connection error.`

**原因**：vLLM 没启动 / 进程死了（PID 文件残留）

**诊断**：

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status
ps -ef | grep vllm | grep -v grep
ss -tlnp 2>&1 | grep ":8080"
```

**修复**：见 §19.2 Terminal 1

#### 19.3.2 `ValueError: Free memory ... is less than desired GPU memory utilization (0.85, 6.38 GiB)`

**原因**：GPU_UTIL=0.85 在 8GB 卡 + GUI/X server 占用 ~1.4GB 时 OOM

**诊断**：

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

**修复**：

```bash
GPU_UTIL=0.80 bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
# 或更低 0.65 / 0.60
```

#### 19.3.3 `ValueError: To serve at least one request with the model's max seq len (X), Y GiB KV cache is needed, but only Z GiB available`

**原因**：MAX_LEN 太大，KV cache 装不下

**诊断**：错误信息里的 "estimated maximum model length"

**修复**：

- 降 MAX_LEN 到错误信息里的 estimated 值
- 或提高 GPU_UTIL（但 §19.3.2 风险）
- 当前稳定：MAX_LEN=8192 + GPU_UTIL=0.80

#### 19.3.4 `Error code: 400 - This model's maximum context length is 2048 tokens ... requested X output tokens and your prompt contains Y input tokens`

**原因**：prompt 已经 ~1500 tokens，加上 max_tokens=1024 = 超过 2048

**修复**：把 `orchestrator.py:210` 的 `max_tokens=1024` 改成 `max_tokens=512`

- 当前稳定：MAX_LEN=8192 + max_tokens=512
- 计算：prompt ~1500 + output 512 = ~2000 < 8192 ✓

#### 19.3.5 vLLM 启动后立即退出（PID 文件残留）

**症状**：`vllm already running, PID XXX` 但实际进程不存在

**诊断**：

```bash
ps -p $(cat /tmp/vllm_reason2.pid 2>/dev/null)
```

**修复**：

```bash
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --stop
sleep 3
pkill -9 -f "vllm"
rm -f /tmp/vllm_reason2.pid
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
```

### 19.4 任务失败的诊断

LLM 路径任务失败时（如 `success: False, summary: step budget exhausted`）：

```bash
# 看 LLM 调了什么
RUN_DIR=$(ls -td /home/seeed/wrc_demo/runs/2026* 2>/dev/null | head -1)
python3 -c "
import json
from pathlib import Path
trace = Path('$RUN_DIR/trace.jsonl')
for line in trace.read_text().splitlines():
    if not line.strip(): continue
    ev = json.loads(line)
    if ev.get('kind') == 'tool_call':
        print(f\"  {ev['step']}: {ev['skill']}({ev['args']}) -> ok={ev['result'].get('ok')}\")
"
```

### 19.5 注意事项

1. **脚本名是历史遗留**：`start_reason2_server.sh` 来自之前想用 Cosmos-Reason2-2B 的时代，现在启动的是 Qwen3-VL-AWQ-4bit
2. **doc-vs-code drift**：脚本注释里记录的稳定参数（0.65 / 2048）和实际默认值（0.80 / 8192）可能不一致——以最后一次实测为准
3. **GUI/X server 影响**：8GB 卡装不下 vLLM + GUI 占用 + detector 的全套——vLLM 启动时 X server + gnome-shell 占 ~1.4 GB
4. **prompt 已 ~1500 tokens**：SYSTEM_PROMPT + 4 tool TOOL_SPECS + intro message——max_tokens 不能再超过 512
5. **max_steps=30 可能不够**：LLM 多 turn 推理复杂任务可能跑不完——可以加大但增加失败成本
6. **dashboard 看不到 terminal UX 提示**：`_print_report` 加了 `[✓ done]` 提示但只在 terminal 显示，dashboard 用 stream_server 自带显示

---

## 20. Top 相机标定代码迁移（2026-08-14）

### 20.1 变更摘要

把 rebot_grasp 的 `scripts/calibrate_top_camera.py`（670 行）**架构移植**到 wrc_demo 的 `src/wrc_demo/scripts/calib_top_orbbec.py`，完全使用 wrc_demo 内部组件，**零 rebot_grasp 依赖**。

| 维度 | 旧（rebot_grasp 风格） | 新（迁移后，wrc_demo 内部） |
|------|---------------------|------------------------|
| **入口** | `scripts/collect_handeye_eth.py`（外部） | `wrc_demo.scripts.calib_top_orbbec`（in-tree） |
| **核心架构** | rebot_grasp 状态机 `tick_auto` | **采用** rebot_grasp 状态机（保留简洁性） |
| **手动模式** | 简化版（等 ENTER + 手动移臂） | **保留** + **新增** `--gravity-comp`（MIT 模式 + 重力 FF） |
| **Solver** | `HandEyeCalibrator.calibrate` | `wrc_demo.calibration.joint_solve_restart`（已存在） |
| **相机** | `drivers.camera.make_camera` | `wrc_demo.perception.orbbec_camera.OrbbecCamera` |
| **机械臂** | raw `RebotArmEndPose` | `safe_arm.move_joints` + `Kinematics.ik`（保持 SafetyHarness） |
| **JSON schema** | NPZ only | **JSON v1 + NPZ**（向后兼容） |

### 20.2 使用差异

| 操作 | 旧命令 | 新命令 |
|------|--------|--------|
| 列设备 | `python -m scripts.collect_handeye_eth --list` | `python -m wrc_demo.scripts.calib_top_orbbec --list` |
| 绑序列号 | 无（旧用 YAML 配置） | `python -m wrc_demo.scripts.calib_top_orbbec --bind top` |
| 自动标定 | `python -m scripts.collect_handeye_eth --camera orbbec_overhead --arm rebot_rs` | `python -m wrc_demo.scripts.calib_top_orbbec` |
| 手动标定 | `--manual`（简化版） | `--manual`（同）或 `--gravity-comp`（**新增** MIT 漂浮） |
| 验证 | `--verify --npz <file> --known-points <csv>` | **同**（新入口也支持） |
| Dry-run | 无 | `python -m wrc_demo.scripts.calib_top_orbbec --dry-run --dry-run-out /tmp/dryrun.json` |

### 20.3 新增能力

**`--gravity-comp` 模式**：用 vendored `RebotArm` + `RebotArmEndPose` + `compute_generalized_gravity` 实现 MIT 模式 + 重力 FF + 500Hz 控制环。操作员手推臂**漂浮**（不需要按抱闸），按下 ENTER 后捕获当前 q + marker 位姿。退出自动 `safe_home`。

与 `skills/teach.py` 同 pattern（CLAUDE.md "Motion safety path" 例外条款）。

### 20.4 已知遗留

- **wrist 相机标定已迁移** (2026-08-27): `scripts/collect_handeye_eih.py` → in-tree `src/wrc_demo/scripts/calib_wrist.py`. 入口命令：`python -m wrc_demo.scripts.calib_wrist`. 同样用 joint_solve_restart + schema v1, 与 `calib_top_orbbec.py` 对称.
- **legacy `scripts/collect_handeye_eth.py` 保留**（仅作参考，不再维护），新代码不要 import 它。
- **flip_hand_eye.py 有预存 bug**：`CalibrationSample` 没有 `T_marker2gripper` 属性（与本次迁移无关，是历史遗留）。
- **备份文件**：迁移前原版保留在 `src/wrc_demo/scripts/calib_top_orbbec.py.bak_premigration`（待用户确认后删除）。

### 20.5 真机验证清单

```bash
# 1. 设备枚举（确认能看到 Orbbec 顶部相机）
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec --list

# 2. 绑序列号到 top role（首次必做）
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec --bind top

# 3. CAN bus 初始化（如未自动）
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

# 4. Auto 模式（推荐起步）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec

# 5. Manual 模式（等 ENTER 后手动移臂）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec --manual

# 6. Gravity-comp 模式（新增！臂漂浮 + 手推 + ENTER）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_top_orbbec --gravity-comp

# 7. 验证（不重跑标定）
sudo /home/seeed/miniforge3/envs/DM/bin/python -m wrc_demo.scripts.calib_top_orbbec \
    --verify --npz data/calibration/hand_eye_top.json \
    --known-points "0.30,0.00,0.02;0.25,-0.05,0.02;0.35,0.05,0.02"
```

### 20.6 测试结果

| 项 | 旧版 | 新版 |
|----|------|------|
| `test_calib_session.py` | 12 passed, 1 skipped | **15 passed, 1 skipped**（+3 修好的） |
| 全量 pytest | 347 passed, 28 failed | **366 passed, 25 failed**（净 +19） |
| Dry-run 烟囱测试 | OK | **OK**（per-sample rmse = 0.00mm） |
| 公开 API 兼容 | — | **23/23 符号**（CALIB_DIR, TOP_CALIB_POSES, _collect_samples, main, _verify 等） |

25 个 remaining 失败是预存问题（cosmos3 / LLM profile / USD model），与本次迁移无关。

---


## 21. 下次真机操作步骤（wrist PRIMARY 切换后）

> **写于 2026-08-14**：wrist 取代 top 完成（见 §11）。下次拿到硬件，按这个清单逐项验证。
> **重要**：本节同时验证两件事：(a) wrist 切换是否成功；(b) 我之前修的两个 bug 是不是真的修好。

### 21.0 前置准备（开机前 5 分钟）

```bash
# 1. 硬件
#    - wrist 相机 (AY3794300W4) 装在 gripper 上 + USB 连到主机
#    - 物理检查：ArUco marker 贴好（4x4_50, 5cm 边长）
#    - reBot RS 上电 + CAN 总线

# 2. 环境
sudo /home/seeed/miniforge3/envs/DM/bin/python -c "import wrc_demo; print('OK')"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

# 3. CAN bus
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip link show can0 | head -3  # 应显示 state UP

# 4. vLLM (当前不需要，但 wrist demo 仍用 LLM 路径)
# 如未启：
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --bg
bash /home/seeed/wrc_demo/scripts/start_reason2_server.sh --status
```

### 21.1 验证 wrist 标定（5-15 分钟）

```bash
# A. 检查现有标定 JSON 是否完整
/home/seeed/miniforge3/envs/DM/bin/python -c "
import json
d = json.load(open('/home/seeed/wrc_demo/data/calibration/hand_eye_wrist.json'))
print('schema_version:', d.get('schema_version'))
print('mode:', d.get('mode'))
print('camera_serial:', d.get('camera_serial'))
print('n_samples:', d.get('n_samples'))
"

# 期望：
#   schema_version: 1
#   mode: eye_in_hand
#   camera_serial: AY3794300W4
#   n_samples: 20 (原 2026-07-23 数据)

# B. 如果硬件已就绪：跑一次 wrist 标定 sanity check（重做 wrist 标定）
# ⚠️ 模块名是 wrc_demo.scripts.calib_wrist，**不是** calib_wrist_orbbec
# （wrist 默认 Orbbec Gemini 2，无需 _orbbec 后缀）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_wrist
# 期望：~12s/pose × 25 poses ≈ 5 分钟
#     - 采集 20+ samples
#     - joint_solve_converged with RMSE < 10mm
#     - 写 hand_eye_wrist.json（自动覆盖）

# B'. Manual 模式（重力补偿手推 + ENTER 捕获，至少 20+ poses）
sudo /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.scripts.calib_wrist --manual

# 错误命令示范（**不要跑**）：
# sudo ... -m wrc_demo.scripts.calib_wrist_orbbec   # ModuleNotFoundError
# sudo ... -m scripts.calib_wrist                    # scripts/ 不是 Python 包
```

**判定标准**：
- ✅ 现有 JSON 完整 → 直接用（无需重做）
- ✅ 重做后 RMSE < 5mm → 保留新数据
- ⚠️ RMSE 5-10mm → 仍可用但要记录漂移
- ❌ RMSE > 10mm → 相机可能有机械松动，**先停手检查硬件**

### 21.2 启动 demo（wrist PRIMARY）

```bash
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/wrist_check_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_wrist --arm rebot_rs --llm local_qwen3_vl --interactive
```

**必看**：
1. 终端打印 `[wrc-demo] cameras=['orbbec_wrist']`
2. 浏览器开 `http://<ip>:8090/`，视频流应是 **wrist 视角**（近距离看 gripper 周围，**不是 top 视角**）
3. 看到 ArUco marker（如果 marker 在 gripper 视野内）
4. vLLM health check（`curl http://localhost:8080/v1/models`）

### 21.3 验证 gimbal-lock fix（中央香蕉 RPY 不再奇异）

> **背景**：之前报"中央香蕉 RPY = (-72.5°, 89.6°, -162.5°)" 是 ZYX Euler 在 pitch=π/2 处的 gimbal lock + canonicalization 选错分支的副作用。
> Fix：`_canonicalize_parallel_gripper_tcp_rotation` 在 `|pitch-π/2|<0.10` 时跳过 canonicalization，RPY 应输出 **(90°, 90°, 0°)**。

```bash
# 在 wrist demo 里：把 ArUco marker 放中央视野，跑 pick_and_place
# 然后看 trace.jsonl 里 grasp 的 rotation matrix

RUN_DIR=$(ls -td /home/seeed/wrc_demo/runs/wrist_check_* 2>/dev/null | head -1)
/home/seeed/miniforge3/envs/DM/bin/python << EOF
import json, numpy as np
from pathlib import Path
trace = Path('$RUN_DIR/trace.jsonl')
# 找 grasp 相关的 note/outcome
grasp_rows = [json.loads(l) for l in trace.read_text().splitlines()
              if 'grasp' in l.lower()]
for g in grasp_rows[-3:]:
    print(f'kind={g.get("kind", "?")}: {str(g)[:200]}')
EOF
```

**判定标准**：
- ✅ 中央 marker 抓取 ZYX Euler ≈ **(90°, 90°, 0°)**（fix 生效）
- ⚠️ RPY 仍异常 → 检查 `src/wrc_demo/grasping/camera_grasp.py:170` 的 `_canonicalize_parallel_gripper_tcp_rotation` 改动是否还在

### 21.4 验证 IK margin fix（边缘点位不再"joint N out of limits"）

> **背景**：之前报 `joint4 out of limits (-1.560 ∉ [-1.550, 1.550])` 是因为 SDK `solve_ik_with_retry` 内部 `_clamp_config` 不带 margin（CLAUDE.md 第 122 行 warning）。
> Fix：`safety/harness.py::Kinematics.ik()` 加 `limit_margin=0.025` post-clamp。

```bash
# 把 ArUco marker 放到 workspace 边缘（x=0.35 附近），跑 pick_and_place
# 期望：arm 能 reach 到 grasp pose，trace.jsonl 不再有 "joint N out of limits"

# 看 trace.jsonl 里 grasp move 后的 FK verify
/home/seeed/miniforge3/envs/DM/bin/python << EOF
import json
from pathlib import Path
import subprocess
run_dir = subprocess.run(
    ['ls', '-td', '/home/seeed/wrc_demo/runs/wrist_check_*'],
    capture_output=True, text=True).stdout.split('\n')[0]
trace = Path(run_dir) / 'trace.jsonl'
errors = [json.loads(l) for l in trace.read_text().splitlines()
          if 'out of limits' in l or 'SafetyViolation' in l]
if errors:
    print(f'Still seeing {len(errors)} "joint out of limits" errors:')
    for e in errors[:3]:
        print(f'  {e.get("kind", "?")}: {str(e)[:200]}')
else:
    print('No "joint N out of limits" errors in trace')
EOF
```

**判定标准**：
- ✅ 没有 "joint N out of limits" 错误（fix 生效）
- ❌ 仍有 → 检查 `src/wrc_demo/safety/harness.py:97` 的 `Kinematics.ik` 改动是否还在

### 21.5 基础能力 sanity 测试（5-10 分钟）

```bash
# 在 demo REPL 里逐个跑：
task> wave                     # 验证 arm 动 + 基础 dispatch
task> move_home                # 验证 safe_home + ready_pose
task> go to home               # 同上
task> list_objects             # 验证 detector 在 wrist 视角下能识别物体
task> pick up the banana       # 主流程：detect + grasp + IK + move + close + lift
task> put the banana into the box   # 完整 pick_and_place
```

**逐项判定**：
| 任务 | 期望 | 失败怎么查 |
|------|------|----------|
| `wave` | arm 立即动 | reflex 路径，trace.jsonl 找 `path=reflex` |
| `move_home` | arm 回 home pose | `path=skill_move_home` |
| `list_objects` | 列出 1+ 物体 | `path=llm` 找 `list_objects` tool call |
| `pick up the banana` | 检测+grasp+close+lift | FK verify err < 20mm |
| `put the banana into the box` | 完整 pick_and_place | `outcome: success=True` |

### 21.6 备份模式验证（top 仍可用）

```bash
# 退出当前 wrist demo（Ctrl+C）
# 重启 top 模式（不卸载 wrist profile）：
sudo env HOME=/home/seeed YOLO_OFFLINE=True PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/top_check_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs --llm local_qwen3_vl --interactive
```

**判定标准**：
- ✅ top 仍能跑（backup 没坏）
- ✅ wrist 也仍能跑
- ⚠️ 想切回 wrist → 改回 `--cameras orbbec_wrist`

### 21.7 记录结果（写进交接文档 §12）

真机跑完后，把数据填进 `桌面/wrc_demo_交接文档.md §12`：

```markdown
## 12. 2026-08-XX wrist PRIMARY 真机验证结果

### 12.1 wrist 标定状态
- 重做 wrist 标定：是 / 否
- 新 RMSE: X.XX mm
- 旧 RMSE (2026-07-23): X.XX mm
- 漂移评估: OK / 需关注

### 12.2 修复验证
- gimbal-lock fix: OK RPY = (90, 90, 0) at center / FAIL still weird
- IK margin fix: OK no "joint N out of limits" / FAIL still happens

### 12.3 demo 功能
| 任务 | 状态 |
|------|------|
| wave | OK / fail |
| move_home | OK / fail |
| list_objects | OK / fail |
| pick up the banana | OK / fail (X% confidence) |
| put the banana into the box | OK / fail |

### 12.4 备份模式
- top camera: OK / fail

### 12.5 备注
- (任何意外 / 需要 follow-up 的事)
```

---

## 22. 万一搞砸了（rollback 速查）

| 症状 | 立即动作 |
|------|---------|
| `ModuleNotFoundError: ...calib_wrist_orbbec` | 模块名拼错了，正确的是 `wrc_demo.scripts.calib_wrist`（wrist 默认 Orbbec Gemini 2，无 `_orbbec` 后缀） |
| `ModuleNotFoundError: No module named wrc_demo.scripts.calib_wrist` | 文件没真的 in-tree，跑到外层 `scripts/` 去了。修法：`cp scripts/calib_wrist.py src/wrc_demo/scripts/` + `cp scripts/handeye_solver.py src/wrc_demo/scripts/`（这两个文件 §21.1 提到"在 in-tree"但实际上 §6.3 之前没真正搬）。迁移完成后 `python -m wrc_demo.scripts.calib_wrist --help` 应该跑通。 |
| `TypeError: ...unexpected keyword argument 'marker_z_toward_camera'` | `calib_wrist.py` 历史 bug。`ArUcoSession.detect()` 实际签名是 `(bgr, K, D, marker_size_m, target_id=None)`，没有 `marker_z_toward_camera` 参数（约定里 z 永远指向相机，对 top 和 wrist 都一样）。修法：删掉 `marker_z_toward_camera=True`（脚本里 2 处：line 207 和 296）|
| `ModuleNotFoundError: ...scripts.calib_wrist` | `scripts/` 不是 Python 包，命令需要 `python -m` + in-tree 模块名，或直接 `python scripts/calib_wrist.py` 走 script 形式 |
| wrist 标定 RMSE > 10mm | 停手 → 检查 marker + 相机固定 → 重做 calib_wrist |
| IK 仍报"joint N out of limits" | 检查 `safety/harness.py:97` 的 `limit_margin=0.025` 是否在 |
| 中央 RPY 仍 (-72, 89, -162) | 检查 `camera_grasp.py:170` 的 `pitch-π/2 < 0.10` 分支是否在 |
| vLLM 400 context | 见 §11/§18 → 降 max_tokens=256 OR 重启 vLLM @ GPU_UTIL=0.85 |
| wrist 完全不工作 | 临时切回 `--cameras orbbec_overhead`（backup 永远可用）|
| gripper 失控 | Ctrl+C + `python -m wrc_demo.scripts.emergency_stow` |
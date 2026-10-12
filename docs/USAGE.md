# wrc_demo 完整使用文档

> 本文档基于对仓库源码的全面梳理 + [Seeed Wiki 教程](https://wiki.seeedstudio.com/cn/wrc_demo_tutorial/) 整理而成，覆盖：项目定位、框架结构、环境安装、模型获取、手眼标定、LLM 配置、运行方式、自定义配置、扩展开发、测试与故障排查。

---

## 1. 项目定位

**wrc_demo** 是 Seeed **reBot Arm B601-RS**（RobStride CAN 总线，6 自由度 + 平行夹爪）的视觉-语言模型（VLM）驱动的桌面抓取演示系统。用户用自然语言下指令（如 "pick up the red block"），系统完成：

1. **感知**：RGB-D 相机 → YOLOE 开放词汇检测/分割 → 深度反投影 → 物体 3D 定位（基坐标系 OBB）
2. **决策**：三级快慢路径 **Reflex（正则模板，µs 级）→ Habit（经验记忆回放，ms 级）→ LLM（工具调用循环，2–15 s/轮）**
3. **规划**：三层抓取规划（camera-frame → GraspGen-X 学习型 → 解析式 OBB 兜底）+ Pinocchio IK
4. **执行**：每个 50 Hz 流式轨迹点经 `SafetyHarness` fail-closed 校验后才下发机械臂

**关键特性**：

- 本地 LLM 默认 Qwen3-VL-2B-Instruct-AWQ-4bit，8 GB 显存可跑（~30 tokens/s）
- 全栈 mock（`--camera mock --arm mock --llm mock`）无硬件即可跑通完整决策流水线，mock 与真机走**同一份安全代码**
- 每个技能调用写入 ASPIRE 风格 trace（`trace.jsonl` + 前后关键帧 JPEG）
- MCP 网关可把机械臂技能暴露给 Hermes / Claude Code / Codex 等外部 Agent

### 硬件清单

| 部件 | 说明 |
|---|---|
| reBot Arm B601-RS | 6-DOF + 夹爪，48V 供电，RobStride 总线 |
| USB2CAN 串口桥 | PCAN-USB → can0 @ 1 Mbps |
| RGB-D 相机 | Orbbec Gemini 2（推荐，顶部主相机）；兼容 RealSense D435i/D455f/L515、UVC 摄像头 |
| 控制主机 | Ubuntu 22.04+，NVIDIA GPU ≥ 8 GB 显存（本地 LLM 需要），内存 ≥ 16 GB，磁盘 ≥ 10 GB |
| PiPER Mate 主臂（可选） | 主从遥操作示教录制/复播 |

---

## 2. 框架结构

### 2.1 仓库顶层目录

```
wrc/
├── src/wrc_demo/        # 核心 Python 包（见 2.2）
├── sdk/                 # vendored reBotArm_control_py SDK（树内快照，不 pip 安装）
├── configs/             # YAML 配置：demo.yaml + arms/ + cameras/ + llm/
├── models/              # 模型权重：YOLOE、MobileCLIP2、Qwen3-VL
├── scripts/             # 启动/安装/运维脚本（start_all.sh 等）
├── examples/            # 示教录制/复播示例
├── tests/               # ~290 个单元+集成测试
├── data/                # 标定产物 + 示教录像
├── runs/                # 每次任务的 trace 目录（时间戳命名）
├── assets/urdf/         # 机械臂 URDF 模型
├── docs/                # 架构文档、ADR、规格说明
├── environment.yml      # conda 环境（env 名固定 wrc-demo）
└── pyproject.toml       # CLI 入口点定义
```

### 2.2 核心包结构（src/wrc_demo/）

```
src/wrc_demo/
├── types.py            # 核心数据类型：Frame(BGR+米制深度+内外参) / Detection / ObjectFix / Grasp / RobotState
├── config.py           # YAML profile 合并加载（demo.yaml + cameras/ + arms/ + llm/ → Cfg）
│
├── perception/         # 感知层
│   ├── camera_base.py      # CameraBase 抽象 + make_camera 工厂
│   ├── orbbec_camera.py    # Orbbec Gemini 2 后端（pyorbbecsdk）
│   ├── realsense_camera.py # RealSense D4xx/L515 后端
│   ├── opencv_camera.py    # UVC 纯 RGB 摄像头
│   ├── mock_camera.py      # 合成桌面场景 / .npz 回放
│   ├── detector.py         # YOLOE/YOLO-World 开放词汇检测（OpenVocabDetector）
│   ├── grounding.py        # Extrinsics：检测→mask 深度→反投影→基坐标系 OBB（ETH/EIH）
│   ├── vlm_ground.py       # 检测漏检时调用 VLM 直接出 bbox 的二级定位
│   ├── depth_provider.py   # 深度策略链：传感器→单目估计→桌面平面几何
│   ├── stream.py           # CameraStream 后台最新帧槽线程 + CameraRig 多相机
│   ├── world.py            # WorldWatcher 常驻感知循环（3 Hz 融合进 BeliefStore + 安全心跳）
│   └── colors.py           # mask 中位 HSV → 颜色名（支撑 "pink object" 类查询）
│
├── memory/             # 记忆层
│   ├── beliefs.py          # BeliefStore 物体恒存信念库（线程安全，label+颜色+3D+新鲜度）
│   ├── episodic.py         # EpisodicMemory 15s 情景记忆环
│   ├── grasp_memory.py     # 抓取结果记忆（跨规划层 re-rank 候选）
│   └── turboquant.py       # TurboQuant 在线向量量化检索索引
│
├── control/            # 控制层
│   ├── __init__.py         # make_arm(cfg) 唯一工厂 → MockArm 或 SafeArm
│   ├── kinematics.py       # Kinematics：Pinocchio FK/IK（joint_signs 镜像约定烘焙进模型）
│   ├── mock_arm.py         # 完美跟踪的测试替身
│   ├── lazy_arm.py         # LazyArm 延迟上电（首个运动命令前电机不通电）
│   ├── gripper.py          # 夹爪状态机：IDLE→POSITION→CLOSING→HOLDING，力矩闭合+失速检测
│   └── rebot_rs_arm.py     # 真机后端（mechPos 参数读取等现场修复见 docstring）
│
├── safety/harness.py   # 安全层：SafetyLimits + SafetyHarness（逐路点 fail-closed 审批，8 项检查）
│                       # + SafeArm（经 SDK waypoint_hook 注入安全门控的薄封装）
│
├── planning/cartesian_planner.py  # SE(3) 测地线轨迹 + CLIK 关节跟踪
│
├── grasping/           # 抓取规划（三层）
│   ├── camera_grasp.py     # Layer1：相机系 minAreaRect + 深度分位反投影
│   ├── graspgenx_backend.py# Layer2：GraspGen-X 学习型 6-DoF（ZMQ+msgpack 客户端，:5556）
│   ├── obb_grasp.py        # Layer3：解析式 OBB 顶抓（永远可用的兜底）
│   ├── selector.py         # select_grasp：夹爪宽度过滤 + grasp/pregrasp 双 IK 可达性筛选
│   └── force.py            # 材质感知夹持力度 profile
│
├── agent/              # 决策层
│   ├── reflex.py           # Tier1 FastPlanner 模板语法 + Tier2 ExperienceMemory 习惯回放
│   ├── orchestrator.py     # Tier3 AgentOrchestrator：分解→行动→验证→恢复 工具调用循环
│   ├── advisor.py          # 失败后咨询 VLM，一句话空间建议回注上下文
│   ├── llm.py              # OpenAICompatClient / AnthropicClient / MockLLM 统一接口
│   └── trace.py            # TraceLogger：trace.jsonl + 关键帧 + summary.txt
│
├── skills/             # 技能层
│   ├── runtime.py          # SkillRuntime + TOOL_SPECS（6 个安全门控工具，唯一调度入口）
│   ├── teach.py/master_arm.py  # PiPER Mate 主臂→reBot 从臂遥操示教
│   ├── library.py          # 已验证修复策略持久化技能库
│   └── rebot_grasp_bridge.py   # rebot_grasp 运动/夹爪 API 桥
│
├── calibration/        # 手眼标定
│   ├── handeye_solver.py   # 联合 SE(3) 求解（LM + Huber + 3*MAD 离群剔除）
│   ├── aruco_session.py    # ArUco 检测 + PnP
│   └── calib_dataset.py    # 标定数据读写
│
├── apps/               # 应用入口
│   ├── demo.py             # 主 CLI（build_runtime 组装全部组件）
│   ├── record.py           # 录真实相机帧为 .npz（供 MockCamera 回放）
│   ├── mcp_server.py       # MCP stdio JSON-RPC 服务器（暴露给外部 Agent）
│   ├── viewer.py           # 实时 RGB+深度+检测覆盖查看器
│   └── stream_server.py    # MJPEG 多相机直播 dashboard（纯 stdlib）
│
└── scripts/            # 标定 CLI 实现（calib_top_orbbec / calib_wrist 等）
```

### 2.3 CLI 入口点（pyproject.toml）

| 命令 | 目标 | 说明 |
|---|---|---|
| `wrc-demo` | `apps.demo:main` | 主演示程序：`--task/--cameras/--arm/--llm/--interactive/--no-serve` 等 |
| `wrc-record` | `apps.record:main` | 录真实相机帧为 .npz 数据集 |
| `wrc-mcp` | `apps.mcp_server:main` | MCP stdio 服务器 |
| `wrc-view` | `apps.viewer:main` | 实时查看器 |
| `wrc-calib-top` | `scripts.calib_top_orbbec:main` | 顶部相机眼在手外（ETH）标定 |
| `wrc-calib-wrist` | `scripts.calib_wrist:main` | 腕部相机眼在手上（EIH）标定 |

### 2.4 核心数据流

```
相机 (make_camera) → CameraStream (最新帧槽, drop-stale) → DepthProvider (深度补全)
  → Extrinsics (手眼外参) → OpenVocabDetector (YOLOE)
  → WorldWatcher (3Hz 后台: 检测→颜色命名→ObjectFix→BeliefStore, 同时给 SafetyHarness 心跳)

聊天指令 → Tier1 REFLEX (正则模板) → Tier2 HABIT (经验记忆) → Tier3 LLM (工具调用循环)
  → SkillRuntime.execute (唯一调度入口, 自动 trace)
    → plan_grasps (camera + graspgenx + obb 三层) → select_grasp (宽度+IK 筛选)
    → SafeArm → 每个 50Hz waypoint 经 SafetyHarness.approve 门控
    → gripper 两段式闭合 (力矩+失速检测)
    → TraceLogger + EpisodicMemory + GraspMemory 回流
```

### 2.5 关键设计决策

- **Frame.rgb 是 BGR，depth_m 是米制 float32**：不同相机深度单位差异在边界统一转换，下游单位安全
- **抓取在基坐标系规划**：相机姿态只影响可见性，不影响抓取几何
- **学习抓取优先，解析抓取兜底**：GraspGen-X 服务器任何错误都静默回退 OBB，展位不会因模型服务器挂掉而卡死
- **fail-closed 安全**：任一 waypoint 校验失败立即中途 abort；IK `limit_margin=0.025` 必须严格大于 harness `joint_margin=0.02`（不变量，两个一起调）
- **LazyArm**：聊天服务器启动时不给电机上电，首个运动命令才接通

---

## 3. 环境安装

### 3.1 最小路径（5 分钟跑通 mock）

```bash
git clone https://github.com/TheMoonAstronaut/wrc.git
cd wrc
conda env create -f environment.yml && conda activate wrc-demo

# 验证（约 30 秒）
pytest tests/test_extrinsics_loader.py \
       tests/test_hand_eye_compensation_per_camera.py \
       tests/test_demo_yaml_llm_profile.py -q
```

### 3.2 系统依赖（真机需要）

```bash
sudo apt update
sudo apt install -y libusb-1.0-0-dev ffmpeg git can-utils
```

### 3.3 单独安装 PyTorch（CUDA 版）

torch 不在 environment.yml 里（ultralytics 把 torch 标为 optional）。按自身 CUDA 版本安装，例如 CUDA 12.8：

```bash
conda activate wrc-demo
PYTHONNOUSERSITE=1 pip install --force-reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

安装 nvidia 库路径 hook（否则新 shell 里 `import torch` 报 `libcudart.so` 找不到）：

```bash
bash scripts/install_nvidia_libs_hook.sh
# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3.4 安装深度相机 SDK

**Orbbec Gemini 2（推荐）**：

```bash
conda activate wrc-demo
PYTHONNOUSERSITE=1 pip install pyorbbecsdk2     # 必须加 PYTHONNOUSERSITE=1
```

网络问题可从源码装（`github.com/orbbec/pyorbbecsdk`，国内用 gitee 镜像）。

**RealSense**：`pyrealsense2` 参考 Intel 官方文档编译。

### 3.5 硬件接线与权限（真机）

```bash
# 一次性 USB 权限
sudo chmod a+rw /dev/bus/usb/*/*

# CAN 总线（机械臂）
sudo modprobe peak_usb
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

---

## 4. 模型权重获取

| 资产 | 大小 | 必要性 | 获取方式 |
|---|---|---|---|
| `models/qwen3-vl-2b-awq-4bit/` | ~2.4 GB | 本地 LLM 必装 | `hf download cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit --local-dir models/qwen3-vl-2b-awq-4bit` |
| `models/yoloe-26s-seg.pt` | ~31 MB | 开放词汇检测（当前默认） | 从 Seeed-Projects/reBot-DevArm-Grasp 拷贝 |
| `models/yoloe-26l-seg.pt` | ~75 MB | 可选（更准，慢 2x） | 同上 |
| `models/mobileclip2_b.pt` | ~571 MB | YOLOE 文本编码源 | `python scripts/setup_models.py --fetch` |
| `models/mobileclip2_b.ts` | ~253 MB | YOLOE 运行时实际加载的文本编码器 | 手动 trace（见下） |

### 4.1 MobileCLIP2 trace（.pt → .ts）

```bash
conda activate wrc-demo
PYTHONNOUSERSITE=1 python scripts/setup_models.py --fetch    # 下 .pt
PYTHONNOUSERSITE=1 pip install "open_clip_torch>=2.30"

PYTHONNOUSERSITE=1 python <<'PY'
import torch, open_clip
full = open_clip.create_model_from_pretrained("MobileCLIP2-B", pretrained="models/mobileclip2_b.pt", return_transform=False)
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, t):
        f = self.m(t); return f / f.norm(dim=-1, keepdim=True)
torch.jit.trace(Wrap(full.text).eval(), torch.zeros(2, 77, dtype=torch.long), strict=False).save("models/mobileclip2_b.ts")
PY
```

> 注意：ultralytics 加载 `mobileclip*.ts` 只看当前工作目录。直接跑 python 时在项目根建软链：`ln -sf models/mobileclip2_b.ts mobileclip2_b.ts`（`start_all.sh` 会自动做）。

---

## 5. 手眼标定（真机必做）

标定目的：建立 **像素坐标 ↔ 机器人基坐标系** 的变换。`data/calibration/hand_eye_top.json` 是脱敏参考矩阵，**你的相机必须重新标定**，结果存到 `hand_eye_top.local.json`（gitignore 忽略）。

### 5.1 准备

1. 打印 ArUco 棋盘：4x4 字典、ID=0、边长 0.10 m（仓库根有 `aruco_10cm.png`）。打印后**精确测量**实际边长作为 `--marker-size` 参数
2. 接好相机和 CAN 总线（见 3.5）
3. 列出并绑定相机：

```bash
wrc-calib-top --list                                  # 列出设备索引+序列号
wrc-calib-top --bind top --serial <你的相机序列号>
```

### 5.2 执行标定（顶部相机 ETH）

```bash
# 自动模式（预设姿态自动跑采样点）
wrc-calib-top --marker-size 0.10 --n 30 \
  --out data/calibration/hand_eye_top.local.json

# 手动模式（每次按 ENTER 采样）
wrc-calib-top --marker-size 0.10 --manual --n 30 \
  --out data/calibration/hand_eye_top.local.json
```

内部原理：Pinocchio FK（RS URDF，`joint_signs=[-1]*6`，`ee_frame='gripper_end'`）+ cv2.aruco PnP + 联合 SE(3) LM 优化（Huber + 3*MAD 离群剔除）。

### 5.3 验证

```bash
wrc-calib-top --verify \
  --json data/calibration/hand_eye_top.local.json \
  --known-points "0.30,0.00,0.02;0.25,-0.05,0.02;0.35,0.05,0.02"
# 预期：RMSE < 10 mm
```

> 相机固定位置建议：距机械臂底座约 36cm × 36cm。若大面积抓取/放置 IK 失败，需调整相机位置并重新标定。

---

## 6. LLM 配置

### 6.1 决策树

```
显存 ≥ 8 GB   → 本地模型 local_qwen3_vl（默认，推荐）
有 API key    → 云端 anthropic / openai / minimax
都没有        → mock（仅测试）
```

### 6.2 本地模型（local_qwen3_vl）

vLLM 跑在独立 uv venv（`~/cosmos3-venv`，与主 conda env 不兼容），OpenAI 兼容 API 在 :8080：

```bash
bash scripts/start_cosmos3_server.sh --bg     # 后台启动（首次加载 ~60s）
bash scripts/start_cosmos3_server.sh --status # 健康检查
bash scripts/start_cosmos3_server.sh --stop
```

可用 `VLLM_*` 环境变量覆盖端口/显存比/max-len（如 `VLLM_MAX_LEN=8192`）。

### 6.3 云端 API

两层配置：YAML profile（`configs/llm/*.yaml`）+ 环境变量 key：

| profile | env var | 说明 |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude（vision + tools） |
| `openai` | `OPENAI_API_KEY`（`OPENAI_BASE_URL` 改 endpoint） | 任意 OpenAI 兼容端点 |
| `minimax` | `MINIMAX_API_KEY` | text-only |
| `local_qwen3_vl` | 不需要 | 本地 vLLM :8080 |
| `mock` | 不需要 | 离线测试 |

```bash
export ANTHROPIC_API_KEY=<你的key>     # 仅当前 shell 有效
# 持久化：echo 'export ANTHROPIC_API_KEY=<你的key>' >> ~/.bashrc
```

### 6.4 切换 LLM 的两种方法

```bash
# 方法1：CLI 临时覆盖（不动配置文件）
wrc-demo --llm minimax --task "go to home" --no-view --no-serve

# 方法2：改 configs/demo.yaml
llm:
  profile: local_qwen3_vl    # ← 改这一行
```

### 6.5 添加新 LLM profile

大多数国产 LLM（DeepSeek / Kimi / 通义 / 智谱 / SiliconFlow）都走 OpenAI 兼容协议：

```bash
cp configs/llm/openai.yaml configs/llm/kimi.yaml
```

改 4 个字段：

```yaml
type: openai
model: moonshot-v1-8k
base_url: https://api.moonshot.cn/v1    # 末尾 /v1 不要漏
env_key: MOONSHOT_API_KEY               # env var 的名字，不是 key 本身
supports_vision: false
```

然后 `demo.yaml` 里 `profile: kimi`，`export MOONSHOT_API_KEY=sk-...`，试跑：

```bash
python -m wrc_demo.apps.demo --llm kimi --task "list the objects" --no-view --no-serve
```

---

## 7. 运行 demo

### 7.1 一键启动（推荐）

```bash
# 真机 + 交互 REPL（需 sudo 权限配置好 CAN）
bash scripts/start_all.sh --repl --real

# mock 模式（无硬件验证）
bash scripts/start_all.sh --repl

# 展位/kiosk 仪表盘模式
bash scripts/start_all.sh --dashboard
```

`start_all.sh` 选项：`--real`（orbbec_overhead + rebot_rs 真机）、`--repl`（前台交互）、`--dashboard`（网页 :8090）、`--stop` / `--status`。

启动后终端会输出 dashboard 地址（如 `http://192.168.x.x:8090/`），浏览器打开即可通过对话控制机械臂。REPL 中直接输入任务，空行退出。

### 7.2 命令行直接跑

```bash
conda activate wrc-demo

# 全 mock 离线
wrc-demo --camera mock --arm mock --llm mock --task "pick up the box"

# 真机
wrc-demo --camera orbbec_overhead --arm rebot_rs --llm local_qwen3_vl \
  --task "pick up the toy"

# 交互模式
wrc-demo --camera orbbec_overhead --arm rebot_rs --interactive
```

### 7.3 可用的指令

**Reflex 快路径指令**（不调 LLM，微秒级响应）：

```
pick up X and put in Y    # 完整 pick+place
pick up the <物体>         # 抓取
open the gripper          # 打开夹爪
look around               # 列出相机识别到的物体
go home                   # 回归原点
wave / 挥手打个招呼        # 挥手
```

**LLM 路径**：任意自然语言，LLM 自主调用技能（`pick_and_place` / `grasp_object` / `place_at/on` / `move_t` / `teach_record/replay` / `task_done`）。

任务执行后终端输出报告：

```
=== task report ===
task:    go to home
success: True
path:    reflex (3.08s)
steps:   1
summary: done via reflex path in 3.08s: move_home()
```

### 7.4 其他运行方式

**展位长跑 dashboard**（自动检测 LLM tier：有 ANTHROPIC_API_KEY → anthropic；:8080 可达 → local；否则 mock）：

```bash
python scripts/dashboard_runner.py --cameras orbbec --arm rebot_rs
# 验证任务可用 env WRC_DASHBOARD_TASK 覆盖，默认 "pick and place pink object"
```

**MCP 服务器**（把技能暴露给外部 Agent）：

```bash
wrc-mcp                                    # stdio JSON-RPC
python scripts/setup_agents.py --host claude --write   # 注册到 Claude Code
python scripts/setup_agents.py --host codex --write    # 注册到 Codex
bash scripts/hermes_demo.sh --arm rebot_rs             # Hermes 交互聊天
```

MCP 工具集 = 6 个技能 − `task_done` + 5 个网关工具（`camera_snapshot` / `world_state` / `live_view_url` / `emergency_stop` / `reset_stop`）= 10 个。

**实时查看器**：`wrc-view`（RGB+深度+检测覆盖窗口）。

### 7.5 示教录制/复播（PiPER Mate 主臂）

```bash
python examples/teach_record.py --slot 1          # 主臂示教录制到槽位1（q 录制, Esc 取消）
python examples/teach_record.py --slot 1 --dry-run  # 不连硬件验证逻辑
python examples/teach_replay.py --slot 1          # 复播（结束自动 safe_home）
```

录像存 `data/recordings/slot{N}.json`（schema v2，30 Hz，含 checksum 校验）。

### 7.6 紧急收臂

正常关机后臂处于自由落体危险姿态时：

```bash
sudo python -m scripts.emergency_stow    # 绕过 demo 循环，开到机械零点后断电
```

---

## 8. 自定义配置（configs/demo.yaml）

### 8.1 抓取参数

```yaml
grasp:
  backend: obb               # obb（默认兜底）/ graspgenx（学习型，:5556 服务器挂掉自动回退 obb）
  pregrasp_offset_m: 0.08    # 悬停高度
  depth_fraction: 0.65       # 从物体顶部往下抓的深度比例
  exempt_radius_m: 0.15      # 抓取下降安全圆柱半径
  approach_pitch_rad: 1.2    # 接近角 ~69°（纯 top-down 在 B601-RS 上 IK 不可达）
  drop_zone: [0.18, -0.17]   # 默认放置点（箱子位置）
  max_pick_attempts: 8       # 抓取重试预算
```

### 8.2 手眼补偿

标定矩阵之上的手动平移补偿（微调抓取落点），在 `configs/cameras/orbbec_overhead.yaml`：

```yaml
extrinsics:
  hand_eye_compensation_m:    # 米；默认 [0,0,0] = 不起作用
    x: 0.00
    y: 0.00
    z: 0.00
```

### 8.3 检测器词汇

```yaml
detector:
  model: ${repo}/models/yoloe-26s-seg.pt
  conf: 0.15                 # 越低检出越多，误检也越多

detect_classes: ["box", "banana", "toy"]   # YOLOE 开放词汇检测标签
```

### 8.4 安全阈值

```yaml
safety:
  workspace:                 # TCP 工作空间 AABB（米，基坐标系）
    min: [0.10, -0.30, -0.01]
    max: [0.50, 0.30, 0.55]
  table_z: 0.0               # 桌面高度
  table_clearance: 0.02      # 桌面净空
  max_joint_vel: 1.2         # rad/s
  joint_margin: 0.02         # rad；必须 < IK 的 limit_margin=0.025（不变量，一起调！）
  watchdog_s: 5.0            # 感知过期停止时间
  keep_out: []               # 禁入区 AABB 列表
```

---

## 9. 扩展开发

### 9.1 添加新技能（5 步）

所有技能加在 `src/wrc_demo/skills/runtime.py` 的 `SkillRuntime` 类：

1. 写 `skill_<name>()` 方法（必须用 `self.arm`（SafeArm），不直接拿 raw）；返回 `{"ok": True, ...}` 字典
2. 在 `TOOL_SPECS` 加 JSON Schema 条目
3. 若会动臂，把名字加进 `_MOTION_SKILLS`（这是唯一暂停 WorldWatcher 信念融合的地方——动臂时不暂停，被持物会被错误重融合到空中位置）
4. trace + MCP 暴露自动完成（`SkillRuntime.execute()` 是单一调度入口）
5. 可选：在 `agent/reflex.py` 的 `_RULES` 加正则，让常见指令走 reflex 快路径

### 9.2 添加新检测器

`src/wrc_demo/perception/detector.py` 实现 `detect(frame) -> list[Detection]`，然后在 `apps/demo.py::_make_detector()` 加分支。

### 9.3 添加新抓取规划器

`src/wrc_demo/grasping/` 下实现返回 `list[Grasp]` 的规划函数，在 `SkillRuntime._plan_grasps()` 叠加为新的一层。

---

## 10. 测试

```bash
conda activate wrc-demo
pytest tests/ -q                # 全套 mock（~290 个，hardware 标记自动 deselect）
pytest tests/ -m hardware -q    # 真机只读测试
```

测试覆盖：安全/控制（safety、kinematics、gripper）、抓取（三层规划）、标定/外参、相机/感知、LLM/Agent/技能、MCP 服务器、示教（67 个）、记忆系统、回归（每轮对抗 review 一个文件，一个确认缺陷一个测试）。

---

## 11. 数据与产物目录

| 目录 | 内容 |
|---|---|
| `data/calibration/` | `cameras.json`（相机绑定）、`hand_eye_top.json`（脱敏参考）、`hand_eye_top.local.json`（**本机真机标定结果**，gitignore） |
| `data/recordings/` | 示教槽位 `slot{N}.json` |
| `runs/<时间戳>/` | 每次任务的 `trace.jsonl` + `keyframes/` + `summary.txt` |
| `runs/experience.json` | Tier-2 习惯经验记忆（追求确定性行为时删除） |
| `~/.wrc_demo/grasp_memory.json` | 抓取结果记忆（同上） |
| `Log/OrbbecSDK.log.txt` | Orbbec SDK 原生日志 |

---

## 12. 故障排查速查

| 症状 | 排查 |
|---|---|
| `import torch` 报 `libcudart.so` 找不到 | 跑 `bash scripts/install_nvidia_libs_hook.sh`，开新 shell 验证 |
| YOLOE 报 "Download failure for mobileclip_*.ts" | .ts 文件必须在 CWD 或 `models/`；`ln -sf models/mobileclip2_b.ts mobileclip2_b.ts` |
| pip 包装了但 conda env 里找不到 | 所有 pip 命令加 `PYTHONNOUSERSITE=1` |
| 检测不到物体 | 先 `wrc-demo --llm mock --task "look around"` 确认检测标签；调低 `detector.conf`；检查 `detect_classes` |
| 抓取落点系统性偏移 | 微调 `configs/cameras/orbbec_overhead.yaml` 的 `hand_eye_compensation_m` |
| 大面积 IK 求解失败 | 相机位置不合适（建议距底座 ~36×36 cm），调整并重新标定 |
| can0 不存在 | `sudo modprobe peak_usb`；`ip -br link` 检查；重新 bring-up |
| LLM 不响应 | `bash scripts/start_cosmos3_server.sh --status`；首次加载需 ~60s；云端检查 env var 是否 export |
| 想恢复确定性行为 | 删除 `runs/experience.json` 和 `~/.wrc_demo/grasp_memory.json` |
| 抓取前检查清单 | 物体 < 8 cm 硬质；放在 x=0.30–0.35, y=±0.05 区域；放置需在 drop_zone（x≈0.18, y≈-0.17）放箱子 |

---

## 13. 参考文档

- [Seeed Wiki 官方教程](https://wiki.seeedstudio.com/cn/wrc_demo_tutorial/)
- `README.md` / `CLAUDE.md` / `操作指南.md`（本机实战清单）
- `docs/ARCHITECTURE.md`（架构设计与决策）
- `docs/ROADMAP.md`（待办与真机 bring-up 清单）
- `docs/adr/`（9 篇架构决策记录）
- `docs/SPEC-local-llm.md`（本地 LLM 运维规程）
- `docs/MIGRATION_PLAN.md` / `docs/HARDWARE_VERIFICATION_HANDOVER.md`（控制栈迁移与真机验证记录）

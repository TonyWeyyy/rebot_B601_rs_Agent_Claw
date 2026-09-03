---
description: wrc_demo 操作指南：reBot Arm B601-RS 视觉夹取 demo 的完整步骤——环境配置、模型下载、LLM 切换、手眼标定、demo 运行、故障排查。
title: wrc_demo 操作指南：reBot Arm B601-RS 视觉夹取
keywords:
  - wrc_demo
  - reBot Arm B601-RS
  - 视觉夹取
  - 视觉语言模型
  - Qwen3-VL
  - YOLOE
  - 手眼标定
  - conda
  - 教程
slug: /wrc_demo_tutorial
last_update:
  date: 2026-08-28
  author: Seeed Studio
translation:
  skip:
    - [zh-CN]
createdAt: '2026-06-15'
updatedAt: '2026-08-28'
url: https://wiki.seeedstudio.com/cn/wrc_demo_tutorial/
---

# wrc_demo 操作指南：reBot Arm B601-RS 视觉夹取

<p align="center">
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04+-orange.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/Camera-RGB--D-green.svg" alt="Camera" />
  <img src="https://img.shields.io/badge/Brain-VLM-purple.svg" alt="VLM Brain" />
</p>

<p align="center">
  <strong>环境配置 → 模型下载 → LLM 切换 → 手眼标定 → demo 运行 → 故障排查</strong>
</p>

本教程是 [wrc_demo](https://github.com/TheMoonAstronaut/wrc) 项目的**操作指南**。按本指南步骤操作，**1-2 小时**即可从零克隆到能跑 demo。

wrc_demo 是 [reBot Arm B601-RS](https://wiki.seeedstudio.com/cn/rebot_b601_rs_getting_started/)（RobStride 总线）的视觉夹取 demo，运行在 Ubuntu 22.04 + Python 3.10 + conda `wrc-demo` 环境。Demo 接收自然语言指令（如"pick up the red block"），自动规划抓取并执行。

<p align="center">
  <img src="https://files.seeedstudio.com/wiki/robotics/projects/rebot_arm/visual_grasp/grasp_rs.gif" alt="wrc_demo 在 reBot Arm 上的视觉抓取演示" />
</p>

## 0. 速查表

### 0.1 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04+ / Debian 12+ / WSL2 |
| Python | 3.10（conda env 内）|
| 内存 | ≥ 16 GB（mock 8 GB 起） |
| 磁盘 | ≥ 10 GB |
| GPU（可选）| NVIDIA RTX 5070 / 4090 / H100 等 |

### 0.2 命令速查

| 任务 | 命令 |
|------|------|
| **激活环境** | `conda activate wrc-demo` |
| **跑 mock 测试** | `pytest tests/test_extrinsics_loader.py tests/test_hand_eye_compensation_per_camera.py tests/test_demo_yaml_llm_profile.py -q` |
| **跑完整 preflight** | `pytest tests/test_phase6_preflight.py -v` |
| **跑 mock demo** | `wrc-demo --task "wave" --no-view --no-serve` |
| **切换 LLM** | CLI：`--llm <profile>`；YAML：编辑 `configs/demo.yaml` 的 `llm.profile` |
| **启动本地 LLM** | `bash scripts/start_cosmos3_server.sh --bg` |
| **顶部相机标定** | `wrc-calib-top --n 30 --out data/calibration/hand_eye_top.local.json` |
| **下载模型权重** | `python scripts/setup_models.py --fetch` |

---

## 1. 项目目标与读者

- **项目目标**：把视觉语言模型（VLM）接到 reBot Arm B601-RS 上，让自然语言指令自动变抓取动作——VLM 选物体、Pinocchio 反解 IK、`SafetyHarness` 每个 50 Hz waypoint fail-closed 校验。
- **适用读者**：有 NVIDIA GPU + Ubuntu 22.04 + Python 3.10 基础，想跑 demo / 改 detector / 加 skill 的开发者。
- **阅读路径**：
  - **5 分钟**：直接看 §3.0（mock 测试 + mock demo）→ 验证 conda env 能跑就够。
  - **1-2 小时**：按 §3 → §4 → §5 → §6 → §7 顺序读，能从零到真机抓取。
  - **改 detector / 加 skill**：先读完 §9，再看 §10.3 关键设计决策。
- **本文档不覆盖**：
  - RobStride 电机底层协议（详见 vendored SDK `wrc_demo/sdk/reBotArm_control_py/`）
  - Pinocchio 数值算法（详见上游 docs）
  - YOLOE 模型训练（详见 [ultralytics 官方文档](https://docs.ultralytics.com/models/yoloe/)）

---

## 2. 项目概览

### 2.1 项目特点

1. **视觉语言模型驱动，三级决策**  
   默认采用 Qwen3-VL-2B-Instruct-AWQ-4bit 作为任务规划脑，可在 8 GB 显存的设备上原生运行。命令解析走 **Reflex → Habit → LLM** 三级快慢路径——常见指令（"pick up the red block"）直接走 Reflex 模板语法，不调用 LLM；只有全新任务才让 VLM 介入。

2. **手眼标定 + 6-DoF 抓取规划（fail-closed）**  
   自带手眼标定脚本（ArUco 棋盘格 + Pinocchio FK + 联合 SE(3) LM 优化），支持顶部（ETH）相机布局。每个流式轨迹点都经过 `SafetyHarness` 校验——任一校验失败，机械臂立即中途停止。

3. **6 个 TOOL_SPECS + 5 个 MCP 网关**  
   顶层暴露 `pick_and_place` / `grasp_object` / `place_at/on` / `move_t` / `teach_record/replay` + `task_done` 共 6 个 safety-gated 工具；MCP 网关额外暴露 `camera_snapshot` / `world_state` / `live_view_url` / `emergency_stop` / `reset_stop` 共 5 个工具。加新技能零协议适配工作。

4. **完整的 trace + episodic 记忆**  
   每个技能调用都写入 ASPIRE 风格的多模态 trace（`trace.jsonl` + 前后关键帧 JPEG），15 秒事件窗口 + 物体持久性信念，让"10 秒前看到的杯子"在遮挡后仍可被操作。

5. **Mock 优先的开发体验**  
   全栈 mock（`--camera mock --arm mock --llm mock`）无需任何硬件即可跑通完整决策流水线。mock 测试和真机走的是**同一份安全代码**。

### 2.2 规格参数

| 类别 | 参数 | 规格 |
|------|------|------|
| **硬件** | 机械臂 | reBot Arm B601-RS（RobStride 总线，6-DOF + 夹爪） |
| | 相机 | Orbbec Gemini 2 / Intel RealSense D435i / D405 |
| | 通信 | CAN Bus via USB2CAN；USB 3.0 相机 |
| | 控制主机 | Ubuntu 22.04+ PC / DGX Spark |
| **本地 LLM** | 默认 profile | `local_qwen3_vl`（Qwen3-VL-2B-Instruct-AWQ-4bit）|
| | 推理框架 | vLLM 0.6+（多模态推理）|
| | 显存需求 | ≥ 8 GB（约 30 tokens/s）|
| | 权重大小 | ~2.4 GB |
| **云端 LLM** | Anthropic | Claude（vision + tools；`ANTHROPIC_API_KEY` env）|
| | OpenAI 兼容 | 任意 OpenAI-compat endpoint（`OPENAI_API_KEY` env）|
| | MiniMax | `api.minimax.com`（`MINIMAX_API_KEY` env；text-only）|
| **检测器** | 默认 | YOLOE-11s-seg（开放词汇检测 + 实例分割）|
| | 可选 | YOLOE-26l-seg（更准但 2× 慢）|
| | 文本编码 | MobileCLIP2-B torchscript（Apple `apple/MobileCLIP2-B` → 自行 trace）|
| **软件栈** | 运动学 | Pinocchio 3.x（FK/IK，DLS + 随机重启）|
| | RobStride 驱动 | motorbridge（SocketCAN）+ vendored SDK |
| | 抓取后端 | camera-camera / GraspGen-X（ZMQ）/ analytic OBB（三层）|
| | Python | 3.10 |
| **技能系统** | TOOL_SPECS 总数 | 6（含 1 个 `task_done` 终止符 + 5 个真技能）|
| | 动臂技能 | 18 个 string literal 在 `_MOTION_SKILLS` |
| | MCP 工具集 | `TOOL_SPECS - _EXCLUDED_TOOLS + _EXTRA_TOOLS` = 6 - 1 + 5 = 10 |
| **安全** | 轨迹检查 | 每个 50 Hz 流式 waypoint 都过 `SafetyHarness.approve()` |
| | 失败模式 | fail-closed（任一违越 → 中途 abort）|
| | 关节预留 | IK `limit_margin=0.025` > harness `joint_margin=0.02`（不变量）|

### 2.3 材料清单（BOM）

| 部件 | 数量 | 是否包含 |
|------|------|----------|
| reBot Arm B601-RS 机械臂 | 1 | ✅ |
| 夹爪（已装） | 1 | ✅ |
| USB2CAN 串口桥 | 1 | ✅ |
| 48V 电源适配器 | 1 | ✅ |
| USB-C / 通信线缆 | 1 | ✅ |
| RGB-D 深度相机（Orbbec Gemini 2 推荐）| 1 | ✅ |
| 摄像头连接件 / 安装支架 | 1 | ✅ |
| Ubuntu 22.04+ 控制主机（NVIDIA GPU ≥ 8 GB）| 1 | ❌ 自备 |

#### 接线说明

1. **机械臂**：48V 电源 → 机械臂；USB2CAN → 主机 USB 口（首次使用需 `sudo ip link set can0 up type can bitrate 1000000`）
2. **相机**：Orbbec USB 3.0 → 主机 USB 3.0 口
3. **权限配置**：

```bash
# 一次性赋予 USB 设备读写权限
sudo chmod a+rw /dev/bus/usb/*/*

# Orbbec udev 规则（首次安装推荐）
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

```bash# 套件里是 PCAN-USB，通常应该直接出现 can0 或 can1
sudo modprobe peak_usb
ip -br link

# 如果出现 can0，再设置 bitrate
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

```

### 2.4 环境要求

#### 硬件

| 项目 | 要求 |
|------|------|
| CPU | x86_64（aarch64 测试未覆盖） |
| 内存 | ≥ 16 GB（mock 8 GB 起，本地 LLM 16 GB 推荐） |
| 磁盘 | ≥ 10 GB（模型权重 + conda env + trace 累积）|
| GPU（本地 LLM 推理）| NVIDIA RTX 5070 / 4090 / H100 |

#### 软件

| 项目 | 要求 |
|------|------|
| Python | 3.10（conda env 内）|
| Conda | miniforge3 / miniconda / anaconda |
| 系统包 | `libusb-1.0-0-dev`, `ffmpeg`, `git`, `can-utils` |

#### 网络

| 资源 | 说明 |
|------|------|
| GitHub | 克隆 `github.com/TheMoonAstronaut/wrc` |
| HuggingFace | 下载 YOLOE / MobileCLIP2 / Qwen3-VL 权重 |
| PyPI | 装 ultralytics / openai / anthropic / pin / motorbridge |

---

## 3. 环境配置（核心 1）

### 3.0 最小路径：5 分钟跑通 mock

如果你只想跑通 mock 验证安装（不读其它内容），三条命令搞定：

```bash
git clone https://github.com/TheMoonAstronaut/wrc.git
cd wrc
conda env create -f environment.yml && conda activate wrc-demo

# 跑通 mock 测试（17 passed, 1 skipped ≈ 30 秒）
pytest tests/test_extrinsics_loader.py \
        tests/test_hand_eye_compensation_per_camera.py \
        tests/test_demo_yaml_llm_profile.py -q
```

:::tip
环境名固定为 `wrc-demo`。如需自定义名字（如团队命名约定），请同时改 `environment.yml` 的 `name:` 字段并相应替换后续命令。
:::

:::tip
`pyorbbecsdk2` 不在 environment.yml 里** —— 它的依赖链 `pyorbbecsdk2 → open3d==0.18.0 → dash → plotly → ipywidgets → ...` 让 pip 经常触发 `resolution-too-deep`。Orbbec SDK 单独装。
:::
:::tip
 `torch` 也不在 environment.yml 里** —— ultralytics 把 torch 标为 optional，但 `import ultralytics` 会触发 `import torch`。cuda env create 的 pip 子进程看到 user site-packages 里有 torch 也会"已满足"跳过，结果 conda env 内没有 torch。torch 单独装在 §3.6。
:::

### 3.1 前置条件

- 已完成 [reBot Arm B601-RS 快速入门](https://wiki.seeedstudio.com/cn/rebot_b601_rs_getting_started/)（机械臂组装、零点初始化、电机 ID 配置）—— **只真机需要**
- 已具备 NVIDIA GPU + CUDA 驱动（**仅本地 LLM 推理需要**）
- 已安装 [miniforge3](https://conda-forge.org/miniforge/) 或同类 conda 工具
- Ubuntu 22.04+ / Debian 12+ / WSL2

### 3.2 系统依赖（真机需要）

```bash
sudo apt update
sudo apt install -y libusb-1.0-0-dev ffmpeg git can-utils
```

- `libusb-1.0-0-dev`：Orbbec 深度相机
- `ffmpeg`：UVC4K / RealSense RGB 流
- `can-utils`：SocketCAN bring-up（机械臂）

### 3.3 post-install：单独装 PyTorch（CUDA 版本）

 **📌 工作目录**：从本节起，所有命令都假设你在 `wrc` 仓库根目录里。如果开了新 shell，先 `cd wrc`。

```bash
# 你现在应该已经 cd wrc 了；如果不是：
cd /path/to/wrc
```

```bash
conda activate wrc-demo
```

#### 3.3.1 安装 Pytorch 和 Torchvision

根据自身机型的CUDA版本，按照[官网教程](https://pytorch.org/index.html) 安装 Pytorch 和 Torchvision

:::tip
如果需要安装stable CUDA 12.8，请执行以下命令

```bash
PYTHONNOUSERSITE=1 pip install --force-reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```
:::

#### 3.3.2 在conda环境下加载 PyTorch

PyTorch 的 cu128/cu130 wheel 把 nvidia 包装到 site-packages/nvidia/lib/，但 conda 不会自动把这些路径加到 LD_LIBRARY_PATH。不装这个 hook，任何新 shell 里 `import torch` 都会报 `OSError: libcudart.so.13: cannot open shared object file`

运行下面的命令安装`nvidia LD_LIBRARY_PATH conda hook`
```bash
bash scripts/install_nvidia_libs_hook.sh

```

验证 hook 生效（新 shell 也能用）
```bash
source /home/seeed/miniforge3/envs/wrc-demo/etc/conda/activate.d/nvidia_libs.sh
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
# 预期：torch: 2.13.0+cu130 cuda: True
```

### 3.4 post-install：单独装深度相机 SDK

本项目支持 Orbbec Gemini 2 与 RealSense D435i / D405 等 RGB-D 深度相机。

#### Orbbec Gemini 2

Orbbec Gemini 2 依赖 pyorbbecsdk（Orbbec SDK v2 的 Python 版本）。推荐**方式一**：

**方式一：通过 pip 安装（推荐）**

```bash
conda activate wrc-demo
PYTHONNOUSERSITE=1 pip install pyorbbecsdk2
```

:::tip
必须加 `PYTHONNOUSERSITE=1`，否则 pip 把包装到 `~/.local/lib/python3.10/site-packages/`（用户级 site-packages），conda env 内还是空的。
:::

**方式二：从 GitHub 源码安装**（网络问题或想要最新版时）

```bash
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

# 切到 ~/sdk 目录（不是 wrc 目录）
mkdir -p ~/sdk && cd ~/sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
PYTHONNOUSERSITE=1 pip install -e .

# 装完回到 wrc 目录
cd -
```

> **中国大陆用户**可以用 gitee 镜像加速：`git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git`

#### RealSense D435i / D405

RealSense SDK（`pyrealsense2`）通常需要从源码编译，建议参考 Intel 官方文档。

三条都跑通 → 环境就绪，下一步 §4（模型权重）或 §6（手眼标定）。失败排查见 §3.8。
---

## 4. 模型权重获取

### 4.1 安装MobileCLIP2 文本编码器

Apple 在 HuggingFace `apple/MobileCLIP2-B` 发布 PyTorch 权重，YOLOE 文本 prompt 检测需要 `.ts` 版本。

```bash
conda activate wrc-demo

# 下载 MobileCLIP2 .pt
PYTHONNOUSERSITE=1 python scripts/setup_models.py --fetch
```
转换格式 trace `.pt → .ts`（手动）**

```bash
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
手动验证

```bash
PYTHONNOUSERSITE=1 python -c "import torch; m=torch.jit.load('models/mobileclip2_b.ts'); o=m(torch.zeros(2,77,dtype=torch.long)); print(o.shape, o.norm(dim=-1).tolist())"
# 预期：torch.Size([2, 512]) [1.0, 1.0]
```

### 4.1.1 ℹ️ 关于 YOLOE 文本编码器的查找路径

ultralytics 加载 `mobileclip*.ts` **只看当前工作目录（CWD），不会去 `models/` 找**。

如果直接 `python -m wrc_demo.apps.demo ...`，CWD 不在 `models/` 下就会报 "Download failure for ... mobileclip_*.ts"。

**好消息**：`scripts/start_all.sh`（以及任何用它的 wrapper）在启动前会自动把 `models/*.ts` 软链接到 CWD，**你不需要手动建 symlink**。这一段只在你绕过 wrapper 直接跑 python 时才需要管：

```bash
# 直接跑 python 时的两种处理方式（任选）
ln -sf models/mobileclip2_b.ts mobileclip2_b.ts     # A. 项目根 symlink
cd models && python -m wrc_demo.apps.demo ...      # B. 在 models/ 下跑
```

> 2026-09-03 session 2 补丁：`start_all.sh` 增加 `Step 0` 自动 symlink（见 §4.3 速查表的"启动器"行）。

### 4.2 下载Qwen3-VL 本地 LLM 权重

如果您打算接入云端API，可以跳过此步骤：

本教程默认 llm.profile=local_qwen3_vl 对应 AWQ-4bit 量化版（8GB 显存跑得动）。
```bash
hf download cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit \
  --local-dir models/qwen3-vl-2b-awq-4bit
# → https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit
```
:::tip
如果使用hf下载存在网络问题可以使用github仓库下载
```bash
git clone https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit models/qwen3-vl-2b-awq-4bit
#如果卡顿太久直接按ctrl + c中止即可
cd models/qwen3-vl-2b-awq-4bit && git lfs pull
```
:::

### 4.3 速查表

| 资产 | 大小 | 必要性 | 获取方式 |
|------|------|--------|----------|
| `qwen3-vl-2b-awq-4bit/` | ~2.4 GB | 本地 LLM 必装（8GB 显存）| `hf download cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit` 或 `scripts/start_cosmos3_server.sh --bg` 自动拉 |
| `yoloe-26s-seg.pt` | ~31 MB | **开放词汇检测（推荐 2026-09+）** | git clone Seeed-Projects/reBot-DevArm-Grasp 后 `cp` 出来；用 `mobileclip2_b.ts`（已有），不需 `blt.ts` |
| `yoloe-26l-seg.pt` | ~75 MB | 可选（更准，慢 ~2x）| 同上；也用 `mobileclip2_b.ts` |
| `yoloe-11s-seg.pt` | ~28 MB | **不推荐**——需 `mobileclip_blt.ts`（572 MB，首次启动要下 1.5h） | 仅在需要 legacy 模型权重时用；切回需要把 `configs/demo.yaml` 的 `model:` 改回 |
| `mobileclip2_b.pt` | ~571 MB | YOLOE 26s/l 文本编码源（PyTorch 原版）| `python scripts/setup_models.py --fetch` 自动从 `apple/MobileCLIP2-B` 拉 |
| `mobileclip2_b.ts` | ~253 MB | YOLOE 26s/l 文本编码（运行时实际加载）| 手动 trace（见 §4.1）；已下载后**必须放在 `models/` 或 CWD**——见 §4.1.1 |
| `mobileclip_blt.ts` | ~572 MB | YOLOE 11s 文本编码（**不推荐**，来源难下）| 手动 trace，源 `apple/MobileCLIP-B-LT`；只在用 11s 时才需要 |
| `pyrealsense2` | 编译产物 | 仅 RealSense 用户 | Intel 官方文档编译 |

> **2026-09 变更**：`configs/demo.yaml` 默认 `model:` 从 `yoloe-11s-seg.pt` 切到 `yoloe-26s-seg.pt`（session 2 改动，待 commit）。如果你已经有仓库 checkout + models/，**不需要任何重新下载**——只是改了一行 demo.yaml。

---

## 5. 手眼标定

标定的目的是让 demo 知道**像素坐标** ↔ **机器人基坐标系** 的变换。`data/clip`/hand_eye_top.json` 是脱敏参考矩阵，**你的相机必须重新标定**。

### 5.1 准备

#### 1. 打印 ArUco 棋盘

打印一张 Aruco 4x4 字典、ID = 0、边长 = 0.10 m（10 cm）的棋盘格。推荐来源：~/wrc/aruco100x100.pdf

打印后**精确测量**边长（打印机会缩放），把测量值用作 `--marker-size` 参数。

#### 2. 相机连接 + CAN bus

```bash
# Orbbec 插 USB 3.0，赋予相机权限
sudo chmod a+rw /dev/bus/usb/*/*

# CAN bus（机械臂）
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

#### 3. 列出设备

```bash
wrc-calib-top --list
# 预期：列出 Orbbec 设备索引 + 序列号

#写入相机序列号
wrc-calib-top --bind top --serial `更改为扫描到的序列号`
```

### 5.2 顶部相机 ETH 标定

**ETH = Eye-To-Hand**：相机装在机械臂外的固定位置。

```bash
# 自动模式（按预设姿态自动跑 25+ 个采样点）
wrc-calib-top --marker-size 0.10 --n 30 \
  --out data/calibration/hand_eye_top.local.json

# 手动模式（每次按 ENTER 采样）
wrc-calib-top --marker-size 0.10 --manual --n 30 \
  --out data/calibration/hand_eye_top.local.json
```

`hand_eye_top.local.json` 文件**自动被 .gitignore 忽略**（不会污染仓库），是真机标定结果。

#### 标定原理

工具内部跑 `joint_solve`（SE(3) LM + Huber + 3*MAD outlier rejection）：
- FK 来自 `wrc_demo.control.kinematics.Kinematics`（Pinocchio on RS URDF，`joint_signs=[-1]*6`, `ee_frame='gripper_end'`）
- PnP 来自 cv2.aruco + SDK intrinsics

输出：

- `data/calibration/hand_eye_top.json` — 脱敏参考矩阵（**不要用**，serial=REDACTED）
- `data/calibration/hand_eye_top.local.json` — **你的真机标定结果**

### 5.3 验证标定 RMSE

不重标定，直接用已知基坐标系点验证：

```bash
wrc-calib-top --verify \
  --json data/calibration/hand_eye_top.local.json \
  --known-points "0.30,0.00,0.02;0.25,-0.05,0.02;0.35,0.05,0.02"
# 预期：RMSE < 10 mm（参考 set.py:320）
```

---


## 6. LLM 配置（核心 3）

### 6.1 决策树

```
你的显存？
├─ ≥ 8 GB  → 选项 A：本地模型（local_qwen3_vl）
├─ 有 API key → 选项 B：云端 API（anthropic / openai / minimax）
└─ 都没有 → 选项 C：mock（仅测试）
```

### 6.2 选项 A：本地模型（推荐，8 GB+ 显存）

启动 vLLM 服务（**另一个终端**）：

```bash
conda activate wrc-demo
bash scripts/start_cosmos3_server.sh --bg    # 后台启动
sleep 60                                    # 等 vLLM 加载模型（首次需要 ~60s）

# 健康检查
bash scripts/start_cosmos3_server.sh --status
# 预期：[status] vllm running, PID <xxx>

```
vLLM 0.6+ 已支持 Qwen3-VL 的多模态推理。AWQ-4bit 量化版本在 8 GB 显存上可跑到 ~30 tokens/s。

### 6.3 选项 B：云端 API（无需 GPU）

**两层配置**——YAML profile 文件 + 环境变量 key：

#### a. YAML profile 文件（`configs/llm/*.yaml`）

| 文件 | 用途 | 关键字段 |
|------|------|---------|
| `anthropic.yaml` | Claude | `type: anthropic`、`model: claude-sonnet-5` |
| `openai.yaml` | OpenAI / OpenAI 兼容 | `type: openai_compat`、`base_url`、`model` |
| `minimax.yaml` | MiniMax | `type: openai_compat` + 固定 `base_url: https://api.minimax.chat/v1` + `env_key: MINIMAX_API_KEY` |
| `local_qwen3_vl.yaml` | 本地 vLLM（默认）| 不需要 key |
| `mock.yaml` | scripted mock | 测试用 |

#### b. 选哪个 profile（`configs/demo.yaml`）

```yaml
llm:
  profile: local_qwen3_vl    # ← 改这一行（mock / anthropic / openai / minimax / local_qwen3_vl）
```

CLI 临时切：`wrc-demo --llm anthropic --task "..."`

#### c. API key（环境变量）

| profile | 读哪个 env var |
|---------|---------------|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY`（`OPENAI_BASE_URL` 改 endpoint）|
| `minimax` | `MINIMAX_API_KEY`（从 YAML 的 `env_key:` 字段读）|
| `local_qwen3_vl` | 不需要 |

#### 跑起来

```bash
conda activate wrc-demo

# ⚠ 重要：下面的 `PUT_KEY_HERE` 是**占位符**，必须替换成你的真 key
#   替换方法：双引键里直接把 PUT_KEY_HERE 删掉、粘上真 key
#   不要保留任何尖括号、引号、$、空格 — bash 会把它们当字面值

# Claude（Anthropic key 格式: sk-ant-api03-...）
export ANTHROPIC_API_KEY=PUT_KEY_HERE

# OpenAI（或任何 OpenAI 兼容 endpoint；key 格式: sk-...）
export OPENAI_API_KEY=PUT_KEY_HERE
# 可选：换 endpoint（Azure / Together / 自建 vLLM 等）
# export OPENAI_BASE_URL=https://api.example.com/v1

# MiniMax（key 格式: sk-cp-...）
export MINIMAX_API_KEY=PUT_KEY_HERE

# 验证env var 真的设上了（看到你的 key 前缀，不是 PUT_KEY_HERE）
echo "ANTHROPIC_API_KEY 前缀: ${ANTHROPIC_API_KEY:0:10}..."
echo "MINIMAX_API_KEY 前缀: ${MINIMAX_API_KEY:0:10}..."
```
使用`export`配置的云端`API_KEY`只有在当前终端有效，如果想要`key`持久化到 `~/.bashrc`**（只 export 当前 shell 有效，新 shell 要重设）：
```bash
# 把 PUT_KEY_HERE 替换成你的真 key（一整行）
echo 'export ANTHROPIC_API_KEY=PUT_KEY_HERE' >> ~/.bashrc

# 立刻重载验证 bashrc 语法没坏（不会暴露 key 内容）
bash -n ~/.bashrc && echo "✓ bashrc 语法 OK"

source ~/.bashrc
# 验证 env var 真设上了
echo "ANTHROPIC_API_KEY 前缀: ${ANTHROPIC_API_KEY:0:10}..."
```

**或者能够在 YAML 里能不能直接写 key（但不推荐）**：

```yaml
# configs/llm/minimax.yaml 有两种写法：

# 写法 A（推荐，key 在 env var 里）
type: openai_compat
base_url: https://api.minimax.chat/v1
model: MiniMax-M3
env_key: MINIMAX_API_KEY          # ← 这里是 "env var 的名字"，不是值

# 写法 B（key 直接写 YAML，但会进 git）
type: openai_compat
base_url: https://api.minimax.chat/v1
model: MiniMax-M3
api_key: PUT_KEY_HERE             # ← 真 key 写这里（**不推荐**，会被 git 跟踪）
```


### 6.4 切换 LLM 的 2 种方法

#### 方法 1：CLI 临时覆盖

```bash
# 单次跑用云端，tip：minimax可以替换为自己实际使用的云端模型
wrc-demo --llm minimax --task "go to home" --no-view --no-serve


# 单次跑用本地模型（覆盖 demo.yaml 的 default）
wrc-demo --llm local_qwen3_vl --task "go to home" --no-view --no-serve
```
:::tip
上面两个测试脚本只是用来测试模型的链路是否连通，并不会实际控制机械臂
:::
---

## 7. demo 运行（核心 5）

### 7.1 启动 demo（CLI 模式）

```bash
# 启动项目
bash /home/seeed/wrc/scripts/start_all.sh --repl --real

#启动网页，例如：http://192.168.3.175:8090/ 打开控制界面，即可通过对话完成机械臂控制

```
```
# 输出示例：
(wrc-demo) seeed@seeed-KUANGSHI-Series:~/wrc$ bash /home/seeed/wrc/scripts/start_all.sh --repl --real
============================================================
  wrc_demo + Qwen3-VL-AWQ-4bit one-shot launcher
============================================================
  mode:    real hardware
  camera:  orbbec_overhead
  arm:     rebot_rs
  task:    <interactive>

[1/3] check vLLM...
  vLLM already running (skipped)

[2/3] start wrc_demo...
[wrc-demo] cameras=['orbbec_overhead'] arm=rebot_rs llm=local_qwen3_vl view=False
[wrc-demo] traces -> /home/seeed/wrc/runs/20260903_153216
load extensions from /home/seeed/miniforge3/envs/wrc-demo/lib/python3.10/site-packages/pyorbbecsdk/extensions
[stream] live view on http://192.168.3.175:8090/
[wrc-demo] LIVESTREAM dashboard: http://192.168.3.175:8090/
Type a task (empty line to quit).
task> 
=== task report ===
task:    go to home
success: True
path:    reflex (3.08s)
steps:   1
summary: done via reflex path in 3.08s: move_home()

[✓ done] task complete — arm parked at home. Enter a new task or press Enter to quit.

=== task report ===
task:    挥手打个招呼吧
success: True
path:    llm (14.66s)
steps:   3
summary: Robot waved hello three times as requested.

```

### 7.2.5 完整启动命令（vLLM + wrc-demo 一体化）

如果想**手动**启动整套（vLLM + wrc-demo + dashboard）而不是用 §7.2 的简化封装，**完整命令**长这样：

#### Mock 全栈（无 sudo，无真机）

```bash
# 1. 启动 vLLM（后台）
bash scripts/start_cosmos3_server.sh --bg

# 2. 启动 wrc-demo（前台，看实时输出）
PYTHONNOUSERSITE=1 \
HOME=/home/seeed \
YOLO_OFFLINE=True \
WRC_RUN_DIR=/home/seeed/wrc/runs/live_$(date +%Y%m%d_%H%M%S) \
WRC_STREAM=1 \
/home/seeed/miniforge3/envs/wrc-demo/bin/python -u -m wrc_demo.apps.demo \
  --cameras mock --arm mock \
  --llm local_qwen3_vl \
  --interactive --no-view
```

#### 真机全栈（需要 sudo，跑 CAN bus）

```bash
# 0. 一次性：缓存 sudo 凭据（sudo -v 在交互终端跑一次）
sudo -v

# 1. 启动 vLLM（后台）
bash scripts/start_cosmos3_server.sh --bg

# 2. 启动 wrc-demo（前台，sudo 拿 CAN bus + USB 设备）
sudo -E env \
HOME=/home/seeed \
PYTHONNOUSERSITE=1 \
YOLO_OFFLINE=True \
WRC_RUN_DIR=/home/seeed/wrc/runs/live_$(date +%Y%m%d_%H%M%S) \
WRC_STREAM=1 \
/home/seeed/miniforge3/envs/wrc-demo/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs \
  --llm local_qwen3_vl \
  --interactive --no-view
```

#### 一行封装（推荐）

```bash
bash scripts/start_all.sh            # mock 全栈
bash scripts/start_all.sh --real     # 真机全栈（需要 sudo）
bash scripts/start_all.sh --stop     # 关掉所有（vLLM + wrc-demo）
```

#### 关键 env var（完整列表看 `src/wrc_demo/apps/mcp_server.py:14-29`）

| 变量 | 作用 | 默认 |
|------|------|------|
| `HOME` | ultralytics / YOLO 缓存路径 | `/root`（可能没 YOLO 缓存权限） |
| `PYTHONNOUSERSITE=1` | conda env 不读 user site-packages | — |
| `YOLO_OFFLINE=True` | ultralytics 不联网下载（GFW后也避免 hang）| `False` |
| `WRC_RUN_DIR` | trace/keyframes 输出目录 | `<repo>/runs/<timestamp>` |
| `WRC_STREAM=1` | 启用 MJPEG livestream dashboard | `0` |
| `WRC_STREAM_PORT` | dashboard 端口（默认从 `configs/demo.yaml` 读）| `8090` |
| `WRC_VIEW=1` | 启用开开真 OpenCV 窗口（需要 DISPLAY）| 有 DISPLAY 自动开 |
| `WRC_PREWARM=0` | 禁用 perception 预热（cold start 调试用）| `1` |
| `WRC_DETECT_CLASSES` | YOLOE 默认词汇（逗号分隔）| 从 `configs/demo.yaml` |
| `WRC_DETECTOR_MODEL` | 覆盖 detector 权重路径 | 从 `configs/demo.yaml` |

Dashboard URL：`http://<your-ip>:8090`

### 7.3 启动 demo（Dashboard 长跑模式）

适合展位 / 演示场景——跑一次脚本化 pick + 持续服务 MJPEG dashboard：

```bash
python scripts/dashboard_runner.py
# 或自定义初始任务
WRC_DASHBOARD_TASK="wave" python scripts/dashboard_runner.py
```

Dashboard URL：`http://<your-ip>:8090`（默认 `0.0.0.0:8090`）

### 7.4 demo 不工作排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `LLMConfigError: ANTHROPIC_API_KEY not set` | 云端 API 没设 env var | `export ANTHROPIC_API_KEY=...` |
| demo 卡住 30s+ 不响应 | 选择了 cloud LLM 但网络慢 | 切到 `--llm mock` 验证 |
| `CUDA OOM` | AWQ-4bit + detector 抢显存 | 切 `--device cpu` 在 `configs/demo.yaml` |
| 物体识别不到 | `mobileclip2_b.ts` 不在 `models/` / ultralytics 的 `clip` fork 没装 | 见 §4.2 步骤 B |
| `bwrap panic`（Codex CLI 用户）| Codex 沙箱 + 中文 basename | 详见 §11.5 |
| YOLOE 没有任何检测 | 详见 FAQ `mobileclip2_b.ts` 路径 | 从 `models/` 目录启动；`MobileCLIPTS` 传绝对路径（§4.2 步骤 B）|

---

## 8. 自定义配置

### 8.1 加新 LLM profile

最简单的一步——复制 yml 改字段：

```bash
# 1. 复制
cp configs/llm/anthropic.yaml configs/llm/my_llm.yaml

# 2. 编辑 configs/llm/my_llm.yaml
```

```yaml
type: openai             # openai / anthropic / mock
model: my-model-name
base_url: https://api.example.com/v1
env_key: MY_LLM_API_KEY
supports_vision: true
temperature: 0.0
max_tokens: 1024
timeout_s: 30
```

```bash
# 3. 在 demo.yaml 设默认
vim configs/demo.yaml
# llm:
#   profile: my_llm        ← 改这一行

# 4. 设 env var + 重启 demo
export MY_LLM_API_KEY=...
wrc-demo --task "..."
```

### 8.2 改 grasp 精度

编辑 `configs/demo.yaml` 的 `grasp.*` block：

```yaml
grasp:
  backend: obb                          # obb (默认 fallback) / graspgenx (学习型)
  pregrasp_offset_m: 0.08              # hover 高度（沿 -approach 方向）
  insertion_depth_m: 0.0              # TCP 沿 +tool-x 偏移
  depth_fraction: 0.35                 # 从物体 TOP 往下抓多深
  exempt_radius_m: 0.15                # 抓取安全圆柱半径
  move_duration_s: 2.5
  descend_duration_s: 2.0
  release_height_m: 0.05
  air_grasp_frac: 0.04                # 夹爪开度低于此 = 空抓
  approach_pitch_rad: 1.2             # approach 方向与 +z 的夹角（~69°）
```

更保守（更精确、更慢）→ 调高 `n_samples`（vLLM 思考多）；更激进（更快）→ 调低。

### 8.3 改 detector 词汇

```yaml
# configs/demo.yaml
detector:
  type: yoloe                          # yoloe / yolo-world
  model: ${repo}/models/yoloe-11s-seg.pt
  conf: 0.20                           # 置信度阈值（越低 = 越多 detection，但越多误检）

detect_classes: ["banana", "cracker box", "soup can", "cube", "box", "pen", "toy"]
# ↑ 这列表告诉 YOLOE 在开放词汇模式下去找哪些标签
```

或者运行时通过 env var 覆盖：

```bash
WRC_DETECT_CLASSES="banana,apple,orange" wrc-demo --task "..."
```

### 8.4 改安全阈值

> ⚠ **不变量**：`Kinematics.ik` 的 `limit_margin=0.025` 必须**严格大于** harness 的 `joint_margin=0.02`——任一边单独调都会破坏 IK 和安全校验的边界。**两个一起调**！

```yaml
# configs/demo.yaml
safety:
  workspace:                          # TCP 工作空间 AABB（米）
    min: [0.10, -0.30, -0.01]
    max: [0.50, 0.30, 0.55]
  table_z: 0.0                        # 桌面高度（基坐标系）
  table_clearance: 0.02              # TCP 桌面净空
  max_joint_vel: 1.2                  # rad/s（每 waypoint 强制）
  joint_margin: 0.02                  # rad（URDF 限位内预留）
  watchdog_s: 5.0                     # 感知过期停止时间
  keep_out: []                        # e-stop AABB 列表
```

---

## 9. 加新技能

> **前提**：熟悉 Python + Pinocchio + ultralytics。本节是 30 分钟 quick start；要理解底层设计看 `src/wrc_demo/skills/runtime.py`。

### 9.1 加新 skill 的 5 步

所有 skill 加到 `src/wrc_demo/skills/runtime.py` 的 `SkillRuntime` 类（不需要新建 .py）。

#### 步骤 1：写 `skill_<name>()` 方法

```python
# src/wrc_demo/skills/runtime.py
from ..types import SkillError, SafetyViolation

class SkillRuntime:
    # ... 已有方法 ...

    def skill_my_skill(self, foo: str, bar: int = 0) -> dict:
        """一句话描述。Returns: {...}"""
        if not foo:
            raise SkillError("foo must be non-empty")
        # ... 你的逻辑（必须用 self.arm（SafeArm），不直接拿 self.arm.raw）...
        return {"ok": True, "result": "..."}
```

#### 步骤 2：加 `TOOL_SPECS` 条目

```python
# src/wrc_demo/skills/runtime.py
TOOL_SPECS.append({
    "name": "my_skill",
    "description": "...",
    "parameters": {
        "type": "object",
        "properties": {
            "foo": {"type": "string"},
            "bar": {"type": "integer", "default": 0},
        },
        "required": ["foo"],
    },
})
```

#### 步骤 3：动臂技能加到 `_MOTION_SKILLS`

```python
_MOTION_SKILLS = {
    # ... 已有 18 个 ...
    "my_skill",
}
```

**这是唯一会暂停 WorldWatcher 信念融合的地方**——动臂时不暂停，被持物会被错误地重融合到空中位置。

#### 步骤 4：tracing + MCP 暴露自动

`SkillRuntime.execute()` 是单一调度入口；`mcp_server.py` 服务 `TOOL_SPECS` 减去 `_EXCLUDED_TOOLS` 加 `_EXTRA_TOOLS`——加完即可被 LLM / MCP 调用。

#### 步骤 5：可选加 regex 到 `_RULES`

让常见指令走 reflex 快路径（不调 LLM）：

```python
# src/wrc_demo/agent/reflex.py
_RULES.append((
    re.compile(r"^my command pattern\s+(\w+)", re.IGNORECASE),
    "my_skill",
))
```

### 9.2 加新 detector

`src/wrc_demo/perception/detector.py` 实现新 detector 类：

```python
from .detector import Detector

class MyDetector:
    def __init__(self, model_path: str, **kwargs):
        # 加载模型
        ...

    def detect(self, frame) -> list[Detection]:
        # 返回 Detection 列表
        ...
```

在 `apps/demo.py::_make_detector()` 加分支：

```python
if dcfg.type == "my_detector":
    from wrc_demo.perception.my_detector import MyDetector
    return MyDetector(dcfg.model)
```

### 9.3 加新 grasp planner

`src/wrc_demo/grasping/my_planner.py`：

```python
from .selector import Grasp

def plan_my_grasps(fix, cloud, cfg) -> list[Grasp]:
    # 返回 Grasp 列表
    ...
```

在 `SkillRuntime._plan_grasps()` 加层（默认叠加到 camera-frame / graspgenx / obb 三层）。

---

## 10. 架构概览

> **本章是参考材料**——理解 wrc_demo 怎么组织代码的指南。要写代码前先读一遍 §9.1 的流程；要理解为什么这样设计，看 §10.3 关键设计决策。

### 10.1 三级决策（Reflex → Habit → LLM）

wrc_demo 把决策拆成**三级快慢路径**，类似人类的快/慢思考：

```
N cameras ──CameraStream (thread each, latest-frame slot, drop-stale)
   │
   ├── WorldWatcher (thread, ~3 Hz): detector + HSV color tag
   │     └─> BeliefStore (thread-safe): label+color+3D+freshness
   └── StreamServer (MJPEG dashboard): camera grid + narration + object table

chat command ("pick and place pink object")
   ├─ tier 1 REFLEX    模板语法 → skill calls                          (~µs)
   ├─ tier 2 HABIT     经验记忆（hashed-BoW cosine ≥ 0.9）              (~ms)
   └─ tier 3 LLM       原始编排器循环（2-15 s/turn）
        全部 tier 走同一份 safety-gated SkillRuntime
```

**关键机制**：

- **Latest-slot 流式**：消费者永远拿到最新帧；传感器不会落后（`perception/stream.py`）
- **始终热的世界模型**：WorldWatcher 把每个相机的深度 + 外参连续融合到 BeliefStore，让命令解析变成字典查找，不是 observe→detect 往返。动时暂停（避免空中重融合），并向 safety 看门狗心跳
- **颜色不用 CLIP**：检测从 mask HSV 中位数映射成颜色名，存到信念，匹配查询里的颜色词——"pink object" 在闭集 COCO detector 上也工作。VLM 仍可被咨询（tier 3 / advisor），但**永不**走热路径
- **One-call pick-and-place**：`skill_pick_and_place` = resolve → grasp → place → home，整段执行时间 = 机械臂运动时间
- **LazyArm**：MCP gateway 在启动时预热摄像头/detector/世界模型，但电机在第一个 motion 命令之前**不**接通——启动 chat 服务器不能让机器人上电

### 10.2 模块图（精简版）

```
src/wrc_demo/
├── types.py            Frame / Detection / ObjectFix / Grasp / RobotState
├── config.py           YAML profiles (cameras/, arms/, llm/) 合并成 Cfg
│
├── perception/         # 4 个相机后端 + YOLOE detector + 深度对齐
├── calibration/        # 手眼标定（ArUco + Pinocchio FK + LM）
├── memory/             # episodic + belief + grasp outcome
├── control/            # Pinocchio FK/IK + mock + lazy + 真 RS + SafeArm
├── safety/             # fail-closed waypoint 校验
├── grasping/           # 3 层抓取规划
├── agent/              # orchestrator + reflex + advisor + LLM clients + trace
├── skills/              # SkillRuntime + TOOL_SPECS + 业务逻辑（teach、master_arm 等）
└── apps/                # demo / record / viewer / stream_server / mcp_server
```

### 10.3 关键设计决策

| 决策 | 为什么 |
|------|--------|
| **Frame.rgb 是 BGR，depth_m 是米制 float32** | L515（0.25 mm/unit）和 D4xx（1 mm/unit）的单位差异在 baseline 上制造过 bug 类；在相机边界转换让所有下游消费者单位安全 |
| **抓取在基坐标系，不在相机坐标系** | baseline 用相机光线方向推 approach，相机姿态影响抓取质量；这里把 mask 点提升到 3D、变换到基坐标系、那里的 OBB 给 yaw + width + height——相机姿态只影响可见性，不影响抓取几何 |
| **学习抓取默认，分析抓取总在** | `grasp.backend: graspgenx` 把物体基坐标系点云送到 GraspGen-X ZMQ 服务（`scripts/serve_graspgenx.sh`，:5556），把排名后的 6-DoF 抓取**前置**到分析 OBB 候选前面。服务端任何错误都静默回退到 OBB——展位永远不会因为模型服务器死了而卡死 |
| **自有运动学包装** | ReBotArm SDK 的运动学默默读自己的全局 config 文件（忽略你传的 hw_yaml），导致 DM 装 URDF 被加载给 RS 臂（错误的工具坐标系）。我们显式加载 `assets/` 里的 RS URDF |
| **反馈，不 sleep** | baseline 用 `sleep(duration + 0.6)`；RS 电机上 motorbridge 的 `get_state()` 永远不解码 type-0x18 报告帧，所以实位来自 `mechPos`（0x7019）param reads——在此 rig 上验证过 |
| **fail-closed 安全** | SDK 在 IK 之外不执行任何东西。我们的 harness 在每个流式 waypoint 上关闸；抓取下降发生在目标周围的显式豁免圆柱内，让"别碰桌面"和"抓住桌面上的物体"共存 |

### 10.4 模块依赖图

```
            ┌──────────────────────────────────────────────────────────────┐
            │                apps/  (entry points)                         │
            │   demo.py (build_runtime)   mcp_server.py   dashboard_runner │
            └────────────┬──────────────────────┬────────────────┬─────────┘
                         │                      │                │
        ┌────────────────▼─────────┐   ┌────────▼────────┐  ┌────▼─────────┐
        │   agent/  (orchestrator) │   │  skills/        │  │  perception/ │
        │   orchestrator.py        │◄──│  runtime.py     │  │  camera_base │
        │   reflex.py (tier1)      │   │   6 skills      │  │  + cameras/  │
        │   advisor.py  (tier3)    │   │  TOOL_SPECS     │  │  detector    │
        │   llm.py                 │   │  trace auto     │  │  depth_prov  │
        └────────┬─────────────────┘   └────────┬─────────┘  └────┬─────────┘
                 │                            │                │
                 ▼                            ▼                ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │              memory/  +  grasping/  +  control/                 │
        │   BeliefStore   GraspMemory   camera_grasp + obb_grasp + graspgenx│
        │   EpisodicMem   (3-layer)        Kinematics + MockArm + SafeArm  │
        └────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                       safety/  (fail-closed)                    │
        │   SafetyHarness.approve(waypoint)  vet_pose()  estop latch   │
        └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
              vendored SDK + pyorbbecsdk + ultralytics + pin + motorbridge
```

### 10.5 Trace 时序图（一次 `pick_and_place` 调用）

```
t=0.0s  user  ──→  build_runtime()
                   ├─ Kinematics(urdf, joint_signs)
                   ├─ make_arm(cfg) ── RebotArm.connect()
                   ├─ CameraRig(N) ── Open + warm_up
                   ├─ LockedDetector(YOLOE weights)
                   ├─ EpisodicMemory / BeliefStore / TraceLogger
                   └─ SkillRuntime(...)

t=0.5s  WorldWatcher thread starts (3 Hz detector pass → beliefs)

t=1.0s  user --task "pick up red block"
        AgentOrchestrator.run_task()
        ├─ tier1 REFLEX: regex match ──→ 命中 pick_and_place?  ──→ 执行
        └─ (fallback) tier2 HABIT ──→ tier3 LLM ──→ tool_calls

t=1.1s  SkillRuntime.execute("pick_and_place")
        ├─ tracer.start()
        ├─ belief.find("red block")  ── wait deadline 5s
        ├─ _plan_grasps() ── camera + graspgenx + obb ── re-rank
        ├─ select_grras() ── IK ── harness pre-vet
        ├─ safe_arm.move_joints() ── each 50Hz waypoint ── SafetyHarness.approve
        ├─ gripper close two-stage ── stall detection
        ├─ lift + place + home
        └─ tracer.finish()  ── trace.jsonl + keyframes + narration

t=N.5s  report.success ── print "✓ done — arm parked at home"
```

### 10.6 Grasp 规划三层对比

| Layer | 触发条件 | 数据源 | 输出 | 失败行为 |
|-------|----------|--------|------|----------|
| **1. camera-frame planner** (`camera_grasp.py::plan_grasp_from_mask`) | 总是——只要 detection 有 mask + depth | segmentation mask + depth → `cv2.minAreaRect` → depth-quantile back-project → camera-frame Grasp | `Grasp(approach, position, width, height)` | layer 2 |
| **2. GraspGen-X ZMQ** (`graspgenx_backend.py`) | `grasp.backend == "graspgenx"` 且 server :5556 可达 | base-frame point cloud → learned 6-DoF grasp | `Grasp` 列表（top-k=32）| server timeout / disconnect → silent fallthrough |
| **3. analytic OBB** (`obb_grasp.py::plan_grasps_from`) | always | base-frame OBB (yaw, width, height) | `Grasp` 列表 | (no failure — always returns) |

所有 layer 输出同一 `Grasp` dataclass；`GraspOutcomeMemory` 跨 layer re-rank；`select_grasp()` 做 jaw-width filter + IK walk + harness pre-vet（含 7 样本沿下降段）。

### 10.7 验证状态（截至 2026-08-28）

- **17 单元 + 集成测试**通过，1 个 skipped（mock stack，约 30s）
- **完整 suite**：约 290 单元 + 集成测试通过（hardware 标记默认 deselected）
- **2 次对抗性多视角 review**：2026-07-16（4 reviewer × skeptic verification，29 confirmed defects）+ 2026-07-18 livestreaming redesign（45-agent workflow，33 confirmed findings）
- **真机验证**：L515 streaming、RobStride mechPos param reads（can0 read-only）、YOLO inference（CUDA, GB10）、GraspGen-X 后端（first-light 验证）
- **未验证**：真机 motion（需要现场夹爪重新验证 + 手眼标定）

---

## 11. 故障排查速查表

### 11.1 安装类

| 症状 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'torch'` | pip 把 torch 装到 user site-packages | 重跑 §3.6 并加 `PYTHONNOUSERSITE=1` |
| `OSError: libcudart.so.13: cannot open shared object file` | conda env 没 CUDA runtime | 重装 torch 用 §3.6 方式 |
| `RuntimeError: condaHTTPError: HTTP 000 CONNECTION FAILED` | `~/.condarc` 与 `environment.yml` channels 不一致 | 已修复（v6.5 env 只配 conda-forge） |
| `resolution-too-deep`（pyorbbecsdk2） | 依赖链太深 | 按 §3.5 方式一单独装 |
| `pytest 默默 0 测试通过` | `addopts = "-m not hardware"` 拦截 | 加 `-m hardware` 强制 |
| `ModuleNotFoundError: No module named 'motorbridge'` / `'pin'` | extras 没装齐 | 重跑 `conda env create -f environment.yml` |
| `ModuleNotFoundError: No module named 'pyorbbecsdk'` | §3.5 还没装 | 装 Orbbec SDK |
| `ModuleNotFoundError: No module named 'fashionstar_uart_sdk'` | teleop SDK 没装 | `pip install fashionstar-uart-sdk` |

### 11.2 标定类

| 症状 | 原因 | 解决 |
|------|------|------|
| RMSE > 10 mm | ArUco 棋盘边长不准 / 采样太少 | 用尺子精确测；`--n 50` 多采样 |
| `ArUco marker not detected` | 棋盘打印模糊 / 光线太暗 | 重打印 600 DPI；增加台灯 |
| `requires_calibration=true but no extrinsics source` | 没跑 / /文件不对 | 跑 §6.2；输出名 `hand_eye_top.local.json` |
| 标定后实物仍抓空 | serial 不匹配 | `wrc-calib-top --list` 看 serial vs `hand_eye_top.json` 里的 |

### 11.3 运行类

| 症状 | 原因 | 解决 |
|------|------|------|
| `LLMConfigError: ANTHROPIC_API_KEY not set` | 云端 API 没设 env var | `export ANTHROPIC_API_KEY=...` |
| demo 卡 30s+ | 云端 LLM 网络慢 | 切 `--llm mock` 验证 |
| `wrc-demo: command not found` | 没 `conda activate wrc-demo` | activate env |
| `Cannot find camera by serial '...'` | 标定 serial 不匹配 | 重新跑 §6.2 或设 `WRC_CAMERAS` env 覆盖 |
| `ModuleNotFoundError: No module named 'pyorbbecsdk'` | §3.5 没装 | 装 Orbbec SDK |

### 11.4 性能类

| 症状 | 解决 |
|------|------|
| `CUDA out of memory` | 在 `configs/demo.yaml` 加 `detector.device: cpu`；或切到 cu128 装 |
| FPS 低（< 5） | `configs/demo.yaml`` perception_loop.rate_hz` 调低 |
| vLLM 启动慢 | `bash scripts/start_cosmos3_server.sh --bg` 后等 60s |

### 11.5 推理类

| 症状 | 排查步骤 |
|------|----------|
| YOLOE 没有任何 detection | | 1. 检查 `models/mobileclip2_b.ts` 存在 <br>2. ultralytics 自动装 `clip` fork 后 `python -c "import clip"` 不报错 <br>3. `MobileCLIPTS` 用绝对路径传 `weight`（CWD-relative 见 §4.2 步骤 B） |
| VLM 答错/不动臂 | | 1. 切 `--llm mock` 验证 reflex 路径 <br>2. 看 trace.jsonl 里 skill call 与预期是否一致 <br>3. 调 `detect_classes` 让 YOLOE 词汇对齐 |
| 物体识别不到指定名称 | | 1. `detect_classes` 加名称 <br>2. `WRC_DETECT_CLASSES` env var 覆盖 <br>3. YOLOE 文本提示精度有限，必要时训练自定义 YOLO |
| `bwrap panic`（Codex CLI）| | 路径含中文 basename 会触发。把工作目录改到纯 ASCII。详见 CLAUDE.md |

---

## 12. 变更日志

### v6.5（2026-08-28，环境踩坑修复 + 文档重写为操作指南）

**环境修复**：
- `environment.yml`：加 `pandas>=2.0,<3` pin；channel 列表去掉 `defaults` 只留 `conda-forge`；移除 `pyorbbecsdk2`（依赖链太深）；加 `torch>=2.0` 显式 pin
- `src/wrc_demo/__init__.py`：ultralytics text-encoder monkey-patch 改 lazy（首次 detector init 才 import）

**文档重写**：
- 从"开发文档视角"（架构详解 + 配置系统 + 二次开发指南）改为"操作指南视角"（环境配置 → 模型下载 → LLM 切换 → 手眼标定 → demo 运行 → 故障排查）
- 新增章节：5 分钟快速开始、自定义配置、加新技能、故障排查速查表
- 删除章节：项目特点 / BOM / 环境要求 / Scripts 总览 / 目录结构 / 架构概览（保留为单独 wiki 章节）/ 配置系统详解 / 技能系统详解 / 测试详解 / 部署 CI / 二次开发指南详解 / FAQ / 联系方式 / 参考文档 / Changelog 详细

**教程使用方式**：
```bash
# mock only：~ 5 分钟
git clone … && conda env create … && pytest … && wrc-demo --task wave

# 真机：~1-2 小时
mock + §3 系统依赖 + §3.5 Orbbec SDK + §3.6 PyTorch + §4 模型下载 + §5 LLM 配置 + §6 手眼标定 + §7 demo 运行
```
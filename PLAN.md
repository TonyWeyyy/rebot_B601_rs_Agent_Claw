# PLAN.md — 真实 Orbbec 顶部相机标定 & wrc_demo 对话控制机械臂

> 版本：v1.2 · 日期：2026-07-20 · 状态：所有开放问题已关闭，可进入实施（Phase 0）
> 变更记录：v1 → v1.1 固化 A1-A10、Q1-Q4、Q6、Q7；v1.1 → v1.2 关闭 Q5（选 α 维持现状）。
> 用途：本仓库首个贯穿"**对话 → Skill → 真实机械臂 + 真实相机**"完整链路的开发规划。
> 核心目标：把 Orbbec Gemini 2 顶部相机接进 wrc_demo、正确标定、让已经在 sim 跑通的 skill 链路在真机生效；腕部 Orbbec 暂只做 stream/overlay。
> 本文件遵循 `spec-driven-development`（spec + plan）的 Phase 1/2 与 `planning-and-task-breakdown` 的任务模板，可直接拆任务执行。

---

## 0. 假设清单（已确认 2026-07-20）

假设已与用户拍板；本阶段以此为前提实施。每条末尾的"用户决议"标注本次确认的措辞，详见 §0.1。

- **A1.** 项目根：`/home/seeed/wrc_demo`（CLAUDE.md 写的 `/home/johnny/...` 是另一台 demo rig，本机路径在 `/home/seeed`）。
- **A2.** 共享 venv 以本机实际为准（CLAUDE.md 老引用 `/home/johnny/Projects/demo/.demo` 是另一台 demo rig，本机常在 `/home/seeed/miniforge3/envs/<env>` 如 `DM`）；所有重依赖走 `uv pip install --python "$PY" ...` 装到目标 venv。`pyorbbecsdk` 不在 base deps，需要在目标 venv 中安装；本机 DM env 已自带，可直接复用。CLAUDE.md 中与本机相关的路径以 `setup_agents.py --print-python` 输出为准。
- **A3.** 已有两台 Orbbec Gemini 2：历史上 `wrist = AY3794302V9`/`top = AY3Z331006L`。本次插线后必须**重新枚举并写入 binding**，不能直接信历史文件。
- **A4.** `assets/` 内 RS URDF/USD 是镜像约定（`q_asset = -q_local`），`joint_signs: [-1,-1,-1,-1,-1,-1]`；本次复用的 cuRobo 工具夹在 `Kinematics(ee_frame='gripper_end', joint_signs=...)` 上即可对齐。cuRobo 项目自身的 `end_link` 不可直接继承。
- **A5.** 第一台相机 = 顶部 Orbbec，承担 manipulation、3D belief 融合、抓取定位；腕部 = 辅助视角，`fuse_beliefs: false`，绝不准参与 3D 融合。
- **A6.** 不复制两套标定参数/约定：标定工具重写为可复用的 `src/wrc_demo/scripts/calib_top_orbbec.py`，**直接 import `wrc_demo.control.kinematics.Kinematics`** 作为 `fk(q_arm)` 来源；不再依赖 cuRobo 那边的 `RebotMotionPlanner` + cuRobo FK。
- **A7.** 历史 `curobo_rebot_demo/data/calibration/hand_eye_top.npz` 视为**不可信残留**：其生成时内嵌 `T_marker2gripper=I` 假设，T 平移 ≈ 850 mm 与真实安装不符，必须重新采集。
- **A8.** 标定方程采用`X = T_cam2base`、`Y = T_marker2gripper`联合 SE(3) 非线性最小二乘；不沿用 `cv2.calibrateHandEye` 把 Y 写死为 I 的旧路径。
- **A9.** 顶部 Orbbec 必须输出 **RGB + 硬件对齐深度 + 米制 + 每相机 SDK 内参 K**；wrist 同样需 SDK K（即便不参与融合也要给 dashboard 提供正确的检测覆盖）。
- **A10.** 本阶段不实施真机抓取动作（CLAUDE.md/ROADMAP 已声明：onsite bring-up checklist 中的"第一次 grasp"是 S7 之后的事）。当前到 S6（rig 验证、table plane 拟合、`fuse_beliefs=true` 主流程跑通）为止。

### 0.1 确认记录（2026-07-20）

| ID | 用户决议 | 落到代码的位置 |
|---|---|---|
| A1 | 本机路径（`/home/seeed/wrc_demo`） | PLAN.md / run_dir / 安装指令一致采用本机路径 |
| A2 | 选项 a：用现有 `.demo` venv，`uv pip install pyorbbecsdk` | `src/wrc_demo/scripts/calib_top_orbbec.py` 不创建 venv；README.md 加 `pyorbbecsdk` 安装一行 |
| A3 | 按实物枚举：binding 必须现场重写 | `src/wrc_demo/scripts/calib_top_orbbec.py --bind` + `data/calibration/cameras.json` |
| A4 | wrc_demo 的 `Kinematics.fk`（RS URDF + `joint_signs=[-1]*6` + `ee_frame='gripper_end'`） | `src/wrc_demo/scripts/calib_top_orbbec.py` 内构造 `Kinematics(...)` |
| A5 | 锁死：腕部 `fuse_beliefs=false`，永不参与 3D | `configs/cameras/orbbec_wrist.yaml` + demo.py `requires_calibration` 策略 |
| A6 | 完全重写：FK 来源切到 wrc_demo；不再依赖 cuRobo 的 planner / bridge | 新写 `src/wrc_demo/scripts/calib_top_orbbec.py`；零 cuRobo import |
| A7 | 不导入历史 npz | `Extrinsics.from_config` 对 cuRobo npz 仅做"识别+警告"，不作为运行时真值 |
| A8 | 不沿用 `cv2.calibrateHandEye`；联合 SE(3) LM + Huber + LOO | `calibration/handeye_solver.py` 完全自实现（基于 §Q7 决定） |
| A9 | 双机都要 SDK K（顶部与腕部各自 K） | `OrbbecCamera` 每实例持自身 `_K`；profile 不允许共用 K |
| A10 | 默认（不实施真机 grasp，到 rig 验证结束） | ROADMAP 中"wrist 同步握抓"放入 Mid-term |

---

## 1. Objective（目标）

### 1.1 用户故事

```
as a demo operator at a booth
I want to say "pick and place the red box" (or any color/label) to the chat
so that the reBot arm grasps the right object on the table using only real
sensors — no manual extrinsics, no sim fallback
```

### 1.2 技术目标

1. **顶部 Orbbec 标定产物可信**：给出一组 `T_cam2base` + `T_marker2gripper`，使基坐标系下的目标 3D 定位（`ObjectFix.position`）≤10 mm 平均误差、≤20 mm 最大误差。
2. **wrc_demo 多相机链路可独立 stream 和 dashboard 化**：顶部相机驱动 `WorldWatcher` 写入 3D belief；腕部相机仅 stream + overlay，绝不参与 3D 融合。
3. **CALIB-DATA-flow 自动化**：可复现的"采集 → 求解 → 验证 → 加载 → 真机使用"pipeline；任意步骤失败 fail-closed，不得 silent fallback 到 identity。
4. **现有 skill 链路零回归**：`build_runtime` 在 `--cameras orbbec_overhead,orbbec_wrist --arm rebot_rs` 下跑通，"pick and place pink object" 在 mock stack 完全等价。

### 1.3 验收点（可测）

- `pytest tests/ -q` 全绿（默认 `-m 'not hardware'`，含新增的 orbbec 离线测试）。
- 新增 hardware 测试 `pytest tests/ -m hardware -q`：`can0` + 顶部 Orbbec 同时存在时通过；离线时若没有则 skip/失败信息清晰。
- `data/calibration/hand_eye_top.json` 由真实采集 + 新求解器产出，含 `metrics.translation_rmse_mm`、`metrics.rotation_rmse_deg`、`metrics.max_err_mm`、`metrics.n_inliers`、`metrics.serial`、`metrics.marker_size_m`。
- 在已知 base 坐标系参考点（建议：marker 重置于桌面固定坐标 `(0.30, 0.0, 0.0)`）做 cross-check，平均绝对误差 ≤10 mm。

---

## 2. 范围与边界

### 2.1 In Scope（本期做）

| # | 能力 |
|---|---|
| S1 | `perception/orbbec_camera.py`：新 backend，lazy import `pyorbbecsdk`，串号绑定，深度对齐，K/D 获取 |
| S2 | `configs/cameras/orbbec_overhead.yaml` & `orbbec_wrist.yaml`：profile 设计、默认参数、强制 `fuse_beliefs` 行为 |
| S3 | `camera_base.make_camera()` 新增 `orbbec` 分支 |
| S4 | 标定采集工具 `src/wrc_demo/scripts/calib_top_orbbec.py`：基于 cuRobo 04b 的状态机与可视化，**FK 切到 wrc_demo `Kinematics`**，采集产物为 `TopCalibSession` (dataclass list of samples) |
| S5 | 新求解器 `calibration/handeye_solver.py`：联合求解 X = T_cam2base、Y = T_marker2gripper；Huber 损失、留一交叉验证、ArUco 重投影残差、已知基坐标 cross-check |
| S6 | `perception/grounding.py:Extrinsics.from_config` 扩展：识别 cuRobo 风格 NPZ（`T_cam2base`/`T_cam2gripper`/`mode`/`camera_serial`/`metrics`）+ 旧 `T_result` 风格做向后兼容；显式校验 mode 与 profile eye-to-hand/eye-in-hand 一致 |
| S7 | NPZ 写入格式翻新：含 `metrics`、`serial`、`schema_version`、`frame_convention`（`'wrc_demo'` / `'curobo'`）、`marker_size_m`、`T_marker2gripper`、`T_cam2base`、`mode` |
| S8 | 单元测试：相机 mock backend + 求解器的合成测试（已知位姿、噪声、离群点）+ Extrinsics loader schema 矩阵测试 |
| S9 | hardware bring-up checklist：复刻 CLAUDE.md/ROADMAP 的"onsite bring-up"步骤并加自动化 smoke |
| S10 | docs 更新：`docs/ARCHITECTURE.md` 加 Orbbec backend 模块；`README.md` 把 onsite bring-up 的 hand-eye 段更新为新工具路径；保留 ROADMAP 不动除非新增条目 |

### 2.2 Out of Scope（本期不做，文档上要明文列出）

- 腕部 Orbbec 的手眼标定（`eye_in_hand`）
- 两台相机的硬件时间同步
- 多相机稠密点云融合 / Bundle Adjustment
- 双相机立体 RGB-D 或 ToF 重建
- 真机 grasp 动作执行（放到 S8/S9 之后的 ROADMAP 中）
- 手眼标定的在线自检/重启
- LLM/MCP 协议层改动

### 2.3 Always / Ask First / Never

**Always**

- 重依赖 `pyorbbecsdk` 在 `OrbbecCamera` 首次 `open()` 时再 import（保持 wrc_demo mock stack 不加载）
- 标定工具不直接信任历史 binding 文件；启动时实枚举、按 USB index 一致性提示
- `Extrinsics.from_config` 校验 `camera_serial` 与 profile 内 `serial` 一致，不一致时发 warning 但允许 override（用于一台相机下线后换机）
- 所有标定产物落在 `data/calibration/`，并附 metrics + 时间戳；不允许 silent overwrite
- `Frame.rgb is BGR`；`depth_m` 是 `float32` 米制；遵守 CLAUDE.md 全部 Contracts & invariants
- `time.monotonic()` 用于 `Frame.t`；`time.time()` 用于持久化字段
- 真实 gripper motion 之前必须有 `SafeArm`；标定采集用 `RebotRSArm` 直连包一层 `SafeArm` 接关
- 退出采集和改 NPZ 都要写 trace 行（`runs/calibration_*.jsonl`），便于审计

**Ask first**

- 新增顶层依赖（`pyorbbecsdk`）、`.demo` venv 内的额外包
- 任何修改 `apps/demo.py` 中 `build_runtime()` 的核心流程
- 任何修改 `Kinematics`、`SafeArm`、`SafetyHarness`、`WorldWatcher` 行为
- 任何把 `joint_signs`、base frame 约定硬编码的变更
- 把 GraspGen-X 路径或任何 grasp pipeline 改动带入

**Never**

- 用 `T_marker2gripper = I` 作为求解手段（harness 的拒绝 = 真抓失败）
- 在 `cfg.camera` 缺失 extrinsics 时把 `fuse_beliefs` 默认成 true
- 让腕部 camera profile（未标定）参与 3D 融合
- 把 cuRobo 的 `end_link` 坐标系直接套到 wrc_demo 的 `gripper_end`（必须用 wrc_demo `Kinematics`）
- 在 motion 上下文外探测 `SafeArm.raw` 的 backend capability
- 用 `==`/`list.remove()` 处理 `ObjectBelief`（numpy 字段会让 dataclass `__eq__` 抛歧义）

---

## 3. Commands（执行清单）

> 全部走 `.demo/bin/python`，设置环境变量 `PYTHONPATH` 视安装情况：editable install 已经做完则不需要；否则 `PYTHONPATH=src`。

```bash
# 1. 装包（首次；按本机 venv；见 ASSUMPTIONS A2）
#    本机环境 pyorbbecsdk 已自带，可跳过；缺失再装
PY="${PY:-$(command -v python3 2>/dev/null || echo /home/seeed/miniforge3/envs/DM/bin/python)}"
#    上述默认只为本机常见路径占位；推荐用 scripts/setup_agents.py --print-python 取实际路径
uv pip install --python "$PY" pyorbbecsdk || echo "[skip] pyorbbecsdk already present in $PY venv"
PY="$(scripts/setup_agents.py --print-python 2>/dev/null || echo /home/seeed/miniforge3/envs/DM/bin/python)"
echo "Using PY=$PY"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"   # editable install 已做可省；不为零加 src/

# 2. 离线 wiring check（不依赖硬件）
$PY -m pytest tests/ -q
$PY -m wrc_demo.apps.demo --task "pick and place pink object"   # 默认 mock

# 3. 列出 Orbbec 设备（不抓帧）
$PY -m wrc_demo.scripts.calib_top_orbbec --list

# 4. 顶部标定采集（实机 + 顶部相机单插）
sudo ip link set can0 up type can bitrate 1000000
$PY -m wrc_demo.scripts.calib_top_orbbec \
    --marker-size 0.10 \
    --settle-time 1.5 \
    --marker-timeout 4.0 \
    --stable-frames 4 \
    --out data/calibration/hand_eye_top.json

# 5. cross-check（已知 base 参考点）
$PY -m wrc_demo.scripts.calib_top_orbbec --verify \
    --npz data/calibration/hand_eye_top.json \
    --known-points "0.30,0.00,0.02;0.25,-0.05,0.02;0.35,0.05,0.02"

# 6. 单相机实机 demo
$PY -m wrc_demo.apps.demo --interactive \
    --cameras orbbec_overhead --arm rebot_rs --llm anthropic

# 7. 双相机实机 demo（顶部标定 + 腕部 stream-only）
$PY -m wrc_demo.apps.demo --interactive \
    --cameras orbbec_overhead,orbbec_wrist --arm rebot_rs --llm anthropic

# 8. 单元测试（增量 + 全量）
$PY -m pytest tests/test_orbbec_camera.py -q
$PY -m pytest tests/test_handeye_solver.py -q
$PY -m pytest tests/test_extrinsics_loader.py -q
$PY -m pytest tests/ -q                       # 全量默认集（不含 hardware）

# 9. hardware 测试（仅在 rig 上）
$PY -m pytest tests/ -m hardware -q
```

---

## 4. Project Structure Changes（路径 + 用途）

> 新增/修改均最小；保持现有 `src/wrc_demo/perception/` `apps/` `control/` `safety/` 边界；新文件不引入新顶层目录。

```text
wrc_demo/
├── PLAN.md                                # 新增（本文件）
├── configs/cameras/
│   ├── orbbec_overhead.yaml               # 新增（top camera profile）
│   └── orbbec_wrist.yaml                  # 新增（wrist camera profile, fuse_beliefs=false）
├── src/wrc_demo/perception/
│   ├── camera_base.py                     # 修改：make_camera() 新增 'orbbec' 分支（仅 6 行）
│   ├── orbbec_camera.py                   # 新增：OrbbecCamera(CameraBase)
│   ├── grounding.py                       # 修改：Extrinsics.from_config 多 schema 识别
│   └── __init__.py                        # 不变
├── src/wrc_demo/calibration/              # 新建模块（解耦标定工具）
│   ├── __init__.py
│   ├── calib_dataset.py                   # 标定 sample 数据类 + JSON 持久化
│   ├── handeye_solver.py                  # 联合 SE(3) 求解器
│   ├── aruco_session.py                   # ArUco 检测 + PnP（项目内统一的 cv2.aruco 调用）
│   ├── frame_convention.py                # base / gripper_end / cuRobo end_link 转换 utilities
│   └── realsense_intrinsics_proxy.py      # 不改：cuRobo 风格 npz 加载（保持旧字段）
├── scripts/                                         不动（保留 dashboard_runner.py、setup_agents.py 等 shell / 辅助 Python）
└── src/wrc_demo/scripts/                            # 见 ADR-8
    ├── __init__.py                                   # 新增：让 wrc_demo.scripts 成合法包
    └── calib_top_orbbec.py                           # 新增：CLI 入口（python -m wrc_demo.scripts.calib_top_orbbec）
├── tests/
│   ├── test_orbbec_camera.py              # 新增：用 fake pyorbbecsdk 测 open/_grab/lifecycle
│   ├── test_handeye_solver.py             # 新增：合成数据（含噪声/离群点）求解精度
│   ├── test_extrinsics_loader.py          # 新增：cuRobo / 老 baseline / 新 schema 三类 NPZ 加载
│   ├── test_frame_convention.py           # 新增：cuRobo fk → wrc_demo fk round-trip
│   ├── conftest.py                        # 不变；可能小幅增 fixture（fake_orbbec_sdk）
│   ├── test_hardware_smoke.py             # 可能增：orbbec 顶部相机硬件 smoke
│   └── test_review_regressions.py         # 不变（本规划新增 review 项若触发需新增 v3 文件）
├── data/calibration/                      # 新建（git ignore 之外用 .gitkeep 占位）
│   ├── hand_eye_top.json                  # 新求解器产物（含 metrics、serial、schema_version）
│   └── cameras.json                       # 角色 → serial binding（替代 cuRobo 同名文件）
├── docs/
│   ├── ARCHITECTURE.md                    # 修改：在 perception 段加 orbbec_camera.py + calibration/ 模块
│   └── ROADMAP.md                         # 不动除非新增条目（顶部标定完成后把 wrist 列入 Mid term）
└── README.md                              # 修改：Onsite bring-up 的 hand-eye 段落改指本计划工具
```

**关键不变量**：

- 不创建新顶层 app / script 命名空间；CLI 入口走 `python -m wrc_demo.scripts.calib_top_orbbec`
- 不复制 cuRobo 的 pose 列表（直接 `from wrc_demo.calibration.calib_dataset import TOP_CALIB_POSES`，常量以 `wrc_demo.calibration.calib_dataset` 为唯一定义点）
- 不复用 cuRobo 的 `RebotMotionPlanner` / `RebotCuroboBridge`（用 wrc_demo 的 `Kinematics` + `SafeArm`）

---

## 5. Code Style（沿用 wrc_demo + 本规划新增的硬性约定）

### 5.1 沿用（CLAUDE.md 强制）

- `Frame.rgb` 是 BGR；`depth_m` 是 float32 米；K 是 float64 3×3
- `Frame.t` / `RobotState.t` / `belief age` 用 `time.monotonic()`
- 全局 Cfg hybrid：`cfg.xxx.attr` for required, `.get` for optional
- 配置变更走 `cfg._data[...]`（无 override API）
- 重依赖必须在引用它的函数/方法体里 lazy import
- `SafeArm.raw` 探测只能在 motion 上下文内（避免驱动 LazyArm 物化）
- 错误分类：`SkillError` / `SafetyViolation` / `RuntimeError` 严格分离
- 命名：snake_case functions/variables，PascalCase classes，UPPER_SNAKE_CASE constants

### 5.2 新增约定（本规划）

- **统一标定 sample / result 数据类**（在 `wrc_demo/calibration/calib_dataset.py`）：`CalibrationSample`、`CalibrationResult`、`CameraBinding`，用 `@dataclass(frozen=True)`
- **JSON 持载标定产物**（不直接 npz，方便 diff/回滚/审计）；NPZ 仅用于 `K`、`D` 二进制
- **文件级 docstring 顶部明示 schema_version + frame_convention**：
  ```python
  """
  Schema version: 1
  Frame convention: wrc_demo (base frame +x away, +y left, gripper_end TCP)
  """
  ```
- **不复制 import 顺序遵循**：stdlib → third-party → local (`from ..types import Frame`)
- **CLI 走 `argparse`，全部默认值与 profile 解耦**（profile 是 runtime 行为；标定工具不依赖 demo.yaml）
- **类型注解一律用 PEP 604（`np.ndarray | None`）**
- **fake SDK 注入命名统一为 `fake_orbbec_sdk`**：`tests/conftest.py` 提供 `fake_orbbec_sdk(monkeypatch)` fixture，替换 `sys.modules['pyorbbecsdk']` 为内部 `FakeSDK` 类；任何 `OrbbecCamera` 测试沿用此 fixture，不再各自定义（与 ADR-6、§21 Phase 1 Task 1.0 描述对齐）

### 5.3 Example

```python
# src/wrc_demo/perception/orbbec_camera.py
"""
Orbbec Gemini 2 backend (single-camera pyorbbecsdk wrapper).

Schema version: 1
Frame convention: wrc_demo (base frame +x away, +y left, gripper_end TCP)

This backend is intentionally a thin layer over `pyorbbecsdk`; all motion / FK
math lives in wrc_demo.control.kinematics.Kinematics and configuration is in
configs/cameras/orbbec_*.yaml. See PLAN.md S1.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


class OrbbecCamera(CameraBase):
    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self._cfg = cfg
        self._serial = str(cfg.get("serial", "")) or None
        self._pipeline: Any = None
        self._align: Any = None
        self._K: np.ndarray | None = None
        self._depth_scale: float = 0.001

    @property
    def has_depth(self) -> bool:
        return True

    def open(self) -> None:
        try:
            import pyorbbecsdk as ob  # type: ignore
            from pyorbbecsdk import Config  # type: ignore
        except ImportError as e:  # pragma: no cover - hardware path
            raise CameraError(
                "pyorbbecsdk not installed; install pyorbbecsdk into .demo or "
                "fall back to configs/cameras/isaac.yaml in CI"
            ) from e

        cfg = Config()
        if self._serial:
            cfg.set_device_serial_number(self._serial)
        # ... configure streams, enable depth-color alignment ...
        self._pipeline = ob.Pipeline()
        self._pipeline.start(cfg)
        # ... read depth_scale and K from device profile ...

    def _grab(self) -> Frame:
        if self._pipeline is None:
            raise CameraError("Orbbec pipeline not started")
        try:
            frames = self._pipeline.wait_for_frames(500)
            frames = self._align.process(frames) if self._align is not None else frames
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if color is None or depth is None:
                raise CameraError("incomplete frameset")
        except Exception as e:  # noqa: BLE001 - SDK raises generic
            raise CameraError(str(e)) from e

        bgr = np.frombuffer(color.get_data(), dtype=np.uint8).reshape(
            color.get_height(), color.get_width(), 3
        ).copy()
        depth = np.frombuffer(depth.get_data(), dtype=np.uint16).astype(
            np.float32
        ) * self._depth_scale
        # apply min/max clamp per profile
        ...
        return Frame(
            rgb=bgr,
            depth_m=depth,
            K=self._K if self._K is not None else np.eye(3),
            depth_source="sensor",
        )
```

---

## 6. Testing Strategy

### 6.1 三层测试金字塔

| 层级 | 范围 | 位置 | 速率 | 何时跑 |
|---|---|---|---|---|
| Unit | 算法（求解器、ArUco、Extrinsics loader、frame convention） | `tests/test_handeye_solver.py` 等 | ms | 全量/CI |
| Integration | backend + extrinsics + WorldWatcher（Fake SDK，不接硬件） | `tests/test_orbbec_camera.py` | 百 ms | 全量/CI |
| Hardware | 真 Orbbec + can0；smoke / cross-check | `tests/test_hardware_smoke.py` 内 sub-section | s/min | `pytest -m hardware` 单独跑 |

### 6.2 覆盖率与门槛

- **新代码 ≥80% 行覆盖**（`pytest --cov` 局部检查）
- **求解器 ≥95% 行覆盖**（核心算法）
- **离线测试集 100% 通过**（默认 `-m 'not hardware'`）
- **新增 review findings 走 `tests/test_review_regressions_v3.py`**（CLAUDE.md 已约定模式）

> **覆盖率门 §6.2 的当前实现**：覆盖率与 addopts gate 已**写进 `pyproject.toml` 的 dev deps 与 [tool.pytest.ini_options]**：
>
> - `[project.optional-dependencies].dev` 加 `pytest-cov>=4.0`（`uv pip install -e ".[dev]"` 时自动装）
> - `[tool.pytest.ini_options].addopts` 计划改为 `"--cov=wrc_demo --cov-report=term-missing --cov-fail-under=80"`
>
> 当前 addopts 暂未启用 `--cov-...` 参数，原因是**本机 DM env 未装 `pytest-cov`**（沙箱无网络装不上）；启用时由后续 PR 在验证 `uv pip install -e ".[dev]"` 装上 `pytest-cov` 之后 push 这一行，避免 CI / 本地因为插件缺失而 break。门槛 80% / solver 95% 的承诺仍按本节条目。

### 6.3 关键测试用例（关键场景必加）

1. `test_handeye_solver_perfect_data`：合成 X、Y，10 个 pose 加零噪声，求解后残差 < 1e-9
2. `test_handeye_solver_robust_to_noise`：σ=1mm/0.5° 噪声，RMSE 增长 < 5 mm
3. `test_handeye_solver_rejects_outliers`：20% 离群 → 求解收敛，inlier 残差 < 5 mm，被剔除 sample 单独报告
4. `test_handeye_solver_baseline_same_T`：与传统 `cv2.calibrateHandEye(eye_to_hand)` 在 marker 已知 = gripper 的子用例上数值一致（仅作 sanity，不作主路径）
5. `test_orbbec_camera_uses_lazy_import`：在没有 `pyorbbecsdk` 的环境构造 `OrbbecCamera` 不会触发 ImportError
6. `test_orbbec_camera_grab_calls_align`：Fake SDK 记录 `align.process` 被调用次数
7. `test_extrinsics_loader_curobo_npz`：加载 cuRobo 风格 npz（`T_cam2base`、`T_cam2gripper`），eye_to_hand mode 下取 `T_cam2base`
8. `test_extrinsics_loader_baseline_npz`：加载老 baseline `T_result` 风格
9. `test_extrinsics_loader_new_schema_json`：加载新 schema（含 metrics、serial、frame_convention）
10. `test_extrinsics_loader_serial_mismatch_warning`：profile serial 与 NPZ serial 不同 → 警告且仍用 NPZ
11. `test_frame_convention_fk_roundtrip`：同一 q 在 cuRobo FK 与 wrc_demo FK 下的 `T_gripper2base` 关系是显式可枚举的刚性变换
12. `test_wrist_profile_force_fuse_false`：即使在 profile 里误设 `fuse_beliefs: true`，loader 也强制降为 false（除非提供 calibration npz + serial + Valid metrics）
13. `test_calib_dataset_save_load_roundtrip`：标定产物 JSON 持久化 100% 复原（位于 `tests/test_calib_dataset.py`）

### 6.4 Hardware Tests

> 用 `-m hardware` 隔离，避免 CI 误跑。

- `test_hardware_smoke.py::test_top_orbbec_serial_enumerate`：能列出设备，serials 与 `cameras.json` 一致
- `test_hardware_smoke.py::test_top_orbbec_first_frame`：启动 → 1s 内拿到 RGB + 对齐深度
- `test_hardware_smoke.py::test_top_orbbec_fuse_beliefs_default`：用 `orbbec_wrist.yaml` profile → `fuse_beliefs=False` 已生效
- `test_hardware_smoke.py::test_can0_param_read_only`：连 RS 上 `mechPos` 可读（已有）
- `test_hardware_smoke.py::test_calib_verify_known_points`：在已知 base 平面点 cross-check，平均 ≤10 mm

### 6.5 旧测试不破

- `test_perception.py`：`Extrinsics.from_config` 行为兼容（已有 `T_result` 用例）
- `test_kinematics.py`：不依赖新模块
- `test_streaming_fastpath.py` / `test_orchestrator_e2e.py`：mock camera 默认场景不受影响
- 不动 `test_review_regressions*.py`，新增发现才补 v3

---

## 7. Success Criteria（项目级 DoD + 任务级 acceptance）

### 7.1 项目级 DoD（本期完成定义）

- [ ] Orbbec backend 加入 `make_camera()`，`orbbec_overhead` 与 `orbbec_wrist` profile 加载、stream 通路完整
- [ ] 真实采集至少 30 个有效样本，SER(3) 残差聚合指标全部达标（见 §15）
- [ ] `Extrinsics.from_config` 识别新旧三类 schema，行为矩阵测试全过
- [ ] 顶部相机 `fuse_beliefs=true` 真实生效于 `WorldWatcher`，可见新 belief 写入
- [ ] 腕部相机 profile 强制 `fuse_beliefs=false`，尝试 override 时显式失败
- [ ] 现有 139 个测试全绿（含 2 个 hardware 用 `-m hardware` 隔离）
- [ ] `tests/test_review_regressions_v3.py` 新增覆盖本期发现的所有 review finding
- [ ] `docs/ARCHITECTURE.md`、`README.md` 更新到位（不更新 `ROADMAP.md` 除非新增条目）
- [ ] `data/calibration/hand_eye_top.json` 与 `data/calibration/cameras.json` 落地产出，含 `metrics`、`serial`、`frame_convention`

### 7.2 一行级 DoD（任务级会在 §21 详细列）

每个任务都包含：「Acceptance（可机器/可观测条件）」+「Verify（如何确认）」+「Files（哪些文件被触碰）」+ 「Dependencies（前序任务）」

---

## 8. Architecture & Data Flow

### 8.1 现有数据流（保留）

```text
camera stream  →  Frame
                 │ (depth_provider.ensure_depth 补 depth)
                 ▼
LockedDetector.detect()
                 │
                 ▼
WorldWatcher._tick(cam)  ←──  WatchedCamera.fuse
                 │
                 ▼ (extrinsics.cam_to_base + mask_to_points_cam)
                 ▼
BeliefStore.update(label, center, ...)   ←──  ObjectFix 重算
```

### 8.2 新增数据流

```text
        ┌─────────────────────────────────────────────┐
        │  src/wrc_demo/scripts/calib_top_orbbec.py  (CLI)          │
        │                                             │
        │   pyorbbecsdk ── Frame (bgr + aligned depth)│
        │     │                                        │
        │     ├── aruco_session.detect_marker_pose ──→ T_marker2cam  │
        │     └── wrc_demo.control.kinematics.Kinematics.fk(q_arm)  │
        │            │                                                   │
        │            ▼                                                   │
        │   CalibrationSample(T_gripper2base, T_marker2cam)             │
        │            │                                                   │
        │            ▼                                                   │
        │   handeye_solver.joint_solve(samples)  → CalibrationResult   │
        │            │                                                   │
        │            ▼                                                   │
        │   calib_dataset.save(result, "data/calibration/hand_eye_top.json")
        │                                                             │
        └─────────────────────────────────────────────────────────────┘

 运行时 ↓                                                              │
 ┌────────────┐                                                        │
 │ demo.py    │   build_runtime()                                     │
 │     │      │     camera profiles: orbbec_overhead, orbbec_wrist    │
 │     ▼      │     ┌──────────────────────────────────────────┐      │
 │ make_camera├──→ │ OrbbecCamera(type='orbbec')               │      │
 │     │      │     │   open/close/_grab  (CameraBase ABC)     │      │
 │     ▼      │     └──────────────────────────────────────────┘      │
 │ CameraStream│           │                                          │
 │     │      │           ▼                                          │
 │     ▼      │      Frame(rgb=BGR, depth_m=float32, K=3x3)            │
 │ WorldWatcher│           │                                          │
 │     │      │           ▼                                          │
 │     ▼      │     Extrinsics.from_config()  ←──  HAND_EYE_NPZ/JSON  │
 │ BeliefStore│     │                                                │
 └────────────┘     ▼                                                │
                  T_cam2base ──→ mask_to_points_cam + transform_points │
                              ─→ ObjectFix.position (in base frame)   │
```

### 8.3 模块边界

| 模块 | 唯一职责 | 不允许 |
|---|---|---|
| `perception/orbbec_camera.py` | `pyorbbecsdk` 适配 + BGR / 米制 / K 输出 | 不做 FK、不做 ArUco |
| `calibration/aruco_session.py` | ArUco 字典、检测、PnP、reprojection 残差 | 不写电机、不读 FK |
| `calibration/calib_dataset.py` | 数据类 + JSON I/O + schema_version 校验 | 不做数值求解 |
| `calibration/handeye_solver.py` | 联合 SE(3) 求解、Robust Kernel、cross-check | 不操作硬件 |
| `calibration/frame_convention.py` | base / gripper_end / cuRobo end_link 转换 + 一致性校验 | 不写 IO |
| `src/wrc_demo/scripts/calib_top_orbbec.py` | CLI glue：枚举 / 采集 / 求解 / verify | 不实现任何数值逻辑 |

---

## 9. ADR / Key Decisions

### ADR-1：单一标定工具路径，不复制 cuRobo 的 04b 段

- **Context**：现有 `curobo_rebot_demo/examples/04b_top_calibration.py` 是状态机成熟的可视化采集脚本；同项目内 `04c_verify_calibration.py` 是轻量验证。
- **Decision**：新建 `src/wrc_demo/scripts/calib_top_orbbec.py`，完全不依赖 cuRobo 的 `RebotMotionPlanner` / `RebotCuroboBridge`。cuRobo 04b 的 04b 状态机仅作参考，搬移实现用 wrc_demo 的 `Kinematics` + `SafeArm`。
- **Rationale**：跨项目复用要解决 frame convention 不一致、CLI 不一致、退出码语义不同。两套并行会形成"哪个真"的歧义。
- **Consequences**：
  - 加一份独立的工具维护成本
  - 状态机移植可能出现 1-2 个细节偏差（必须有 `test_calib_state_machine` 覆盖）
  - cuRobo 项目历史脚本原地保留，本次不对它做改动

### ADR-2：JSON 作为标定产物主格式，NPZ 仅存 K/D

- **Context**：NPZ 单值矩阵定位方便，但 diff/审计/回滚/schema 演化痛苦；JSON + zipped npz（仅 `K`、`D` 二进制）可读又好做 CI 校验。
- **Decision**：`CalibrationResult` 序列化到 `data/calibration/hand_eye_top.json`；K/D 用 `<hand_eye_top>.npz`（仅两个 array）。
- **Rationale**：
  - reviewer 用 `git diff` 直接看 metrics、serial、frame_convention 变化
  - metric 字段随 schema_version 演化，强制 `schema_version` 校验
  - NPZ 在 CI 里极难发现静默字段丢失
- **Consequences**：
  - wrc_demo 的 `Extrinsics.from_config` 必须支持两类文件
  - 公开函数 `wrc_demo.calibration.calib_dataset.save(result, base_path)` 同步写出 `.json` 和 `.npz`

### ADR-3：联合 SE(3) 求解，不沿用 OpenCV 把 Y 写死

- **Context**：`sdk/rebot_grasp_lib/calibration/hand_eye.py:181-187` 把 `T_cam2base` 通过 `T_g2b @ inv(T_t2c)` 计算并取 `candidates[0]`，这隐含 `T_marker2gripper = I`。实拍图显示 marker 装在白色支撑板上，与 URDF `end_link` 有明显偏移，假设不成立。
- **Decision**：在 `calibration/handeye_solver.py` 提供 `joint_solve(samples, robust=True, n_restart=8) -> (T_cam2base, T_marker2gripper, metrics)`；残差定义为 SE(3) 李代数上 `log(A_i X Y B_i^{-1})`；用 Levenberg-Marquardt + Huber 损失。
- **Rationale**：
  - 数学上严格，无 identity 假设
  - 能在 1-2 mm 噪声下稳定收敛（已合成验证）
  - outlier 不污染求解
- **Consequences**：
  - 新求解器需单元测试对比 `cv2.calibrateHandEye` 在 marker-as-gripper 情形（验证数值一致）
  - 与 cuRobo 04b 的 `cv2.calibrateHandEye` 完全解耦；旧的 `HandEyeCalibrator` 不动，本次不复制

### ADR-4：CalibrationSample 由 wrc_demo 的 Kinematics 来源

- **Context**：cuRobo 04b 用 cuRobo `RebotMotionPlanner.fk(q_arm)`；wrc_demo FK 走 `Pinocchio` on RS URDF + `joint_signs=[-1]*6`。
- **Decision**：标定工具统一调 `wrc_demo.control.kinematics.Kinematics(ee_frame='gripper_end', joint_signs=[-1]*6).fk(q_arm)`；不再使用 cuRobo fk。
- **Rationale**：末端 frame convention 必须与运行时一致；否则 T_cam2base 看上去合理但放回 wrc_demo 里点位会错。
- **Consequences**：
  - 在 `src/wrc_demo/scripts/calib_top_orbbec.py` 的 `main()` 中实例化 `Kinematics(...)`，cuRobo 配置文件不参与
  - 必须加 frame convention round-trip test

### ADR-5：腕部 profile 强制 `fuse_beliefs: false`

- **Context**：CLAUDE.md §Architecture 指出"相机只在 profile 有 `extrinsics:` 块时才默认参与 3D fusion"，但 `apps/demo.py:106` 在 profile 缺 `extrinsics` 时 `fuse = cfg.get('fuse_beliefs', 'extrinsics' in cfg)`。如果用户人为填 `extrinsics:` 但 NPZ 缺失，依然会出问题。
- **Decision**：腕部 `orbbec_wrist.yaml` 不带 `extrinsics` 块，并加注释"calibration pipeline not scheduled"；模型层在 demo.py 中新增更严格检查：若 profile 标记 `requires_calibration: true` 且无 `hand_eye_npz` 或 `.json`，则强制 `fuse=false` 并发 stderr 警告。
- **Rationale**：fail-closed 比允许错误的 3D belief 更可接受。
- **Consequences**：
  - 在 `apps/demo.py:build_runtime()` 里增 ~4 行逻辑；该改动必须配单元测试
  - 新增的强制逻辑影响其它未标定 profile（如 `uvc4k.yaml`），但它们的 `requires_calibration` 默认 false，无副作用

### ADR-6：所有 sw 依赖在 `OrbbecCamera` 函数体内 lazy import

- **Context**：CLAUDE.md 强制所有重依赖 lazy import；本仓库 mock stack 不依赖 `pyorbbecsdk`。
- **Decision**：`import pyorbbecsdk as ob` 仅出现在 `OrbbecCamera.open()` 函数体内；测试用 `FakeOrbbecSDK` 注入 `pyorbbecsdk` 模块路径。
- **Rationale**：`pytest tests/ -q` 默认不会因缺少 `pyorbbecsdk` 而失败。
- **Consequences**：
  - `tests/conftest.py` 加 fixture `fake_orbbec_sdk(monkeypatch)`：替换 `sys.modules['pyorbbecsdk']` 为 `FakeSDK`
  - 真机使用需要在 `.demo` venv 装 `pyorbbecsdk`，setup 走 README

### ADR-7：Feature flag 在 profile YAML，而不在代码

- **Context**：CLAUDE.md 强调"profile 决定 behavior"；feature flag 走 `--enable-features` 不符合现有约定。
- **Decision**：feature flag 用 profile YAML key 表达（如 `requires_calibration: true`）；运行时依据 profile 自动判定。
- **Rationale**：保持 demo 的"profile 驱动"模型一致。
- **Consequences**：所有新行为都有 profile-level key，`make_camera()` / `build_runtime()` 不增加 CLI 参数。

### ADR-8：CLI 入口路径规范（统一在 `src/wrc_demo/scripts/`）

- **Context**：标定工具既要被 `python -m` 调用（必须落在 `src/wrc_demo/scripts/`），又要避免与仓库根 `scripts/*.sh`（已有的 `dashboard_runner.py`、`hermes_demo.sh`、`serve_graspgenx.sh` 等）混淆。PLAN §4 / §11.1 / §20 之间早期版本把文件位置定在仓库根 `scripts/`，但所有 `python -m wrc_demo.scripts.calib_top_orbbec` 调令都指向 `src/wrc_demo/scripts/`——两者互斥，本 ADR 落槌。
- **Decision**：
  - `src/wrc_demo/scripts/calib_top_orbbec.py` 是 CLI 入口；同级建 `__init__.py` 让 `wrc_demo.scripts` 成为合法包。
  - 调令固定为 `python -m wrc_demo.scripts.calib_top_orbbec ...`；不写到根 `scripts/`。
  - 仓库根 `scripts/` 继续留给 shell / 辅助 Python 脚本（`*.sh`、`dashboard_runner.py`、`serve_graspgenx.sh` 等）。
- **Rationale**：避免「同一个文件名，两套入口路径」歧义；并让 `python -m` 路径与 typing / test 收集打通。
- **Consequences**：
  - §4 / §11.1 / §20 / §19 / ADR-1 / ADR-4 等所有引用都必须统一到 `src/wrc_demo/scripts/calib_top_orbbec.py`。
  - `src/wrc_demo/scripts/__init__.py` 必须存在（即使为空）；列入 §11.1 新增文件清单。

---

## 10. 实施阶段与依赖（垂直切片）

> 每个阶段都"端到端可见"：阶段末能在 `--cameras orbbec_overhead --arm mock`（offline 路径） 或 `--cameras orbbec_overhead --arm rebot_rs`（实机）至少跑一项用户可见的命令。

### 依赖图

```text
   ┌─────────────────────────────────────────┐
   │ S0  calibration 工具骨架（无硬件）     │
   │     - calib_dataset 数据类 + JSON I/O   │
   │     - frame_convention utilities         │
   │     - aruco_session（轻量 cv2 封装）    │
   │     - handeye_solver（含合成测试）       │
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │ S1  OrbbecCamera backend + profile      │
   │     - orbbec_camera.py（fake SDK 可测） │
   │     - orbbec_overhead.yaml/orbbec_wrist.yaml│
   │     - camera_base.py make_camera +'orbbec'│
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │ S2  src/wrc_demo/scripts/calib_top_orbbec.py（CLI）  │
   │     - 状态机移植（04b 风格）           │
   │     - 用 wrc_demo Kinematics           │
   │     - 用 S0 handeye_solver 求解        │
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │ S3  Extrinsics loader 多 schema         │
   │     - 新 schema (JSON .json + npz)     │
   │     - cuRobo npz                       │
   │     - baseline T_result npz            │
   │     - serial 一致性校验                │
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │ S4  demo.py 集成                       │
   │     - required_calibration 强制 fuse=F │
   │     - WorldWatcher 串号→binding 校验   │
   │     - 或bbec_overhead/wrist test       │
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │ S5  Rig verify (hardware bring-up)    │
   │     - calibration verify 已知base点   │
   │     - depth scale / plane_z test       │
   │     - dashboard 双画面                │
   └─────────────────────────────────────────┘
```

### 阶段定义

| 阶段 | 完成定义（用户可见） | 验收 |
|---|---|---|
| S0 | `pytest tests/test_handeye_solver.py` 全过；合成 X、Y、metrics 正确 | 单元测试 |
| S1 | `pytest tests/test_orbbec_camera.py` 全过；`fake_orbbec_sdk` 下流过一帧 | 单元测试 |
| S2 | 离线 `--list` / 离线 `joint_solve_demo` 跑通 | 集成测试 |
| S3 | `Extrinsics.from_config` schema 矩阵测试全过 | 单元测试 |
| S4 | `--cameras orbbec_overhead --arm rebot_rs` 跑 `demo.py --interactive`，WorldWatcher 不抛 fuse error | 集成测试 |
| S5 | 实机 cross-check ≤10 mm；dashboard 双相机视频可见；depth plane_z 误差 ≤5 mm | hardware |

### 阶段间 Checkpoint

每阶段结束，提交一次 commit，进入 `tests/test_review_regressions_v3.py` 作"已完成 + 测试覆盖"标签。建议在 S0、S2、S4 后加一段人审。

---

## 11. File-level Change List

> 路径 + 变更类型 + 描述。**所有变更可单元测试覆盖**。

### 11.1 新增文件

| 路径 | 用途 |
|---|---|
| `PLAN.md` | 本文件 |
| `configs/cameras/orbbec_overhead.yaml` | 顶部 profile，serial 经 binding 自动填 |
| `configs/cameras/orbbec_wrist.yaml` | 腕部 profile，强制 `fuse_beliefs: false` |
| `src/wrc_demo/perception/orbbec_camera.py` | `OrbbecCamera(CameraBase)` |
| `src/wrc_demo/calibration/__init__.py` | 模块入口 |
| `src/wrc_demo/calibration/calib_dataset.py` | `CalibrationSample` / `CalibrationResult` / `CameraBinding` + JSON I/O |
| `src/wrc_demo/calibration/handeye_solver.py` | 联合 SE(3) 求解 + Huber + 留一交叉验证 |
| `src/wrc_demo/calibration/aruco_session.py` | `cv2.aruco` 字典、detect、PnP、reprojection |
| `src/wrc_demo/calibration/frame_convention.py` | base / gripper_end / cuRobo end_link 转换 + 一致性测试 |
| `src/wrc_demo/scripts/calib_top_orbbec.py` | CLI 入口 |
| `tests/test_orbbec_camera.py` | backend 单测（fake SDK） |
| `tests/test_handeye_solver.py` | 求解器单测（合成） |
| `tests/test_extrinsics_loader.py` | loader schema 矩阵 |
| `tests/test_frame_convention.py` | fk round-trip |
| `tests/test_calib_dataset.py` | 数据类 JSON 持久化 |
| `tests/test_calib_session.py` | 状态机（dry-run，注入 fake sources） |
| `data/calibration/.gitkeep` | 空目录占位 |
| `src/wrc_demo/scripts/__init__.py` | 让 `wrc_demo.scripts` 成合法包（ADR-8） |
| `tests/test_review_regressions_v3.py` | 仅当发现 v3 review 缺陷时新增 |
| `tests/test_aruco_session.py` | 新增：合成图像 ArUco 检测 + PnP 单测（Task 0.3） |

### 11.2 修改文件（最小动作）

| 路径 | 修改内容 | 行数估计 |
|---|---|---|
| `src/wrc_demo/perception/camera_base.py` | 在 `make_camera()` 添加 `orbbec` 分支（lazy import） | +6 |
| `src/wrc_demo/perception/grounding.py` | `Extrinsics.from_config` 支持 cuRobo npz (`T_cam2base`/`T_cam2gripper`) + 新 JSON schema；加 schema_version / serial / frame_convention 校验 | +30 / -5 |
| `src/wrc_demo/apps/demo.py` | `build_runtime()` 中：profile 标识 `requires_calibration` 时强制 `fuse=False` 若缺 npz/json；WorldWatcher 启动时检查 serial 一致性 | +12 / -0 |
| `docs/ARCHITECTURE.md` | 在 perception 段落加 orbbec_camera 与 calibration/ 模块；脚注"Orbbec backend lazy 引入" | +30 |
| `README.md` | "Onsite bring-up checklist"段 hand-eye 段改指本计划工具 | 修改一段 |

### 11.3 不动的文件（确认）

- `src/wrc_demo/control/{kinematics,arm_base,rebot_rs_arm}.py` 不动（仅作为外部 import）
- `src/wrc_demo/safety/harness.py`、`SafeArm` 不动
- `src/wrc_demo/perception/{stream,world,depth_provider,detector}.py` 不动
- `src/wrc_demo/skills/runtime.py`、`apps/mcp_server.py` 不动
- `src/wrc_demo/agent/*` 不动
- 所有现有 profile YAML 不动
- 所有现有测试不动（除非新增 review findings）
- 不复用 `curobo_rebot_demo` 或 `rebot_grasp` 任何代码作为依赖（仅作参考）

---

## 12. Schemas

### 12.1 camera profile `orbbec_overhead.yaml`

```yaml
# Orbbec Gemini 2 顶部相机 — eye-to-hand, 已标定主操作相机
type: orbbec
serial: ""                      # 由 calib_top_orbbec.py --bind 自动回填；或手工填
width: 1280
height: 720
fps: 30
min_depth_m: 0.25
max_depth_m: 2.5
warmup_frames: 15

# depth-color align: SDK 硬件对齐，输出 BGR 与 depth 同分辨率
align_depth_to_color: true

extrinsics:
  source: "data/calibration/hand_eye_top.json"   # 或 .npz；若不存在则 warn
  mode: eye_to_hand
  # T 字段在 JSON loader 中读取后覆盖；此处仅占位避免 schema 漂移
  T:
    - [1, 0, 0, 0]
    - [0, 1, 0, 0]
    - [0, 0, 1, 0]
    - [0, 0, 0, 1]

requires_calibration: true     # 用于 demo.py 强制 fuse 策略
fuse_beliefs: true             # 顶部相机参与 3D 融合
```

### 12.2 camera profile `orbbec_wrist.yaml`

```yaml
# Orbbec Gemini 2 腕部相机 — 辅助视角，本期不标定、不参与 3D 融合
type: orbbec
serial: ""                      # 由 calib_top_orbbec.py --bind 自动回填
width: 1280
height: 720
fps: 30
min_depth_m: 0.15
max_depth_m: 1.5
warmup_frames: 10

align_depth_to_color: true

# 不填 extrinsics；requires_calibration: true + 无源 = 强制 fuse=false
requires_calibration: true
fuse_beliefs: false             # 第一阶段硬约束；配合 demo.py 强制检查
```

### 12.3 `hand_eye_top.json` schema

```json
{
  "schema_version": 1,
  "frame_convention": "wrc_demo",
  "frame_convention_note": "base +x away robot, +y left; TCP = gripper_end; joint_signs = [-1]*6",
  "created_at": "2026-07-20T15:32:11Z",
  "method": "joint_se3_lm_huber",
  "mode": "eye_to_hand",
  "camera_serial": "AY3Z331006L",
  "camera_index_when_bound": 0,
  "marker_size_m": 0.10,
  "aruco_dict": "4x4_50",
  "aruco_marker_id": 0,
  "n_samples": 38,
  "n_inliers": 36,
  "n_outliers": 2,
  "outlier_indices": [12, 27],
  "T_cam2base": [[...4x4...]],
  "T_marker2gripper": [[...4x4...]],
  "metrics": {
    "translation_rmse_mm": 4.7,
    "translation_max_err_mm": 9.8,
    "rotation_rmse_deg": 0.62,
    "rotation_max_err_deg": 1.4,
    "aruco_reprojection_rmse_px": 0.43,
    "known_point_rmse_mm": 7.1,
    "known_point_max_err_mm": 13.0,
    "lm_iterations": 38,
    "lm_converged": true
  },
  "samples_for_repro": [
    {
      "pose_label": "auto pose 1",
      "T_gripper2base": [[...4x4...]],
      "T_marker2cam": [[...4x4...]],
      "residual_translation_mm": 3.1,
      "residual_rotation_deg": 0.21
    }
  ]
}
```

### 12.4 `cameras.json` schema（替代 cuRobo 同名字段）

```json
{
  "schema_version": 1,
  "updated_at": "2026-07-20T15:32:11Z",
  "top":  {"serial": "AY3Z331006L", "index_when_bound": 0},
  "wrist": {"serial": "AY3794302V9", "index_when_bound": 1}
}
```

### 12.5 backward-compat: 旧的 cuRobo `hand_eye_top.npz`

仍包含 keys：`T_cam2base`, `T_cam2gripper`, `n_samples`, `method`, `camera_serial`, `marker_size_m`, `mode`（部分）。`Extrinsics.from_config` 必须识别并报告 `frame_convention=curobo`（兼容性），不允许 silent 使用其 `T_cam2base` 作为 wrc_demo `T_cam2base`，因为 frame 不一致。要在 profile yaml 用 `frame_convention_override: wrc_demo` 显式确认，或重新采集。

---

## 13. 手眼标定数学模型

### 13.1 方程（Joint Solve）

记：

- \(X = T_{\text{cam} \to \text{base}} \in SE(3)\)
- \(Y = T_{\text{marker} \to \text{gripper}} \in SE(3)\)
- 每个样本 \(i\)：\(G_i = T_{\text{gripper} \to \text{base}}^{(i)}}\)（Pinocchio FK），\(M_i = T_{\text{marker} \to \text{cam}}^{(i)}}\)（ArUco PnP）

目标：

\[
G_i \, Y \approx X \, M_i \quad \forall i
\]

定义残差（李代数）：

\[
r_i = \log\left( (G_i Y)^{-1} (X M_i) \right) \in \mathbb{R}^6
\]

总损失：

\[
\mathcal{L}(X, Y) = \sum_i \rho(\|r_i\|), \quad \rho(\cdot) = \text{Huber}(\delta=1.5\text{mm})
\]

### 13.2 求解

- 参数化：\(X\) 用 \((t, \text{rotvec})\)，\(Y\) 同上 → 共 12 维
- Levenberg-Marquardt（`scipy.optimize.least_squares` 或自实现）
- 初值：\(X_0 = G_1 M_1^{-1}\)（无 Y 假设下的样本 0），\(Y_0 = I\)
- N restarts = 8（在 SO(3) 单位球上 8 个正多面体顶点为初值扰动）
- 离群判定：残差 > 3 × MAD(iter) → out；最少保留 60% 样本作为 inliers
- 收敛判据：相对参数变化 < 1e-8 或 50 iter 内无显著改进

### 13.3 验证

1. **ArUco 重投影 RMSE**：用 \(Y X^{-1}\) 与每个样本 \(M_i^{-1}\) 推导的角点像素预测 vs 实际角点
2. **已知基坐标 cross-check**：在桌面 3-5 个已知坐标点（用尺规标定），让 marker 在某 pose 上面对该点，目标位置 vs `X^{-1} @ G_i @ Y` 反推出的 marker 中心坐标
3. **留一交叉验证**：N-1 样本求解 → 在被剔除样本上算残差 → 取 mean / max

### 13.4 末端 frame 与 base frame

- **wrc_demo Kinematics**：\(G_i = \text{Pinocchio.fk}(q_i; \text{ee\_frame} = \text{gripper\_end}, \text{joint\_signs} = [-1]\times 6)\)
- **cuRobo Kinematics**：\(G_i^{\text{curobo}} = \text{cuRobo fk}(q_i; \text{ee\_frame} = \text{end\_link})\)，且基座下相对原点可能与 wrc_demo 不同；这两者间的常量变换表达为 \(S\)（**必须在 Task 0.0 中实测并固定**，详见下方硬约束）
- 若 cuRobo 4b 的样本被历史保留用于旧工程，必须显式标注 `frame_convention=curobo` 并禁止被 wrc_demo runtime 加载使用

> **S 矩阵实测硬约束（Task 0.0 acceptance）**：`test_frame_convention.py::test_S_constant_under_random_q` 对 8 个随机 \(q\) 同时调 cuRobo FK（若有）与 wrc_demo `Kinematics.fk`，计算 \(S_q = \text{inv}(G_q^{\text{curobo}}) @ G_q^{\text{wrc_demo}}\)，要求 8 个 \(S_q\) 与其均值的 Frobenius 偏差 < 1e-6。**S 测得即定**，作为模块常量存 `frame_convention.S`；触碰 cuRobo FK 或 wrc_demo URDF 任一者即重跑此测试。**Phase 0 不出门直到此测试通过。**

### 13.5 Marker 与 gripper 关系

- 默认 marker 100 mm ArUco 4x4_50 ID 0，平贴于 gripper 上方白板
- `T_marker2gripper` 包括白板的安装偏移 + 角度，应通过求解自动获得
- 真正安装一旦变化（拆装白板），必须重新 calibration

---

## 14. Schema 强制与运行时检查

`Extrinsics.from_config(cfg, fk_tcp2base=None)` 调用顺序：

1. 若 profile 有 `extrinsics.source` → 加载 → 校验 `schema_version` 与 `frame_convention=wrc_demo`
2. 若 profile 有 `extrinsics.hand_eye_npz` → 加载 → 检查 keys（`T_cam2base` / `T_cam2gripper` / `mode` / `camera_serial` 等）
3. 若 profile 有 `extrinsics.T` → 直接采用（仅 placeholder / 测试用）
4. 任一路径缺失且 profile `requires_calibration: true` → 警告 stderr；运行时：
   - `cfg.extrinsics` 标记为 `valid=False`
   - `build_runtime()` 强制 `fuse = False` 并 stderr 提示
   - 不抛异常（兼容 dev 流），但记一行 `runs/runtime_extrinsics.jsonl`
5. serial mismatch：profile `serial` vs npz `camera_serial` → stderr warning，但优先采用 npz（以避免因为换机后 binding 未刷新阻塞）

---

## 15. Test Matrix & Hardware Acceptance

### 15.1 离线测试矩阵（默认集）

| 测试 | 覆盖 | 期望 |
|---|---|---|
| `test_orbbec_camera_lazy_import` | lazy 行为 | pass（无 `pyorbbecsdk` 时 init 不抛） |
| `test_orbbec_camera_grab_calls_align` | SDK 调用契约 | pass |
| `test_orbbec_camera_grab_returns_metric_depth` | depth 单位 | pass |
| `test_orbbec_camera_fails_closed_on_sdk_error` | error 传递 | 抛 CameraError 30 次后抛给上层 |
| `test_handeye_solver_perfect_data` | 数值正确性 | RMSE < 1e-9 |
| `test_handeye_solver_noise_1mm` | 噪声稳定 | RMSE 增长 ≤ 5 mm |
| `test_handeye_solver_robust_to_outliers` | 鲁棒 | 20% 离群 → 收敛、inlier 残差 ≤ 5 mm |
| `test_handeye_solver_compare_cv2_handeye_when_marker_eq_gripper` | 旧路径回归 | 数值一致（< 0.1 mm） |
| `test_extrinsics_loader_curobo_npz_with_serial` | schema 支持 | 通过 |
| `test_extrinsics_loader_baseline_T_result` | 旧 schema | 通过 |
| `test_extrinsics_loader_new_json` | 新 schema | 通过 |
| `test_extrinsics_loader_serial_mismatch_warning` | 一致性 | 警告但仍加载 |
| `test_extrinsics_loader_missing_npz_required_calibration` | fail-closed | `valid=False` |
| `test_wrist_profile_force_fuse_false` | 强制约束 | fuse=False |
| `test_frame_convention_wrc_demo_fk_roundtrip` | fk 不变量 | pass |
| `test_calib_dataset_save_load_roundtrip` | JSON 序列化 | 完整恢复 |
| `test_build_runtime_orbbec_wrist_no_extrinsics_loads_with_fuse_false` | 集成 | pass |
| `test_build_runtime_orbbec_overhead_no_extrinsics_warns_but_runs` | 集成 | pass |

### 15.2 hardware 测试矩阵

| 测试 | 命令 | 判定 |
|---|---|---|
| `test_hardware_smoke_orbbec_enumerate` | `python -m wrc_demo.scripts.calib_top_orbbec --list` | 2 设备 serials 与 binding 一致 |
| `test_hardware_smoke_orbbec_first_frame_top` | 顶部相机 open → 1s 内拿帧 | 拿到 RGB + 米制 depth |
| `test_hardware_smoke_orbbec_fuse_beliefs` | `--cameras orbbec_overhead,orbbec_wrist` build → WorldWatcher 启动 | 不抛 `extrinsics invalid for fuse=true` |
| `test_hardware_smoke_orbbec_calib_verify_known_points` | 见 §16 | 平均 ≤ 10 mm |
| `test_hardware_smoke_can0_read_param` | 已有 | pass |

### 15.3 验收门槛

| 指标 | 目标值 | 失败处理 |
|---|---|---|
| `translation_rmse_mm` | ≤ 5 mm | 拒收，重采 |
| `translation_max_err_mm` | ≤ 15 mm | 拒收 |
| `rotation_rmse_deg` | ≤ 1.0° | 拒收 |
| `rotation_max_err_deg` | ≤ 3.0° | 拒收 |
| `aruco_reprojection_rmse_px` | ≤ 1.0 px | 拒收 |
| `known_point_rmse_mm` | ≤ 10 mm | 拒收 |
| `known_point_max_err_mm` | ≤ 20 mm | "demo 级宽松失败阈值" |
| `n_inliers / n_samples` | ≥ 80% | 提示增采 |
| schema_version 正确 | 是 | 否：拒收 |

> 30 mm 仅作 "demo 级宽松失败阈值"，不进入 acceptance gate。

---

## 16. Hardware Bring-up Checklist（实机验证清单）

> 步骤顺序强依赖；任何步骤 fail 立即停止，回退上一阶段。

### 16.1 准备
- [ ] `.demo` venv 装好 `pyorbbecsdk`（`uv pip install --python "$PY" pyorbbecsdk`）
- [ ] `sudo ip link set can0 up type can bitrate 1000000`
- [ ] 停止 motorbridge-gateway / Studio
- [ ] 拔掉 wrist Orbbec USB；只保留 top Orbbec
- [ ] 顶部相机三脚架位置确认（与上次标定前一致）
- [ ] ArUco 4x4_50 ID 0 100 mm marker 平贴 gripper 上方白板，正面朝上

### 16.2 设备枚举 & binding
- [ ] `python -m wrc_demo.scripts.calib_top_orbbec --list`
- [ ] 检查 serial = `AY3Z331006L`（与历史一致）或记录新 serial
- [ ] `python -m wrc_demo.scripts.calib_top_orbbec --bind --role top`
- [ ] 确认 `data/calibration/cameras.json` 写入

### 16.3 采集
- [ ] `python -m wrc_demo.scripts.calib_top_orbbec`
- [ ] 自动采集 ≥30 个有效样本（目标 38 个）
- [ ] 终端 / 仪表盘显示每个样本的 inlier 残差
- [ ] 实时显示 marker 检测框 + pose 编号 OSD

### 16.4 求解 + 验证
- [ ] 求解器自动收敛；产出 `data/calibration/hand_eye_top.json` + `.npz`
- [ ] 查看 `metrics.translation_rmse_mm ≤ 5`
- [ ] 跑 cross-check：
  - [ ] 选 3-5 个已知 base 坐标点（在桌面用尺规标，文档中给出常见点）
  - [ ] `python -m wrc_demo.scripts.calib_top_orbbec --verify --known-points "..."`
  - [ ] `known_point_rmse_mm ≤ 10`

### 16.5 应用加载
- [ ] `configs/cameras/orbbec_overhead.yaml` 写入 `extrinsics.source: "data/calibration/hand_eye_top.json"`
- [ ] `python -m wrc_demo.apps.demo --interactive --cameras orbbec_overhead --arm rebot_rs`
- [ ] 在 StreamServer dashboard 看到顶部相机画面 + 检测 overlay
- [ ] `runtime.world_watcher.stats()['paused'] is False`，且 `beliefs.find(<color>)` 在 deadline 内能拿到（前提：桌上有物体）；此处不引入 `fuses` 字段，避免动 WorldWatcher 现有 stats() schema
- [ ] belief：摆一个已知色块在桌面，跑 `python -m wrc_demo.apps.demo --task "look at the table and report what objects you see"`

### 16.6 双相机加入
- [ ] 插上腕部 Orbbec USB
- [ ] `python -m wrc_demo.scripts.calib_top_orbbec --bind --role wrist` 写入
- [ ] `python -m wrc_demo.apps.demo --interactive --cameras orbbec_overhead,orbbec_wrist --arm rebot_rs`
- [ ] dashboard 两路视频可见
- [ ] `cfg.cameras[1].fuse == False`（打印 stats 验证）

### 16.7 失败回滚

| 故障 | 回滚方案 |
|---|---|
| `pyorbbecsdk` 装失败 | 重新激活 `.demo` venv 后 `uv pip install pyorbbecsdk`；或临时切到 `--cameras isaac` 在 sim 跑 |
| 标定 RMSE > 5 mm | 不更新 yaml；标记为旧 npz；保留旧 yaml extrinsics 备份 |
| known_point 验证 > 10 mm | 重采；若连续 3 次失败，回复到 issue 队列 |
| WorldWatcher 抛 extrinsics 失败 | 检查 yaml `extrinsics.source` 路径；yaml serial 与 binding 一致；配置 `requires_calibration` |
| 真机 RS 启动失败 | 不动本次更改；按 CLAUDE.md "bring-up order" 走 `pytest -m hardware` 单独排 |

---

## 17. 风险、回滚、兼容

### 17.1 风险表

| 风险 | 概率 | 影响 | 缓释 |
|---|---|---|---|
| `pyorbbecsdk` 装包失败 / ABI 冲突 | 中 | 高 | CI 中只跑 fake SDK；装包失败回退 `--cameras isaac` |
| cuRobo 与 wrc_demo frame convention 不一致算不出 T | 中 | 高 | 联合求解吸收常数偏移；frame_convention test 提前解 |
| 求解器在离群点下不收敛 | 低 | 中 | Huber + N restart + 至少 80% inlier 退出策略 |
| 真实 marker 安装偏移被忽略 | 中 | 中 | 强制 Y 不为单位阵；旧 baseline npz 必须重采 |
| 顶部相机物理位置松动 | 中 | 高 | verify 已知 base 点一定要做；30 mm 阈值 |
| 标定工具 CI 跑超时（占用 USB） | 低 | 低 | CI 跳过 `--list` 之外的硬件路径；用 `-m hardware` 隔离 |
| `Extrinsics.from_config` 旧行为破坏 | 低 | 高 | schema 矩阵测试覆盖；保留 `T_result` 老路径 |
| `demo.py` 加 `requires_calibration` 策略误伤其它 profile | 低 | 中 | 默认值 default false；只对显式声明的 profile 生效 |

### 17.2 回滚策略

- **配置回滚**：保留每次产出的 `hand_eye_top_<timestamp>.json`，git tag 旧版；yaml 用 `extrinsics.source` 切回旧产物
- **代码回滚**：`src/wrc_demo/perception/camera_base.py:make_camera()` 的 `orbbec` 分支用 feature flag（profile `requires_calibration`）保护，紧急情况下删除 profile 即回到原 demo
- **测试回滚**：`tests/test_review_regressions_v3.py` 新增的 failing test 不会阻塞 CI（默认 `-m 'not hardware'`），逐个修

### 17.3 兼容性

- **cuRobo NPZ 兼容**：`Extrinsics.from_config` 加载时打 `frame_convention=curobo` 标记；runtime 校验不通过则不参与 `fuse_beliefs=true`（仅 stream）
- **baseline NPZ 兼容**：`T_result` 单键加载仍可用，提示"recalibration recommended"
- **CLAUDE.md 兼容**：所有 convention (Frame / time / joint_signs / error taxonomy / lazy import / gripper open_pos 等) 严格遵守

---

## 18. 工作量估算与推荐实施顺序

| 阶段 | 任务数 | 估算工时 | 关键不确定 |
|---|---|---|---|
| S0 算法骨架 | 4 | 2 人天 | 求解器数值稳定性 |
| S1 Backend | 3 | 1.5 人天 | pyorbbecsdk API 细节 |
| S2 CLI 工具 | 4 | 2 人天 | FK / motion 集成 |
| S3 Loader | 3 | 1 人天 | schema 演化 |
| S4 集成 | 3 | 1 人天 | demo.py 改动安全性 |
| S5 验证 | 4 | 1.5 人天（含硬件） | 实机时间 |
| Docs / 杂项 | - | 1 人天 | - |
| **合计** | **~21 个任务** | **~10 人天** | 含硬件 1 天 |

### 实施顺序（高风险早做）

1. **S0.0 frame_convention utilities**（最早；任何 FK / cuRobo 差异都在此处暴露）
2. **S0.1 handeye_solver 单元测试**（合成测试先于实采）
3. **S0.2 calib_dataset JSON I/O**
4. **S1.x backend（含 fake SDK 测试）**
5. **S3.x Extrinsics loader**
6. **S2.x CLI 工具**
7. **S4.x demo.py 集成**
8. **S5.x 实机验证**

---

## 19. 复用 / 废弃 / 新增 清单

| 类型 | 项 | 路径（参考来源） | 处理 |
|---|---|---|---|
| **复用（仅作参考）** | 视觉采集流程 04b | `curobo_rebot_demo/examples/04b_top_calibration.py` | 不复制；按 ADR-1 重写 |
| **复用（仅作参考）** | pose 列表 | `curobo_rebot_demo/rebot_curobo/cameras/calibration.py:TOP_CALIB_POSES` | 复制 38 个 pose 到 `calib_dataset.py`，注明来源 |
| **复用（仅作参考）** | ArUco 工具 | `curobo_rebot_demo/sdk/rebot_grasp_lib/calibration/aruco_utils.py` | 重写 `aruco_session.py`（保持简洁 + frame convention 注入） |
| **复用（仅作参考）** | Orbbec SDK wrapper | `rebot_grasp/drivers/camera/orbbec_gemini2.py` | 重写 `orbbec_camera.py`（CLAUDE.md 严禁 vendor import） |
| **废弃** | cuRobo `HandEyeCalibrator` | `curobo_rebot_demo/sdk/.../hand_eye.py` | 不复用，按 ADR-3 重写求解器 |
| **废弃** | 历史 `hand_eye_top.npz` | `curobo_rebot_demo/data/calibration/hand_eye_top.npz` | 不导入；新数据覆盖 |
| **废弃** | `--cameras l515` 入口 | wrc_demo 原 l515/d435i 等 profile | 不动；本期不参与 |
| **新增** | `OrbbecCamera`, `calibration/`, src/wrc_demo/scripts/calib_top_orbbec.py, orbbec_overhead/wrist.yaml | 本规划 | — |

---

## 20. 最终项目架构（修改后）

> 标 `[NEW]` = 本规划新增；标 `[MOD]` = 本规划修改；未标 = 不动。

```text
wrc_demo/
├── PLAN.md                                           [NEW] 本文件
├── README.md                                         [MOD] onsite bring-up 改指本计划
├── CLAUDE.md                                         不动
├── docs/
│   ├── ARCHITECTURE.md                               [MOD] 加 orbbec + calibration 模块描述
│   ├── ROADMAP.md                                    不动
│   └── ... (现有不变)
│
├── configs/
│   ├── demo.yaml                                     不动
│   ├── cameras/
│   │   ├── d435i.yaml                                不动
│   │   ├── d455f.yaml                                不动
│   │   ├── isaac.yaml / isaac_side.yaml / isaac_wrist.yaml    不动
│   │   ├── l515.yaml                                 不动
│   │   ├── mock.yaml                                 不动
│   │   ├── uvc.yaml / uvc4k.yaml                     不动
│   │   ├── orbbec_overhead.yaml                      [NEW]
│   │   └── orbbec_wrist.yaml                         [NEW]
│   ├── arms/                                         不动
│   └── llm/                                          不动
│
├── data/                                             不动（除 calibration 子目录）
│   └── calibration/                                  [NEW]
│       ├── .gitkeep                                  [NEW]
│       ├── hand_eye_top.json                         [NEW, runtime 产出]
│       ├── hand_eye_top.npz                          [NEW, runtime 产出]
│       └── cameras.json                              [NEW, runtime 产出]
│
├── src/wrc_demo/
│   ├── __init__.py                                   不动
│   ├── types.py                                      不动
│   ├── config.py                                     不动
│   ├── agent/                                        不动
│   ├── apps/
│   │   ├── demo.py                                   [MOD] build_runtime + ~12 行 requires_calibration
│   │   ├── mcp_server.py                             不动
│   │   ├── stream_server.py / live_view.py / viewer.py / record.py   不动
│   ├── calibration/                                  [NEW]
│   │   ├── __init__.py                               [NEW]
│   │   ├── calib_dataset.py                          [NEW] 数据类 + JSON I/O
│   │   ├── handeye_solver.py                         [NEW] SE(3) 联合求解
│   │   ├── aruco_session.py                          [NEW] ArUco 字典 + PnP
│   │   └── frame_convention.py                       [NEW] base/gripper_end 转换
│   ├── control/
│   │   ├── arm_base.py / isaac_arm.py / kinematics.py / lazy_arm.py / mock_arm.py / rebot_rs_arm.py / usd_model.py   不动
│   ├── grasping/                                     不动
│   ├── memory/                                       不动
│   ├── perception/
│   │   ├── __init__.py                               不动
│   │   ├── camera_base.py                            [MOD] make_camera +6 行 orbbec 分支
│   │   ├── colors.py                                 不动
│   │   ├── depth_provider.py                         不动
│   │   ├── detector.py                               不动
│   │   ├── grounding.py                              [MOD] Extrinsics.from_config 多 schema
│   │   ├── isaac_camera.py                           不动
│   │   ├── mock_camera.py / opencv_camera.py / realsense_camera.py    不动
│   │   ├── orbbec_camera.py                          [NEW]
│   │   ├── stream.py                                 不动
│   │   ├── vlm_ground.py                             不动
│   │   └── world.py                                  不动
│   ├── safety/                                       不动
│   ├── sim/                                          不动
│   └── skills/                                       不动
│
├── scripts/                                         不动（shell / 辅助 Python：dashboard_runner.py、setup_agents.py、serve_graspgenx.sh 等）
└── src/wrc_demo/scripts/                            # 见 ADR-8
    ├── __init__.py                                   [NEW] 空文件，让 wrc_demo.scripts 成合法包（ADR-8）
    └── calib_top_orbbec.py                           [NEW] CLI 入口（python -m wrc_demo.scripts.calib_top_orbbec）

└── tests/
    ├── conftest.py                                   [MOD (小幅)] fake_orbbec_sdk fixture
    ├── test_calib_dataset.py                         [NEW]
    ├── test_calib_session.py                         [NEW]
    ├── test_extrinsics_loader.py                     [NEW]
    ├── test_frame_convention.py                      [NEW]
    ├── test_aruco_session.py                         [NEW] ArUco 检测 + PnP（Task 0.3）
    ├── test_handeye_solver.py                        [NEW]
    ├── test_hardware_smoke.py                        [MOD (小幅)] 加 orbbec hardware sub-tests
    ├── test_orbbec_camera.py                         [NEW]
    ├── test_orchestrator_e2e.py / test_kinematics.py / test_grasping.py / test_live_view.py / test_memory.py / test_mcp_server.py / test_persistence_loop.py / test_safety.py / test_setup_agents.py / test_streaming_fastpath.py / test_turboquant.py / test_usd_model.py / test_perception.py / test_llm_and_library.py / test_isaac_bridge.py   不动
    └── test_review_regressions.py / _v2.py           不动；v3 [NEW] 仅本期 review 发现新缺陷时追加
```

**模块依赖图（按本规划修改后的稳态）**：

```text
                 ┌─────────────────────────┐
                 │ apps/demo.py            │ (composition root)
                 └─────────┬───────────────┘
                           │
       ┌───────────────────┼────────────────────┐
       ▼                   ▼                    ▼
perception/        control/               agent/
   │                (Kinematics,            │
   │                 SafeArm,               │
   │                 harness)               │
   │                   │                    │
   ▼                   ▼                    ▼
calibration/   ←─ imports `Kinematics.fk` (only) & `SafeArm` (only for motion)
   │
   ├─ calib_dataset:    {CalibrationSample, CalibrationResult, CameraBinding, JSON I/O}
   ├─ handeye_solver:   {joint_solve(samples), verify_known_points, LOO}
   ├─ aruco_session:    {make_dict, detect, reprojection_residual}
   └─ frame_convention: {wrc_demo_fk, curobo_fk_to_wrc_demo_transform}
                           │
                           └─→ src/wrc_demo/scripts/calib_top_orbbec.py   (CLI glue)
```

**重要不变量（修改后再次强调）**：

- `calibration/` 模块只 `import` `perception`, `control` 中的 **类型 / 函数**，**不引入新顶层依赖**（`cv2`、`numpy`、`scipy.optimize.least_squares` 已含于 `.demo`）
- `orbbec_camera.py` 仅在 `open()` 内 `import pyorbbecsdk`
- 任何 backend、profile、CLI 改动严格保持 mock / isaac 路径零回归

---

## 21. 任务列表（按 planning-and-task-breakdown 模板）

> 每个任务满足：单次 ~3 文件以内、acceptance 可机器验证、verify 命令明确、dependencies 显式。每个阶段完成后给出 checkpoint。

### Phase 0 — 算法骨架（无硬件）

#### Task 0.0：frame_convention utilities

- **Description**：建立 base / gripper_end / cuRobo end_link / Pinocchio fk 之间的转换 utilities 与一致性测试。
- **Acceptance**：
  - [ ] 提供 `wrc_demo_fk(q) -> T_gripper2base`
  - [ ] 提供 `curobo_fk_to_wrc_demo_T(curobo_T)` 常量变换（先为 identity，标定后填）
  - [ ] `test_frame_convention.py` 通过：同一 q 在 wrc_demo 与 cuRobo fk 下输出关系是显式刚性变换
- **Verify**：`pytest tests/test_frame_convention.py -q`
- **Files**：`src/wrc_demo/calibration/frame_convention.py`、`tests/test_frame_convention.py`
- **Dependencies**：无
- **Scope**：M

#### Task 0.1：calib_dataset 数据类 + JSON I/O

- **Description**：定义 `CalibrationSample`、`CalibrationResult`、`CameraBinding` dataclass，含 schema_version 字段，提供 save/load 函数与 .npz K/D 二进制辅助。
- **Acceptance**：
  - [ ] `data/calibration/.gitkeep` 存在
  - [ ] `save(result, path)` 同步写 json + npz
  - [ ] `load(path)` 校验 schema_version
  - [ ] `test_calib_dataset.py` 覆盖保存/加载 round-trip
- **Verify**：`pytest tests/test_calib_dataset.py -q`
- **Files**：`src/wrc_demo/calibration/calib_dataset.py`、`tests/test_calib_dataset.py`
- **Dependencies**：Task 0.0
- **Scope**：M

#### Task 0.2：handeye_solver 联合求解

- **Description**：实现 SE(3) LM + Huber 联合求解 (X, Y)；outlier 拒绝；LOO 验证。
- **Acceptance**：
  - [ ] 完美数据残差 < 1e-9
  - [ ] σ=1mm 噪声 RMSE 增长 ≤ 5 mm
  - [ ] 20% 离群收敛、inlier 残差 ≤ 5 mm
  - [ ] 与 `cv2.calibrateHandEye(eye_in_hand)` 在 marker-as-gripper 子用例上差 < 0.1 mm
- **Verify**：`pytest tests/test_handeye_solver.py -q`
- **Files**：`src/wrc_demo/calibration/handeye_solver.py`、`tests/test_handeye_solver.py`
- **Dependencies**：Task 0.0, 0.1
- **Scope**：L

#### Task 0.3：aruco_session 封装

- **Description**：封装 cv2.aruco 字典、detect、PnP、reprojection 残差；frame_convention 注入。
- **Acceptance**：
  - [ ] `detect_marker_pose(bgr, K, D, marker_size) -> T_marker2cam`
  - [ ] `reprojection_residual(bgr, K, D, T_marker2cam, marker_size) -> float RMSE px`
  - [ ] 单元测试覆盖：合成图像 marker 已知位姿检测误差 ≤ 1 px
- **Verify**：`pytest tests/test_aruco_session.py tests/test_calib_dataset.py -q`
- **Files**：`src/wrc_demo/calibration/aruco_session.py`、`tests/test_aruco_session.py`
- **Dependencies**：Task 0.1
- **Scope**：M

**Phase 0 Checkpoint**
- [ ] `pytest tests/test_frame_convention.py tests/test_calib_dataset.py tests/test_handeye_solver.py tests/test_aruco_session.py -q` 全过
- [ ] 提交 + 人审 + tag `s0-done`

### Phase 1 — Backend

#### Task 1.0：OrbbecCamera 类骨架（fake SDK 可测）

- **Description**：在 `perception/orbbec_camera.py` 实现 `OrbbecCamera(CameraBase)`；fake SDK 注入实现单元测试。
- **Acceptance**：
  - [ ] `from wrc_demo.perception.orbbec_camera import OrbbecCamera` 不导入真实 `pyorbbecsdk`
  - [ ] `OrbbecCamera(cfg).open()` 用 fake SDK 不抛
  - [ ] `_grab()` 用 fake SDK 返回 BGR + 米制 depth + K
  - [ ] `_grab()` 失败 → `CameraError`，30 次后抛给上层
- **Verify**：`pytest tests/test_orbbec_camera.py -q`
- **Files**：`src/wrc_demo/perception/orbbec_camera.py`、`tests/test_orbbec_camera.py`、`tests/conftest.py`（fake_orbbec_sdk fixture）
- **Dependencies**：Phase 0
- **Scope**：M

#### Task 1.1：camera_base.make_camera orbbec 分支

- **Description**：在 `camera_base.py:make_camera()` 添加 `orbbec` 分支，lazy import。
- **Acceptance**：
  - [ ] `make_camera(cfg where type='orbbec')` 不在无 SDK 的环境抛 ImportError
  - [ ] 行为契约与现有 `realsense/uvc/mock/isaac` 一致
- **Verify**：`pytest tests/ -q`
- **Files**：`src/wrc_demo/perception/camera_base.py`
- **Dependencies**：Task 1.0
- **Scope**：XS

#### Task 1.2：orbbec_overhead.yaml + orbbec_wrist.yaml profile

- **Description**：两个 profile，附注释；`requires_calibration: true`；`orbbec_wrist.yaml` 强制 `fuse_beliefs: false`。
- **Acceptance**：
  - [ ] `load_profile('cameras','orbbec_overhead')` 不抛
  - [ ] 配置字段对齐 §12 schema
- **Verify**：`python -c "from wrc_demo.config import load_profile; load_profile('cameras','orbbec_overhead')"`
- **Files**：`configs/cameras/orbbec_overhead.yaml`、`configs/cameras/orbbec_wrist.yaml`
- **Dependencies**：Task 1.0, 1.1
- **Scope**：S

**Phase 1 Checkpoint**
- [ ] `pytest tests/ -q` 全过（含原有 139 个测试）
- [ ] `make_camera()` 在 type='orbbec' 时行为符合契约（fake SDK 路径）

### Phase 2 — CLI 工具

#### Task 2.0：src/wrc_demo/scripts/calib_top_orbbec.py 骨架

- **Description**：`argparse` + subcommand 入口：`--list` / `--bind` / 采集 / `--verify`；不依赖 cuRobo。
- **Acceptance**：
  - [ ] `--help` 列出全部子命令
  - [ ] `--list` 不依赖真机（无 SDK 时 fail-loud 但不抛 stack overflow）
  - [ ] 模块 docstring 含 schema_version + frame_convention
- **Verify**：`python -m wrc_demo.scripts.calib_top_orbbec --help`
- **Files**：`src/wrc_demo/scripts/calib_top_orbbec.py`
- **Dependencies**：Phase 1
- **Scope**：S

#### Task 2.1：自动采集状态机

- **Description**：移植 04b 状态机：用 SafeArm + wrc_demo Kinematics；FK 来源切换；TOP_CALIB_POSES 由 `calib_dataset` 提供。
- **Acceptance**：
  - [ ] 单元测试（dry-run，注入 fake sources）覆盖：settling → searching → capture 状态切换
  - [ ] 离群 / timeout / idle 状态正确迁移
- **Verify**：`pytest tests/test_calib_session.py -q`
- **Files**：`src/wrc_demo/scripts/calib_top_orbbec.py`、`tests/test_calib_session.py`
- **Dependencies**：Task 2.0, Task 0.3
- **Scope**：L

#### Task 2.2：求解 + JSON 落盘

- **Description**：采集完成后自动调 `handeye_solver.joint_solve`；产物写入 `data/calibration/hand_eye_top.json` + `.npz`；写 `runs/calibration_*.jsonl` 审计行。
- **Acceptance**：
  - [ ] 文件包含 §12.3 全部字段（含 metrics、outlier_indices）
  - [ ] 触发 schema_version 校验
- **Verify**：手动跑 dry-run；检查文件存在
- **Files**：`src/wrc_demo/scripts/calib_top_orbbec.py`
- **Dependencies**：Task 2.1
- **Scope**：M

#### Task 2.3：cross-check (--verify)

- **Description**：用已知 base 坐标点验证 T_cam2base；输出 `metrics.known_point_*`。
- **Acceptance**：
  - [ ] 输入 CSV-like 字符串，解析为点列表
  - [ ] 报告每个点误差 + 整体 RMSE
  - [ ] 不合格 exit code 2（CLI 可被 CI 用）
- **Verify**：合成 synthetic 验证
- **Files**：`src/wrc_demo/scripts/calib_top_orbbec.py`
- **Dependencies**：Task 2.2
- **Scope**：M

**Phase 2 Checkpoint**
- [ ] `python -m wrc_demo.scripts.calib_top_orbbec --list --help` 友好
- [ ] 单元 + 集成测试全过

### Phase 3 — Extrinsics loader 多 schema

#### Task 3.0：Extrinsics.from_config 扩展

- **Description**：识别 cuRobo npz / 老 baseline npz / 新 schema json 三类；`frame_convention` 标记；`schema_version` 校验；serial 一致性 warning。
- **Acceptance**：
  - [ ] `test_extrinsics_loader.py` 矩阵：6 个 case（3 schema × 2 mode）全过
  - [ ] `T_result` 旧路径仍可用（向后兼容）
  - [ ] `camera_serial` 不匹配 → stderr 警告但不抛
- **Verify**：`pytest tests/test_extrinsics_loader.py -q`
- **Files**：`src/wrc_demo/perception/grounding.py`、`tests/test_extrinsics_loader.py`
- **Dependencies**：Phase 0
- **Scope**：M

#### Task 3.1：profile→Extrinsics source 调度

- **Description**：`Extrinsics.from_config` 优先看 `extrinsics.source` 字段；无则降级到 `hand_eye_npz` / `T`。
- **Acceptance**：
  - [ ] `source` 字段优先级最高
  - [ ] 不存在的路径报错清晰
- **Verify**：`pytest tests/test_extrinsics_loader.py -q`
- **Files**：`src/wrc_demo/perception/grounding.py`
- **Dependencies**：Task 3.0
- **Scope**：S

#### Task 3.2：fail-closed 的 requires_calibration 策略

- **Description**：`Extrinsics` 增加 `valid: bool` 属性；`build_runtime` 据此强制 `fuse=False` 并 stderr。
- **Acceptance**：
  - [ ] profile `requires_calibration: true` + 无源 → `fuse=False` + 警告
  - [ ] profile `requires_calibration: false` + 无源 → 行为保持原状
- **Verify**：`tests/test_build_runtime_*.py`（新增）
- **Files**：`src/wrc_demo/apps/demo.py`、`src/wrc_demo/perception/grounding.py`
- **Dependencies**：Task 3.1
- **Scope**：S

**Phase 3 Checkpoint**
- [ ] `pytest tests/test_extrinsics_loader.py tests/test_perception.py -q` 全过
- [ ] 139 个原测试零回归

### Phase 4 — 集成

#### Task 4.0：demo.py build_runtime 集成

- **Description**：profile→CameraRig 的 binding 串号校验；WorldWatcher 启动时检查所有需 fuse 相机的 `extrinsics.valid`，有 invalid 写 stderr + 强制降级。
- **Acceptance**：
  - [ ] `--cameras orbbec_overhead,orbbec_wrist` 能完整启动
  - [ ] 任一相机 `extrinsics.valid=False` 时仍能 stream + overlay，仅不参与 3D 融合
  - [ ] 跑默认 mock / isaac / l515 路径零回归
- **Verify**：`pytest tests/ -q`
- **Files**：`src/wrc_demo/apps/demo.py`
- **Dependencies**：Task 3.2, 1.2
- **Scope**：M

#### Task 4.1：WorldWatcher binding 校验

- **Description**：启动时若 profile.serial 存在且 binding 不一致 → stderr warning 但允许（避免换机阻塞）。
- **Acceptance**：
  - [ ] 现存 l515/d435i 等无 binding 概念的 profile 不受影响
- **Verify**：`pytest tests/test_perception.py -q`
- **Files**：`src/wrc_demo/apps/demo.py`
- **Dependencies**：Task 4.0
- **Scope**：S

#### Task 4.2：StreamServer 显示双相机

- **Description**：dashboard 自动列出所有相机；腕部视频可见；顶部视频 + overlay + 在有 belief 时显示。
- **Acceptance**：
  - [ ] `--cameras orbbec_overhead,orbbec_wrist` dashboard 同时显示
  - [ ] 原配置 (l515, uvc) 行为不变
- **Verify**：`pytest tests/test_live_view.py -q`
- **Files**：`src/wrc_demo/apps/stream_server.py`（无改动可能）/ `tests/test_live_view.py`
- **Dependencies**：Task 4.0
- **Scope**：S

**Phase 4 Checkpoint**
- [ ] `pytest tests/ -q` 全过
- [ ] `--cameras orbbec_overhead,orbbec_wrist --arm mock` 可启动 demo.py（offline）
- [ ] `--cameras orbbec_overhead --arm rebot_rs` 出现正确警告（若 extrinsics 缺失）

### Phase 5 — 硬件验证

#### Task 5.0：hardware smoke 测试套件

- **Description**：`tests/test_hardware_smoke.py` 加 orbbec 相关 sub-tests；`-m hardware` 隔离。
- **Acceptance**：
  - [ ] 无 SDK/无相机时 skip；接硬件时跑通
  - [ ] 覆盖 §15.2 全部 case
- **Verify**：`pytest tests/ -m hardware -q`（接硬件时）
- **Files**：`tests/test_hardware_smoke.py`
- **Dependencies**：Phase 4
- **Scope**：M

#### Task 5.1：Rig verify 程序（cross-check on real points）

- **Description**：按 §16.4 流程跑真实标定 + cross-check；记录所有 metrics。
- **Acceptance**：
  - [ ] 真实采集 ≥30 个 inlier
  - [ ] known_point RMSE ≤ 10 mm
  - [ ] 失败回滚到旧 npz，git tag 旧版
- **Verify**：人工 + 自动报表
- **Files**：无新增
- **Dependencies**：Task 5.0
- **Scope**：人工 ~1 人天

#### Task 5.2：表平面 / depth 链验证

- **Description**：`DepthProvider.fit_table_plane` 一次；在 desktop 看是否得到合理 `table_z`。
- **Acceptance**：
  - [ ] table_z 误差 ≤5 mm
  - [ ] 空桌面帧在 lifting 到 base 后落在正确 z
- **Verify**：人工测；CLAUDE.md `fit_table_plane`
- **Files**：无新增
- **Dependencies**：Task 5.1
- **Scope**：S

#### Task 5.3：文档 + 交接

- **Description**：ARCHITECTURE.md / README.md 同步；CLAUDE.md 状态保留。
- **Acceptance**：
  - [ ] ARCHITECTURE.md 增 orbbec + calibration 描述
  - [ ] README.md "Onsite bring-up"段指本计划工具
  - [ ] 新 review 发现补 `test_review_regressions_v3.py`
- **Verify**：人审
- **Files**：`docs/ARCHITECTURE.md`、`README.md`
- **Dependencies**：Phase 4
- **Scope**：S

**Phase 5 Checkpoint**
- [ ] 实机 cross-check 通过；产物归档
- [ ] 真机 demo 可启动 dashboard、WorldWatcher 写入 belief（人工确认）
- [ ] docs 更新到位

---

## 22. Questions Resolution Log（2026-07-20）

| ID | 用户决议 | 落地位置 |
|---|---|---|
| **Q1** | 与 04b 一致（marker 平贴于 gripper 上方白板，正面朝上；100 mm ArUco 4x4_50 ID 0） | `src/wrc_demo/scripts/calib_top_orbbec.py` 的 `--marker-size` 默认 0.10 |
| **Q2** | 只做参考、重新撰写：rebot_grasp 与 cuRobo 的 Orbbec 驱动**只读不引用**；`OrbbecCamera` 重写；`src/wrc_demo/scripts/calib_top_orbbec.py` 零 vendor import | CLAUDE.md 严禁 vendor import；本模块唯一依赖为 `pyorbbecsdk` 自身 |
| **Q3** | 本阶段不做；列入 ROADMAP Mid-term | `docs/ROADMAP.md` 不动直到本期结束；`PLAN.md` §17 风险表中将"wrist 同步握抓"标为下期 |
| **Q4** | 顺手统一：在本次编辑涉及的文件里，把硬写的 `/home/johnny/...` 一并改成本机 `/home/seeed/...`；CLAUDE.md 中已知陈旧条目（如 `.mcp.json`）按现有"regenerate with setup_agents.py"约定保留 | README.md、ARCHITECTURE.md、scripts/setup_agents.py 中非 vendor 路径同步更新 |
| **Q5** | α 维持现状：CI 不变；`addopts=-m 'not hardware'` 继续；offline 路径自动跑；hardware 仅人工/实机 | §22.1 标为已选定 α；CLAUDE.md 同步承诺保持；不新增 CI workflow |
| **Q6** | 选 a：`*.json` 跟踪（diff 可见）、`*.npz` 不跟踪（K/D 二进制 + 设备指纹） | 在根目录 `.gitignore` 加 `data/calibration/*.npz`；`hand_eye_top.json` 与 `cameras.json` track |
| **Q7** | 选现有（`scipy.optimize.least_squares`） | `src/wrc_demo/calibration/handeye_solver.py` 内 `scipy.optimize.least_squares(method='lm')`；`scipy` 已在 `.demo` 基础 env |

### 22.1 Q5 — CI 策略（已选定 α，2026-07-20）

**当前 CI 契约（CLAUDE.md）**：`uv sync --extra dev --extra kinematics && uv run pytest tests/ -q` (Python 3.12)，无 lint；`pyproject.toml` `addopts="-m 'not hardware'"`，hardware 测试天然被隔离。

**选定方案 α — 维持现状**：

- 主 CI 不变；新增的 offline 测试（fake SDK、合成求解、loader schema、frame convention）自动纳入 `uv run pytest tests/ -q`
- hardware 测试仅在实机本地跑 `pytest tests/ -m hardware -q`；§16 手工 checklist + §21 Phase 5 验收即可
- 实机 metrics 走 `runs/calibration_*.jsonl` + `data/calibration/hand_eye_top.json` 审计，不进 CI
- 0 个新 workflow 文件；CLAUDE.md 同步承诺保持

**未选方案备查**：

- β nightly job：需要 rig runner、CAN/USB secret 管理，工程量与本阶段 ROI 不匹配
- γ 仿真 smoke：CI 增加仿真帧 dry-run，但与真机偏差无法覆盖
- δ β+γ：组合方案，本阶段不评估

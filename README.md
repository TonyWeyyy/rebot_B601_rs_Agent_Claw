# wrc_demo — agentic grasping for the reBot DevArm

[![CI](https://github.com/johnnynunez/wrc_demo/actions/workflows/ci.yml/badge.svg)](https://github.com/johnnynunez/wrc_demo/actions/workflows/ci.yml)

Camera-agnostic, LLM-orchestrated tabletop manipulation on the Seeed reBot
DevArm B601 (RobStride build), driven from an NVIDIA DGX Spark. This is the
agentic evolution of the
[reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp)
baseline, designed after NVIDIA GEAR's
[ASPIRE](https://research.nvidia.com/labs/gear/aspire/) (curated skill API +
multimodal traces + skill library) and
[Agentic-VLA](https://arxiv.org/abs/2605.22896) (task decomposition + VLM
advisor + experience memory).

```
              ┌───────────────────────────── agent ─────────────────────────────┐
task ────────▶│ reflex ▸ habit ▸ LLM — decompose → tool call → verify → recover │
              │ routine commands never wait on the model; VLM advisor on failure│
              └──────────────────────────┬──────────────────────────────────────┘
                 curated skill API — 20 traced skills (ASPIRE-style)
   ┌────────────┬────────────┬───────────┴────────┬──────────────┬─────────────┐
   ▼            ▼            ▼                    ▼              ▼             ▼
get_observation localize   grasp_object        place_at/on    push_object  recall_memory
   │            │            │                    │              │             │
┌──┴────────────┴──┐   ┌─────┴──────┐      ┌──────┴───────┐ ┌────┴─────┐ ┌─────┴────────┐
│ perception       │   │ grasping   │      │ control      │ │ safety   │ │ memory       │
│ camera-agnostic  │   │ GraspGen-X │      │ FK/IK (pin)  │ │ harness  │ │ 15 s episodic│
│ Frame(+depth_m)  │   │ ▸OBB fallbk│      │ min-jerk     │ │ gates    │ │ + beliefs +  │
│ depth chain      │   │ material → │      │ streaming    │ │ every    │ │ habit/grasp  │
│ open-vocab YOLO  │   │ force prof │      │ RobStride CAN│ │ waypoint │ │ memory       │
└──────────────────┘   └────────────┘      └──────────────┘ └──────────┘ └──────────────┘
```

## Install

wrc_demo is a pip-installable Python package distributed with a single
conda environment file that bundles everything (core stack + LLM clients +
every hardware SDK). The standard env name is **`wrc-demo`**.

### 0. One-time system deps (Ubuntu/Debian/WSL2)

```bash
sudo apt update
sudo apt install -y libusb-1.0-0-dev ffmpeg git can-utils
```

* `libusb-1.0-0-dev` — Orbbec Gemini 2 / Femto Bolt depth cameras
* `ffmpeg` — UVC4K / RealSense2 RGB streams + `video_stream` tool
* `can-utils` — SocketCAN bring-up for the reBot Arm (`sudo ip link set can0 up type can bitrate 1000000`)
* `git-lfs` is only required if you clone a sibling project that LFS-tracks
  its own weights; wrc-demo's own weights are downloaded by
  `scripts/setup_models.py` (see step 4 below), not via LFS.

### 1. Clone the repo

```bash
git clone https://github.com/TheMoonAstronaut/wrc.git
cd wrc
```

### 2. Create and activate the conda env (one shot)

```bash
conda env create -f environment.yml
conda activate wrc-demo
```

`environment.yml` gives you:

- **Python 3.10 + numpy + pyyaml + pytest** (conda-forge)
- **`wrc-demo` package editable with every extras**:
  - `[perception]` → `ultralytics` (YOLOE / YOLO-World detector)
  - `[llm]` → `openai` + `anthropic` (covers llama.cpp / vLLM local servers too)
  - `[kinematics]` → `pin` (Pinocchio FK/IK)
  - `[arm]` → `motorbridge` (RobStride over SocketCAN)
  - `[dev]` → `pytest` + `pytest-cov`
- **`pyorbbecsdk2`** (Orbbec Gemini 2 SDK; pip-only)
- **`fashionstar-uart-sdk`** (PiPER Mate master arm; optional)
- **`huggingface-hub`** (for the manual-download path in `setup_models.py`)

That's a single conda env that supports **all four run modes** (mock,
real cameras, real arm, real everything) — no tier switching needed.

> **Why `opencv` is not in the conda deps**: `pyproject.toml` pins
> `opencv-python-headless` (so `cv2.imshow` can never silently crash on a
> headless server). Letting the pip-installable `-e .[...]` provide it
> avoids the conda-forge opencv ↔ opencv-python-headless conflict.

### 3. (Real hardware only) set up USB + CAN permissions

```bash
# Orbbec udev rules (one-time after install)
bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger

# CAN bus for the reBot Arm (each session, or persist via systemd)
sudo ip link set can0 up type can bitrate 1000000
```

### 4. Model weights

The `models/` directory is gitignored. After install, fetch the weights
your config needs:

```bash
python scripts/setup_models.py                # list what's there / missing
python scripts/setup_models.py --fetch        # auto-download what we can
python scripts/setup_models.py --fetch --profile all   # everything
```

The script prints manual-instruction hints for weights that can't be
auto-downloaded (custom-trained YOLOE from the sibling `rebot_grasp`
project, MobileCLIP torchscript exports, etc.). See
[`models/README.md`](models/README.md) for the full manifest.

### 5. Verify the install

```bash
# ~30s mock test suite (17+ tests, all pass on a clean checkout)
python -m pytest tests/test_extrinsics_loader.py \
                 tests/test_hand_eye_compensation_per_camera.py \
                 tests/test_demo_yaml_llm_profile.py -v

# Mock demo (no hardware, no LLM — exercises wiring + reflex path)
python -m wrc_demo.apps.demo --task "wave"
```

If both run, your install is healthy. Pick a tier in the next section and
go.

### Fallback: a uv venv for mock-only work

If you can't or won't install conda (e.g. on a strict CI runner), you can
use a uv-managed venv at `./.venv` for mock testing:

```bash
bash scripts/setup_env.sh           # mock-only (pytest + wrc-demo[dev])
bash scripts/setup_env.sh --all     # full install (perception + llm + arm + dev)
```

This is a mock-friendly environment only — for real hardware always use
the conda `wrc-demo` env above.

## The four challenges, addressed

1. **Depth** — every camera backend yields `Frame` objects with float32
   *metric* depth aligned to color (kills the L515 0.25 mm vs D4xx 1 mm scale
   bug class). Cameras without depth fall through a strategy chain:
   sensor → optional mono-depth plugin → table-plane ray-casting (known
   plane ⇒ whole-image depth; the YOLO detection mask is applied later at
   grounding to lift object pixels to 3D), so an RGB-only webcam still
   grasps tabletop objects. `src/wrc_demo/perception/depth_provider.py`.
2. **Harness** — a fail-closed `SafetyHarness` vets *every streamed waypoint*
   (joint limits, velocity caps, workspace AABB, table-plane clearance with a
   grasp-exemption cylinder, keep-out zones, perception watchdog, e-stop
   latch), and every skill call writes an ASPIRE-style multimodal trace
   (`trace.jsonl` + before/after keyframes). `src/wrc_demo/safety/harness.py`.
3. **Grasping / force** — learned 6-DoF grasps from a GraspGen-X ZMQ server
   (`grasp.backend: graspgenx`, the default — `scripts/serve_graspgenx.sh`)
   with automatic fallback to analytic 3D OBB grasps planned in the base
   frame (short-axis jaw opening, height-fraction grasp depth) whenever the
   server is down. Candidates are re-ranked by a persisted per-object
   grasp-outcome memory, then vetted against IK *and* the safety-harness
   geometry before execution. Material-aware grip profiles
   (rigid/fragile/soft/deformable/slippery/heavy) map to two-stage
   stall-aware closes; the LLM passes the material hint it infers.
   `src/wrc_demo/grasping/`.
4. **Kinematics** — self-contained Pinocchio FK/IK on the RS URDF (ships in
   `assets/`), damped-least-squares with random restarts, min-jerk joint
   streaming with feedback-based settling. Live joint positions on the RS
   arm come from RobStride `mechPos` (0x7019) param reads — the only reliable
   feedback path on this hardware. `src/wrc_demo/control/`.

Plus **memory**: a 10–15 s episodic window (events + JPEG thumbnails +
optional TurboQuant-compressed embeddings, arXiv:2504.19874) and an
object-permanence belief store, so "the mug you saw 10 seconds ago" is still
actionable after occlusion. `src/wrc_demo/memory/`.

## Quick start — three tiers

Pick the tier that matches your hardware. Every tier expects the install
section above to be done first.

### Tier 1 — Mock (laptop, no USB, no GPU)

```bash
# offline wiring check: mock camera + mock arm + scripted LLM, no hardware.
# Routine commands run on the REFLEX fast path (no LLM); a livestream
# dashboard with N camera streams + robot narration prints its URL.
PYTHONPATH=src python -m wrc_demo.apps.demo --task "pick and place pink object"

# tests (unit/integration; live-hardware tests deselected by default)
python -m pytest tests/ -q
python -m pytest tests/ -m hardware -q     # needs L515 + can0 up (read-only)

# learned grasps (the default backend) need the GraspGen-X server running;
# without it grasping silently falls back to the analytic OBB planner
scripts/serve_graspgenx.sh

```

### Tier 2 — Real rig (cameras + arm + cloud LLM)

```bash
# real rig, N cameras (first = manipulation camera), cloud LLM fallback
sudo ip link set can0 up type can bitrate 1000000
python -m wrc_demo.apps.demo --interactive \
    --cameras l515,uvc4k --arm rebot_rs --llm anthropic

```

### Tier 3 — Real rig + local LLM

```bash
# real rig, local Qwen3-VL on the Spark (start the server first)
scripts/start_cosmos3_server.sh --bg                # vLLM serves Qwen3-VL-2B-AWQ-4bit
python -m wrc_demo.apps.demo --task "..." \
    --cameras l515 --arm rebot_rs --llm local_qwen3_vl
```

> **Switching the LLM profile** (no fallback cascade — pick one):
>
> ```bash
> # edit configs/demo.yaml: llm.profile: minimax | anthropic | openai | local_qwen3_vl | mock
> # or override ad-hoc:
> python -m wrc_demo.apps.demo --llm minimax --task wave
> ```
>
> See [`configs/demo.yaml`](configs/demo.yaml) for the full profile list
> and the LLM-resolution priority (CLI > demo.yaml > mock). ADR-0008 in
> the docs covers the optional `llm.fallback` chain.

### Handy entry points (after `pip install -e .`)

```bash
wrc-demo                # main demo CLI (= python -m wrc_demo.apps.demo)
wrc-record              # episode recorder
wrc-mcp                 # MCP server (for Codex / Hermes / Claude Code / etc.)
wrc-view                # camera dashboard
wrc-calib-top           # top-camera hand-eye calibration (= sudo wrc-calib-top)
wrc-calib-wrist         # wrist-camera hand-eye calibration
```

On this contributor rig, `scripts/setup_env.sh` (legacy uv venv installer) is
kept for history; new users should use `environment.yml` as shown above. pyrealsense2 comes from the local
[librealsense L515 fork](https://github.com/johnnynunez/librealsense) build.
Open-vocabulary text prompts (YOLOE/YOLO-World) additionally need
`uv pip install git+https://github.com/ultralytics/CLIP.git`; the closed-set
`yolo11n.pt` works without it.

YOLOE also needs a MobileCLIP text encoder — for the shipped
`yoloe-11s-seg.pt` that is `mobileclip_blt.ts` (~572 MB, too big for
GitHub, so it is gitignored; this checkout carries it at
`models/mobileclip_blt.ts`). Ultralytics resolves it **relative to the
working directory** and auto-downloads it there on first use — launch from
`models/` (or copy the `.ts` next to your CWD), otherwise every detection
silently vanishes.

## LLM backends

| profile | backend | notes |
|---|---|---|
| `mock` | scripted (no network) | tests / wiring checks; default in `tests/conftest.py` |
| `anthropic` | Claude (cloud) | `ANTHROPIC_API_KEY` env; vision + tools |
| `openai` | any OpenAI-compatible cloud endpoint | `OPENAI_API_KEY` env |
| `minimax` | MiniMax cloud (`api.minimax.chat/v1`, model `MiniMax-M3`) | `MINIMAX_API_KEY` env; text-only (no vision); see `scripts/test_minimax_connectivity.py` |
| `local_qwen` | local Qwen via llama.cpp / vLLM on the Spark | OpenAI-compatible; MTP speculative decoding (~1.4–2.2× decode). Needs ~24 GB. |
| `local_cosmos3` | local `nvidia/Cosmos3-Edge` 4B (AWQ W4A16) via vLLM | 8 GB VRAM; Qwen3-Coder XML tool protocol; see `docs/SPEC-local-llm.md` and `scripts/serve_cosmos3_vllm.sh` |

> **Secrets are env vars, never in YAML.** Cloud profiles
> (`anthropic`/`openai`/`minimax`) read their key from the env var named
> in `env_key:` (see `.env.example`). The old practice of writing
> `api_key: sk-…` into the YAML is gone — see `docs/adr/0006-move-secrets-to-env-vars.md`.

> **Cold start** selects via `--llm <profile>`. **Runtime switch** uses
> the MCP `llm_switch` tool (see `docs/adr/0008-llm-profile-switching.md`).
> A `fallback_chain: [...]` in the orchestrator walks through other
> profiles on transient LLM failures; a `LLMConfigError` (missing env
> var) is **not** masked and propagates immediately.

> **Keep the profile in sync with the server:** `configs/llm/local_qwen.yaml`
> currently pins `model: Qwen3VL-30B-A3B-Instruct-Q4_K_M` while both serve
> scripts fetch **Qwen3.6-27B**. llama.cpp ignores the request's model name
> (and its script wires vision via the mmproj projector, so
> `supports_vision: true` still holds there); vLLM rejects a mismatched
> model name and has no vision wiring. Set the profile's `model:` (and
> `supports_vision:` if your server lacks a vision projector) to whatever
> the server actually loads. For a fresh local-only setup on 8 GB
> VRAM, prefer `local_cosmos3` (ADR-0007) — it's a 4 B model that
> actually fits and ships with a working tool-calling path.

## Run it under any MCP agent platform

The whole skill runtime is also exposed as an **MCP stdio server**
(`wrc_demo/apps/mcp_server.py`) — so instead of the built-in loop, any
MCP-capable agent platform can drive the arm. The agent gets the same 20
safety-gated skills (only the loop-internal `task_done` is excluded) plus
five gateway extras — `camera_snapshot` (returns a live JPEG the agent can
*see*), `world_state`, `live_view_url`, and `emergency_stop`/`reset_stop` —
25 tools total. Safety harness, tracing,
memory and the always-on camera window are identical — only the brain swaps.

One registrar for every host — prints what each platform needs, `--write`
applies the file edits (preserving unrelated entries):

```bash
python scripts/setup_agents.py                     # show all hosts
python scripts/setup_agents.py --host codex --write
python scripts/setup_agents.py --camera l515 --arm rebot_rs --write
```

| platform | mechanism | setup |
|---|---|---|
| **Hermes** | `~/.hermes/config.yaml` `mcp_servers` | `./scripts/hermes_demo.sh` (interactive: register + test + chat) |
| **Claude Code** | project `.mcp.json` (ships in this repo, pinned to the demo rig's venv paths) | regenerate for your machine first: `setup_agents.py --host claude --python <venv python> --write`; user-scope: `--host claude` prints the `claude mcp add` one-liner |
| **Claude Desktop** | `claude_desktop_config.json` | paste the JSON block from `setup_agents.py --host claude` |
| **Codex CLI** | `~/.codex/config.toml` `[mcp_servers.wrc-demo]` | `setup_agents.py --host codex --write`, verify with `codex mcp list` |
| **OpenClaw** | native `mcp.servers` (2026+) or [mcporter](https://docs.openclaw.ai/cli/mcp) | `setup_agents.py --host openclaw` prints the `openclaw mcp set` one-liner + JSON block |

The server pre-warms perception at startup (cameras + detector + world
model) while the ARM stays unpowered until the first motion command
(LazyArm) -- so "pick and place pink object" from the chat is a single
`pick_and_place` tool call that starts moving immediately. Env knobs:
`WRC_CAMERAS` (comma list, first = manipulation camera), `WRC_CAMERA`
(single-camera fallback), `WRC_ARM`, `WRC_DETECTOR_MODEL`,
`WRC_DETECT_CLASSES`, `WRC_VIEW`, `WRC_PREWARM`, `WRC_STREAM`,
`WRC_STREAM_PORT`, `WRC_RUN_DIR` (trace dir), `DISPLAY`.

## Safety notes for the live rig

- The safety harness fails closed; motions abort mid-stream on violation.
- Gripper open/close angles in `configs/arms/rebot_rs.yaml` were
  characterized on the DM build — **re-verify travel and stall torque on the
  RS gripper before the first grasp**.
- Do not run `motorbridge-gateway` / MotorBridge Studio while the demo runs
  (host-id 0xFD conflict on the CAN bus).
- Hand-eye extrinsics in the camera profiles are placeholders — calibrate
  on-site with `python -m wrc_demo.scripts.calib_top_orbbec` (the legacy
  baseline repo's `hand_eye.npz` still loads via `hand_eye_npz:` in the
  extrinsics block; the new JSON schema from the calibration tool loads
  via `source:` and is the preferred path on the reBot + Orbbec rig).
- Strict top-down tool poses are only IK-reachable below z ≈ 0.15 m on the
  B601-RS (wrist limits); grasp heights + hover offsets are configured
  accordingly.

### Hand-eye calibration on-site (Orbbec rig)

```bash
# 1. enumerate devices, pick the top camera and the wrist camera
python -m wrc_demo.scripts.calib_top_orbbec --list
python -m wrc_demo.scripts.calib_top_orbbec --bind top   --index 0
python -m wrc_demo.scripts.calib_top_orbbec --bind wrist --index 1

# 2. collect the calibration dataset (>= 30 samples, two-stage press-enter)
sudo ip link set can0 up type can bitrate 1000000
python -m wrc_demo.scripts.calib_top_orbbec \
    --marker-size 0.10 --n 30 --out data/calibration/hand_eye_top.json

# 3. cross-check against known base-frame points (rig-measured, <10 mm RMSE)
python -m wrc_demo.scripts.calib_top_orbbec --verify \
    --npz data/calibration/hand_eye_top.json \
    --known-points "0.30,0.00,0.02;0.25,-0.05,0.02;0.35,0.05,0.02"

# 4. run the demo with the new calibration
python -m wrc_demo.apps.demo --interactive \
    --cameras orbbec_overhead --arm rebot_rs --llm anthropic
```

The tool runs `joint_solve` (SE(3) LM + Huber + 3*MAD outlier rejection)
on FK from `wrc_demo.control.kinematics.Kinematics` (Pinocchio on the RS
URDF with `joint_signs=[-1]*6` and `ee_frame='gripper_end'`) and ArUco
PnP against the SDK intrinsics. Result: `data/calibration/hand_eye_top.json`
(+ `.npz` with K, D) and `data/calibration/cameras.json` (serial binding).

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module-by-module design and
  the decisions behind it
- [docs/ROADMAP.md](docs/ROADMAP.md) — VLA policy backend, GraspGen-X
  follow-ups, NuRec sim2real, skill-library growth
- [CLAUDE.md](CLAUDE.md) — working guide for AI coding agents (commands,
  invariants, gotchas)

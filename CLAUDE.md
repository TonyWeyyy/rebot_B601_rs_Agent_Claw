# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

LLM-orchestrated tabletop manipulation on the Seeed reBot DevArm B601 (RobStride), driven from an NVIDIA DGX Spark. Read `README.md` (user-facing overview, quick start, safety notes for the live rig) and `docs/ARCHITECTURE.md` (module map, design decisions) first — this file covers what they don't, and flags where they are stale (see "Doc drift" at the bottom).

## Environment & commands

The project's standard environment is the conda env **`wrc-demo`**, created
from `environment.yml`. Everything (mock tests + real hardware + LLM
clients) is installed there; you should never need another Python.

```bash
conda activate wrc-demo                              # the only env you need

# Mock tests (~30s, all pass on a clean checkout)
pytest tests/test_extrinsics_loader.py \
        tests/test_hand_eye_compensation_per_camera.py \
        tests/test_demo_yaml_llm_profile.py -v

# Full mock suite
pytest tests/ -q                                      # ~290 tests; 2 hardware tests auto-deselected

# Single test
pytest tests/test_safety.py::test_velocity_cap -v

# Live-rig tests (L515 + can0, read-only)
pytest tests/ -m hardware -q

# Offline wiring check: mock camera + mock arm + scripted LLM, no hardware
python -m wrc_demo.apps.demo --task "pick and place pink object"
python -m wrc_demo.apps.demo --interactive --no-view --no-serve   # REPL, no GUI/dashboard
```

### Legacy: conda `DM` + uv `.venv`

This repo was originally developed against a hand-built conda env named
`DM` (`/home/seeed/miniforge3/envs/DM/`) plus a sibling uv venv at
`./.venv`. As of 2026-08-28 that split is **deprecated** — the conda
`wrc-demo` env from `environment.yml` carries every package both of those
environments had (motorbridge, ultralytics, pyorbbecsdk, pin, anthropic,
openai, wrc-demo editable, etc.). New checkouts and CI use `wrc-demo`
only. The `DM` env still works for existing rigs but will not be
updated; the `.venv` is kept as a mock-only fallback (see
`scripts/setup_env.sh`).

### Things to know about the conda env

- Tests need no `PYTHONPATH` (`tests/conftest.py` inserts `src/`). The
  package is installed editable in `wrc-demo`, so apps run without it
  too; `PYTHONPATH=src` (as in the README) is only needed with an
  interpreter that lacks the editable install.
- `addopts = "-m 'not hardware'"` in pyproject applies even to
  explicitly named files: `pytest tests/test_hardware_smoke.py` runs
  **zero tests silently** — you must add `-m hardware`.
- Pinocchio-dependent tests skip via `needs_pin` from conftest when
  `pin` or the RS URDF under `assets/` is missing.
- CI is exactly `conda env create -f environment.yml && conda run -n
  wrc-demo pytest tests/ -q` (Python 3.10). There is no lint step.

## Architecture

`docs/ARCHITECTURE.md` has the module map and data-flow diagram. The load-bearing facts beyond it:

**Composition root.** `apps/demo.py:build_runtime()` is the single place everything is wired: kinematics → LazyArm → SafetyHarness/SafeArm → CameraRig of CameraStreams → LockedDetector → WorldWatcher → BeliefStore/EpisodicMemory → TraceLogger → SkillRuntime → optional StreamServer/RigViewer. The demo CLI, `apps/mcp_server.py`, and `scripts/dashboard_runner.py` all call it; tear down with `shutdown_runtime()`. Detector selection: only the FIRST/manipulation camera's profile can override the global `detector:` in `configs/demo.yaml` — one shared `LockedDetector` serves the whole rig, and `detector:` blocks on non-primary camera profiles are silently ignored (the mock camera pins `type: mock` so offline runs never import ultralytics). The first camera in any `--cameras`/`WRC_CAMERAS` list is the manipulation camera; the rest only stream/fuse. A camera fuses 3D beliefs only if its profile has an `extrinsics:` block (or `fuse_beliefs: true`).

**Skill system (the main extension point).** No registry: `SkillRuntime.execute()` in `src/wrc_demo/skills/runtime.py` dispatches to `skill_<name>` methods by name, and `TOOL_SPECS` at the bottom of the same file is a hand-maintained list of JSON schemas. To add a skill: (1) write `skill_<name>()` returning a dict — raise `SkillError`/`SafetyViolation` for failures, never let other exceptions escape by design (`execute()` converts everything to `{"ok": false, ...}`); (2) add a `TOOL_SPECS` entry — `tests/test_llm_and_library.py` asserts every spec has a matching `skill_` method, but the reverse is unchecked: forgetting the spec entry fails no test, the skill just never becomes visible to the LLM/MCP; (3) **if it moves the arm, add it to `_MOTION_SKILLS` (runtime.py:31)** — this is the only thing that pauses WorldWatcher belief fusion during motion; forgetting it re-registers the held object at bogus mid-air positions; (4) tracing (trace.jsonl + keyframes + narration) and MCP exposure are automatic — `execute()` is the single choke point, and `mcp_server.py` serves `TOOL_SPECS` minus `_EXCLUDED_TOOLS` plus `_EXTRA_TOOLS`; (5) optionally add a regex to `_RULES` in `agent/reflex.py` so the command runs LLM-free (rules are order-sensitive, first match wins, and they drive the real arm with zero LLM involvement — test against `tests/test_streaming_fastpath.py`). One schema source feeds all three consumers (OpenAI-compat, Anthropic, MCP) — a new skill needs zero per-backend work.

**Three-tier dispatch.** `AgentOrchestrator.run_task()` tries tier-1 reflex (regex grammar) then tier-2 experience (hashed-BoW cosine ≥ 0.9 AND wins > losses, persisted in `runs/experience.json`) before ever calling the LLM. The LLM loop ends only via a `task_done` tool call or `max_steps` (default 30); only the FIRST tool call per turn executes (extras get error results back — the OpenAI protocol requires a result per call id). Import the orchestrator as `from wrc_demo.agent.orchestrator import AgentOrchestrator` — `agent/__init__.py` deliberately doesn't re-export it (circular import with `skills.runtime`).

**Motion safety path.** Skills only ever hold a `SafeArm`; raw arm objects never leak upward. `SafeArm.move_joints()` auto-stretches duration to keep min-jerk peak velocity under the cap, checks perception freshness once at `begin_motion()` (deliberately not per-waypoint), then `ArmBase.stream_to()` calls `harness.approve()` on every 50 Hz waypoint — `SafetyViolation` aborts mid-stream. `approve()` has "escape rules" (an arm already outside a limit may move strictly back toward validity); `vet_pose()` is the static pre-motion twin without them. Gripper commands bypass geometric gating (e-stop check only); `SafeArm.raw` is the sanctioned escape hatch for backend-specific ops.

**Grasp pipeline.** Three layers, evaluated in order by `SkillRuntime._plan_grasps`:

1. **Camera-frame planner** (`src/wrc_demo/grasping/camera_grasp.py::plan_grasp_from_mask`, primary when the detection has a real segmentation mask and the last frame has depth — i.e. always on the real rig and the mock stack with seg weights). Mirrors `rebot_grasp/utils/ordinary_grasp.py::estimate_grasp`: `cv2.minAreaRect` on the mask → depth-quantile back-project → `approach = -normalize(position_cam)` (the gripper dives along the camera's line of sight, so a tilted camera no longer biases the grasp geometry) → `_grasp_axes_to_rebot_tcp_rotation` flips the up-pointing plane normal to a down-pointing tool-forward, then `T_cam2base @ T_grasp_cam` lifts to base frame and `insertion_depth_m` (default 0.015 m, matches the rebot_rs gripper's tip offset below TCP) pushes the TCP INTO the object so the fingers wrap the mass instead of pinching the surface. The output `Grasp` has `approach = R[:,0]` (tool-forward in base) and the existing `Grasp.pregrasp_position(offset) = position − approach * offset` and `select_grasp` IK seed-from-home flow are unchanged.

2. **GraspGen-X server** (`grasp.backend: graspgenx`) — learned 6-DoF grasps from the ZMQ server (`scripts/serve_graspgenx.sh`, :5556) when configured and reachable.

3. **Analytic OBB planner** (`src/wrc_demo/grasping/obb_grasp.py::plan_grasps_from_fix`) — the silent fallback for the no-mask / no-depth case (mono cameras, hardware bring-up before seg weights land) and as the IK fallback when layer 1 or 2 produces nothing.

All layers produce the same `Grasp` dataclass; `GraspOutcomeMemory` re-ranks/z-nudges the concatenated list, then `select_grasp()` filters by jaw width, solves IK (pregrasp then grasp), and pre-vets harness geometry including 7 samples along the descent segment. Grasp IK is seeded from `home_q`, not the live pose, and re-homes before the pregrasp on purpose (elbow-down IK branches dip links below the table) — don't "optimize" either away.

**Config system.** `load_demo_config()` loads `configs/demo.yaml` and inserts whole profile dicts under `camera`/`cameras`/`arm`/`llm` — no deep merge, no env interpolation; the only substitution is `${repo}`/`${assets}` in strings. Adding a profile = dropping a YAML file in `configs/{cameras,arms,llm}/`; the `type:` key selects the backend class in the `make_camera`/`make_arm`/LLM factories. `Cfg` is an attr/dict hybrid: attribute access for required keys (raises), `.get()` for optional; the sanctioned runtime-override idiom is writing into `cfg._data` (see `mcp_server.py`). Paths derive from `config.py`'s location, so the package only works from a source checkout.

**Adding a camera/arm backend.** Subclass `CameraBase` (implement `open/close/has_depth/_grab`; `_grab` raises `CameraError` for retryable faults and must NOT set `frame_id` — the watcher dedupes on it) or `ArmBase` (six abstract methods), add a lazy-import branch in `make_camera()`/`make_arm()`, add a profile YAML. Arm profiles pointing at the shipped assets MUST carry `joint_signs: [-1]*6` (see contracts below). Heavy imports (pyrealsense2, ultralytics, pinocchio, zmq, the arm SDK) always stay lazy inside the branch/method that needs them so the mock stack imports nothing heavy.

**Optional runtime hand-eye compensation.** Add `hand_eye_compensation_m: {x, y, z}` (meters, default `[0, 0, 0]` = identity) to a camera profile's `extrinsics:` block for a manual offset on top of the calibrated matrix. Per-camera only -- no demo.yaml-level global; every camera must declare its own value. Mirrors rebot_grasp `cfg.calibration.hand_eye_compensation_m`; see ADR-0009.

**MCP server.** `apps/mcp_server.py` is hand-rolled newline-JSON-RPC over stdio, and `stream_server.py` serves HTTP with stdlib `http.server` — both deliberate; don't add MCP-SDK or web-framework deps here. It hardcodes `llm='mock'`: the MCP host is the brain and the built-in tiers are bypassed. Stdout discipline: protocol frames only — every runtime call is wrapped in `redirect_stdout(sys.stderr)`; new code on this path must never print to stdout.

## Contracts & invariants

- **Units/frames:** positions in meters in the ROBOT BASE frame; joints in radians (degrees only in `get_observation` output). Base frame: +x away from robot, +y left; "left" = larger y. OBB convention everywhere: extents sorted descending, axes as matrix columns — never use `extent[2]` as height, use `top_z`.
- **`Frame.rgb` is BGR** (OpenCV) despite the name; `depth_m` is float32 metric meters aligned to color, 0 = invalid. `Frame.size` is (w, h) — opposite of numpy shape.
- **In-process perception/state timestamps (`Frame.t`, `RobotState.t`, belief ages, `MemoryEvent.t`) are `time.monotonic()`**, never wall clock — belief freshness math depends on it. Persisted/reporting timestamps (trace.jsonl, grasp-memory `saved_at`, dashboard status) are wall-clock `time.time()`; never mix the two.
- **Joint mirror:** both shipped RS assets (URDF and USD) are authored mirrored (`q_asset = -q_local`); `joint_signs` bakes the flip in at model load, so every constant in the repo (home_q, limits, tuned poses) is LOCAL. `tests/test_usd_model.py` pins URDF↔USD kinematic identity — it fires if you touch `assets/`.
- **Error taxonomy** (`types.py`): `SkillError` = failure the agent should reason about; `SafetyViolation` = harness rejection; plain `RuntimeError` = crash/hardware fault. The orchestrator's recovery logic depends on this split.
- **Invariant:** `Kinematics.ik` `limit_margin` (0.025 rad) must stay strictly greater than the harness `joint_margin` (0.02) so IK never returns a pose the harness rejects — a past review defect; never tune one side alone.
- **Booleans from LLMs arrive as strings** ("false" is truthy) — coerce with the `_as_bool` pattern in any new skill taking a bool.
- Gripper `open_pos`/`closed_pos` polarity differs per arm profile (RS: open=-6.8 rad; mock: 0→1; Isaac: fraction, open=1.0) — compute travel as `closed_pos - open_pos`, sign-aware.
- Comments in control/safety/grasping encode rig-verified failure observations that tests rely on — preserve and extend them.

## Testing conventions

- Shared surface in `tests/conftest.py`: `REPO`, `URDF`, `USD`, `JOINT_SIGNS`, `has_pinocchio()`, `needs_pin`, and the `demo_cfg` / `rng` fixtures. Import via `from conftest import ...`.
- E2E tests build the real pipeline with `build_runtime(cfg, tmp_path / "run")` on the mock stack; control grasp outcomes with `arm.object_stop_frac` (0.5 = object held, None = air-grasp). Always poll `runtime.beliefs.find("red object")` in a deadline loop before executing skills (the watcher thread needs a detector pass), and tear down with try/finally `shutdown_runtime(runtime, arm)` — these are real threads.
- Config overrides in tests mutate `cfg._data[...]` directly; there is no override API. Servers bind port 0; `MockLLM(responses)` plays a script and records `.requests` for assertions.
- `tests/test_review_regressions*.py` is a convention: each adversarial review pass gets its own file, one test per confirmed defect with the root cause in the docstring. New review findings follow this pattern (v3 file), not silent folding into feature tests.
- `scripts/` modules are tested via `sys.path.insert(0, str(REPO / "scripts"))`, not packaging.

## Gotchas

- **Stale absolute paths**: prior revisions of `scripts/hermes_demo.sh`,
  `scripts/dashboard_runner.py`, and `scripts/setup_agents.py` hardcoded
  `/home/spark/Projects/demo/...` paths from the original 2026 rig layout.
  As of 2026-08-28 all four scripts resolve their Python interpreter and
  repository root from `Path(__file__).resolve().parents[...]` plus
  env-var overrides (`WRC_DEMO_PY`, `WRC_DETECTOR_MODEL`, etc.) — see
  the comments at the top of each script for the exact fallback chain.
  `.mcp.json` is still regen-per-machine; use `python
  scripts/setup_agents.py --host claude --python <interpreter> --write`.
  New runner scripts should take the interpreter from an env var/flag
  like `setup_agents.py` does.
- **Persistent learned state outside the run dir** changes behavior between otherwise-identical sessions: `~/.wrc_demo/grasp_memory.json` and `runs/experience.json` (tier-2 habits replay stored call lists verbatim at cosine ≥ 0.9). Delete both for deterministic/fresh demo behavior.
- YOLOE's text encoder (`mobileclip*.ts`) resolves relative to the CWD — wrong launch directory makes all detections silently vanish. In this checkout the encoder lives at `models/mobileclip2_b.ts` (or `mobileclip_blt.ts` for the larger variant), so launch from `models/` or copy the `.ts` into your launch directory; the detector weights themselves are fine anywhere (`${repo}/models/...` is absolute). Set `YOLO_OFFLINE=True` or ultralytics phones GitHub and stalls the watcher.
- Never use `==`/`list.remove()` on `ObjectBelief` — its numpy fields make dataclass `__eq__` raise ("truth value of an array is ambiguous"); remove by identity as `BeliefStore.mark_removed` does.
- Accessing any backend-specific attribute on a LazyArm-backed `SafeArm.raw` — anything not on LazyArm's own surface, including `hasattr` probes for extras like `close_gripper_two_stage` — materializes the arm and **powers the motors** via `__getattr__`. LazyArm's own members (`connected`, `stop`, `disconnect`, `resume`) are safe to touch; never probe backend capabilities outside a motion context.
- Beliefs are keyed by DETECTOR labels, not user-query words — `WRC_DETECT_CLASSES` must name what users will ask for ("banana" can't resolve a belief labeled "fruit").
- `DepthProvider.ensure_depth` mutates the shared `Frame` in place; plane-cast depth is only valid for pixels on the table (tall-object heights are wrong on RGB-only cameras).
- Ctrl+C in the demo CLI is a soft stop (freeze + e-stop latch, no free-fall); the second Ctrl+C exits. Park the arm (`move_home`) before disconnecting the RS arm — `disconnect()` disables torque and a loaded arm falls.
- Base deps pin `opencv-python-headless` — `cv2.imshow` viewers only work because the conda `wrc-demo` env installs full opencv (conda's `opencv` package includes the GUI build); in a CI/minimal install they silently degrade.
- The Isaac Sim backend was removed 2026-08-28 — see the `IsaacSim` references in `master_arm.py`'s old docstring for what NOT to do (don't re-add `sim/bridge_client.py`, `isaac_arm.py`, etc.; see git history pre-`de9703a` if you need to peek at the old code).

## Doc status (README/ARCHITECTURE/ROADMAP synced to code on 2026-07-20)

- Counts go stale fast — before citing them, re-derive: test count via `pytest tests/ -q --collect-only`, skill count via `len(TOOL_SPECS)` in `skills/runtime.py` (6 specs = 5 skills + `task_done`; MCP = 5 + 5 extras = 10 tools). `_MOTION_SKILLS` has 18 string literals (pauses WorldWatcher belief fusion during motion).
- Still true and easy to be misled by: `skills/library.py` (markdown skill library) and episodic-memory embeddings (`embed_dim`) are written and tested but **not wired** into the shipped demo; the only live TurboQuant user is tier-2 ExperienceMemory. Both docstrings/docs now say so — keep them updated if you wire them.
- (Resolved 2026-08-27) The `local_qwen` profile was deleted because it was never wired: its model name (`Qwen3VL-30B-A3B-Instruct-Q4_K_M`) had no matching serve script, and `serve_qwen_*.sh` fetched a different model. The actual production LLM is `local_qwen3_vl` (Qwen3-VL-2B-Instruct AWQ-4bit, served by `scripts/start_cosmos3_server.sh`).
- The shipped `.mcp.json` stays pinned to the demo rig on purpose — regenerate locally with `setup_agents.py --host claude --python <interpreter> --write` instead of editing the docs or the file to match this machine.

## Recent fixes (2026-07-24)

- **Mechanical-arm droop (v5 §P0-1 + P0-2)**: `RebotRSArm` now drives the SDK's own 500 Hz control loop (`arm.start_control_loop(self._loop_cb, rate=500.0)`); the callback is the SOLE writer to `arm.arm.send_mit` / `gripper.send_mit`. `joint_signs` are baked in at the read and write boundaries (`_read_positions` × signs → local_q; `send_joint_target` × signs → asset_q). Gravity feed-forward (`pin.computeGeneralizedGravity` + per-joint `gravity_scale`) is threaded as the `tau` arg to every `send_mit`. See `CONTROL_LOOP_FIX_PLAN.md`, `HANDOFF_V5_CONTROL_FIX.md`, `HARDWARE_VERIFICATION.md` (still unverified on real hardware as of 2026-07-24).
- **Skill / backend boundary (v5 §P1.3)**: `SkillRuntime` no longer reaches into `self.arm.raw` private attrs. `ArmBase.close_gripper_two_stage` and `ArmBase.resume` are concrete methods; `SafeArm` delegates both with a harness e-stop veto; `LazyArm.close_gripper_two_stage` is on the LazyArm class surface (so a `hasattr` probe never materializes a never-powered arm). `tests/test_skill_backend_boundary.py` pins this with AST-grep guards + end-to-end tests. If a new backend adds a private field a skill needs, add a public method on `ArmBase` instead.
- **Stow limits (v5 §P2.4)**: `home_q` (per arm profile) and the mechanical-zero target that `scripts/emergency_stow.py` drives are pinned to be inside URDF joint limits with > 0.02 rad safety margin via `tests/test_stow_limits.py`.

## Recent additions (2026-08-11)

- **`local_cosmos3` profile** (`configs/llm/local_cosmos3.yaml`): runs `nvidia/Cosmos3-Edge` 4B via an external vLLM server on `localhost:8080`. See `docs/adr/0007-local-llm-strategy.md` for the design and `docs/SPEC-local-llm.md` for ops. Architecture is two-env:
    - **wrc_demo (conda `wrc-demo` env)** runs unchanged — `OpenAICompatClient` just hits `http://127.0.0.1:8080/v1`.
    - **cosmos3-llm (uv venv at `~/cosmos3-venv`, Python 3.13)** runs `vllm serve` with `torch 2.13+cu130` + `transformers 5.15`. The two envs are deliberately separate because the vLLM/transformers/cu13 stack (2026-07+) is incompatible with the wrc_demo `wrc-demo` env stack (cu12, older torch). Communication is HTTP only.
  - **Don't** try to put vLLM into the `wrc-demo` env; it breaks pinned deps. Use `scripts/start_cosmos3_server.sh` (auto-patches + auto-detects driver CUDA + manages PID).
  - **Don't** use docker — the cu13 + RTX 5070 + vLLM combination works fine in a plain `uv` venv. ADR-0007 lists docker as the fallback path; we got the uv path working first.
- **8 GB VRAM tuning** (RTX 5070 / Blackwell sm120):
    - `--max-model-len 8192 --gpu-memory-utilization 0.85` (BF16 4.87 GB + KV 0.94 GB + vision 0.83 GB + 8 GB total budget — see script for details).
    - `--limit-mm-per-prompt '{"image": 0, "video": 0}'` — vision off; the 0.83 GB SigLIP plus per-image activations OOMs otherwise. Add back when a quantization lands.
    - `--enforce-eager` (skip CUDA graphs; saves ~0.5 GB).
    - `--attention-backend FLASH_ATTN` (works around a flashinfer 0.6.16 + sm120 incompatibility).
- **Tool-calling verified end-to-end** with `local_cosmos3`:
    - `tests/test_cosmos3_smoke.py::test_01_two_tools_returns_structured_tool_calls` passes — Cosmos3-Edge returns structured `tool_calls` (matches Qwen3-Coder XML template byte-for-byte; vLLM serves it via `--tool-call-parser qwen3_xml`).
    - `tests/test_cosmos3_smoke.py::test_02_picks_correct_tool_among_N` passes — model picks `list_objects` from `len(TOOL_SPECS)` (currently 6) correctly. *(legacy cosmos3 entry; test file no longer in tree — kept here as historical record of the verification done in 2026-07)*
    - Full wrc_demo mock-stack integration: `python -m wrc_demo.apps.demo --llm local_cosmos3 --task wave` → reflex 0.03s; `--task "list the objects"` → LLM path, 30 steps, exhausts step budget on empty mock scene (correct behavior).
    - **Caveat**: model often emits empty arguments (`{}`) for tool calls — template-level wiring is correct, but per-tool argument filling is mid-quality. If this blocks production, swap `Cosmos3-Edge` for a Qwen3-Coder fine-tune (ADR-0007 §"Fallback plan").
- **3 required patches** that the script applies automatically (idempotent; safe to re-run):
    - `vllm/transformers_utils/processors/cosmos3_edge.py`: import `get_image_size` from `transformers.image_utils` instead of `transformers.models.qwen3_vl.video_processing_qwen3_vl` (the symbol moved in transformers v5; vLLM 0.27 + transformers v5 collision).
    - `flashinfer/jit/core.py::check_cuda_arch` + `flashinfer/compilation_context.py::__init__`: seed `TARGET_CUDA_ARCHS` with sm_120 when empty (flashinfer 0.6.16 reports empty on RTX 5070 / Blackwell).
    - `flashinfer/compilation_context.py`: skip `SM 12.x requires CUDA >= 12.9` check (we have cu13 toolchain via PATH).
- **minimax API key migrated** to env var (ADR-0006): `configs/llm/minimax.yaml` now uses `env_key: MINIMAX_API_KEY`. The old plaintext key is gone; **rotate the key** on the provider side after this lands (the prior commit history may still contain it locally).
- **LLM profile switching + fallback** (ADR-0008): `AgentOrchestrator.switch_llm(name)` and `_chat_with_fallback()` — both backward-compatible (optional kwargs). Wire-up via `cfg.llm.fallback: [local_cosmos3, mock]` in `demo.yaml`. New `LLMConfigError` in `agent/llm.py` raises at startup if a fallback profile's `env_key` is unset, surfacing the env var name in the message.
- **`TraceLogger.record_event(kind, payload)`** added (was missing): used by `_chat_with_fallback` to log `llm_fallback`, `llm_fallback_build_failed`, `llm_switch` events. Backward-compatible — `record()` and `finish()` unchanged.

## Recent additions (2026-08-14) — Teach-Record/Replay skills (v2: master-slave teleop)

- **Teach-Record / Teach-Replay skills** (`src/wrc_demo/skills/teach.py` + `master_arm.py` + `runtime.py`): LLM-callable demonstration flow via **PiPER Mate → reBot B601-RS** master-slave teleoperation.
  - `skill_teach_record(slot: int) -> dict`: open PiPER Mate master arm (`/dev/ttyUSB0` via `fashionstar-uart-sdk`), connect reBot slave arm in POS_VEL mode. Capture `master_q` + `slave_q` (7 floats each, radians) at 30 Hz to `data/recordings/slot{N}.json`. User presses `q` to start/stop recording into the given slot, `Esc` to cancel. Long-running; pauses WorldWatcher belief fusion via `_MOTION_SKILLS`.
  - `skill_teach_replay(slot: int) -> dict`: read slot JSON, execute `slave_q` trajectory via `RebotArmEndPose.move_joints_traj`, return to motor mechanical zero via `safe_home`. Returns `{"replayed_to_home": True}`.
  - **Data schema v2**: `{schema_version: 2, frames[i]: {t, master_q[7], slave_q[7]}}`. **Incompatible with v1** (single `joints` field — deleted). Old slot1-4.json were removed on 2026-08-14.
  - **Master→slave mapping** is **hardcoded** in `master_arm.py::master_to_slave()` (no calibration, per user decision 2026-08-14): joint1/4/6 方向取反 + URDF joint limits clip + gripper × `GRIPPER_RATIO=6`. Joint limits in `JOINT_ANGLE_LIMITS` derived from **URDF** (`sdk/urdf/00-arm-rs_asm-v3/...`), not the demo's mismatched values.
  - Added to `_MOTION_SKILLS` (16 → 18 string literals) and `TOOL_SPECS` (4 → 6 entries, descriptions updated to mention "master-slave teleoperation" + "PiPER Mate"). Current counts (2026-08-27): `len(TOOL_SPECS) == 6`, `len(_MOTION_SKILLS) == 18`, MCP tools = 10 (6 - 1 + 5 extras). Re-derive via these expressions before citing.
  - Bypasses `SafeArm` harness by reaching `self.arm.raw` → vendored `RebotArm` directly, because teach needs an external-event-triggered control loop (SafeArm only exposes `move_joints` trajectory streaming). Acceptable because slave uses POS_VEL mode (motor PID handles trajectory, not SafeArm).
  - **New dependency**: `fashionstar-uart-sdk` for PiPER Mate. Already installed by `environment.yml` (see `[perception,llm,kinematics,arm,dev]` extras + the standalone `fashionstar-uart-sdk` pin in the pip block). If installing manually outside the standard env: `pip install fashionstar-uart-sdk`.
  - Examples: `examples/teach_record.py` and `examples/teach_replay.py` — both thin wrappers over `teach.py` business logic, support `--dry-run` mode for logic testing without hardware.
  - Tests: `tests/test_master_arm.py` (23 tests, 100% line coverage of `master_arm.py::master_to_slave`) + `tests/test_teach_logic.py` (27 tests, ≥ 80% coverage of `teach.py` business logic) + `tests/test_teach_skills.py` (17 tests, 100% coverage of skill entry points). **Total: 67 new tests, all passing**. Pre-existing 29 failures unchanged — no regression introduced.
  - Spec: `docs/specs/teach_record_replay.md`. Plan: `tasks/plan.md` + `tasks/todo.md`. **Do not modify `wrc_demo/sdk/reBotArm_control_py/`** — vendored in-tree snapshot, upstream merge would overwrite.

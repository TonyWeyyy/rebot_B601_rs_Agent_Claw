#!/usr/bin/env bash
# Start vLLM serving the local Qwen3-VL AWQ-4bit model for wrc_demo.
#
# This is the **canonical** bring-up for the `local_qwen3_vl` LLM profile
# (see configs/llm/local_qwen3_vl.yaml). wrc_demo talks to the served model
# over OpenAI-compatible HTTP at http://localhost:8080/v1 — vLLM runs in its
# own environment because the vLLM stack (cu13 + torch 2.13 + transformers
# v5) is incompatible with wrc_demo's main environment.
#
# Architecture (ADR-0007):
#
#   ┌─ wrc_demo (wrc-demo conda env) ─┐  HTTP  ┌─ vllm (this script) ─────┐
#   │  OpenAICompatClient              │        │  vllm 0.27 (FLASH_ATTN)  │
#   │    http://localhost:8080/v1     ├───────>│  Qwen3-VL-2B-AWQ-4bit   │
#   │  --llm local_qwen3_vl           │        │  hermes tool parser     │
#   └─────────────────────────────────┘        └─────────────────────────┘
#
# Usage:
#   bash scripts/start_cosmos3_server.sh           # start in foreground
#   bash scripts/start_cosmos3_server.sh --bg      # start in background
#   bash scripts/start_cosmos3_server.sh --stop    # stop a running server
#   bash scripts/start_cosmos3_server.sh --status  # check if running
#
# Environment variables (override defaults):
#   VLLM_VENV_BIN       path to the vllm venv's bin/  (default: ~/cosmos3-venv/bin)
#   VLLM_MODEL          path to or HF id of the model (default: <repo>/models/qwen3-vl-2b-awq-4bit)
#   VLLM_PORT           port to bind                       (default: 8080)
#   VLLM_MAX_LEN        --max-model-len                    (default: 2560)
#   VLLM_GPU_UTIL       --gpu-memory-utilization           (default: 0.65)
#   VLLM_LOG_FILE       log file path                      (default: /tmp/vllm_cosmos3.log)
#   VLLM_PID_FILE       pid file path                      (default: /tmp/vllm_cosmos3.pid)
#
# On the Seeed  rig the special cu13/ptxas symlinks are required for RTX
# 5070 / Blackwell sm120 (Triton JIT needs cu13 ptxas). Other GPUs that
# already have a working CUDA toolchain on PATH don't need any of those.
#
# Prerequisites (one-time setup, on the Seeed  rig):
#   1. uv venv ~/cosmos3-venv --python 3.13
#   2. source ~/cosmos3-venv/bin/activate
#   3. uv pip install --torch-backend=auto "vllm>=0.23.0" "transformers>=5.11.0"
#   4. hf auth login (token in ~/.cache/huggingface/stored_tokens)
#   5. hf download QwenLM/Qwen3-VL-2B-Instruct-AWQ-4bit \
#         --local-dir <repo>/models/qwen3-vl-2b-awq-4bit
#      (or let vLLM auto-pull on first start)
#
# Once running, wrc_demo can use:
#   python -m wrc_demo.apps.demo --llm local_qwen3_vl ...
# (no Docker, no extra environment juggling beyond the vllm venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configurable paths (env-var override; see header) ──────────────────
VLLM_VENV_BIN="${VLLM_VENV_BIN:-$HOME/cosmos3-venv/bin}"
VLLM_MODEL="${VLLM_MODEL:-$REPO_DIR/models/qwen3-vl-2b-awq-4bit}"
VLLM_PORT="${VLLM_PORT:-8080}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-2560}"        # 8 GB VRAM: longer KV explodes; raise on bigger GPUs
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.65}"      # X server / gnome reserve ~2 GB
VLLM_LOG_FILE="${VLLM_LOG_FILE:-/tmp/vllm_cosmos3.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-/tmp/vllm_cosmos3.pid}"

# ── Seeed  rig-only: cu13 nvcc + triton ptxas ──────────────────────────
# System nvcc 11.5 doesn't support sm_120; we symlink the cu13 nvcc into a
# per-launch bin dir so it shadows PATH for this script only. Triton JIT
# also needs cu13 ptxas (TRITON_PTXAS_PATH).
CUSTOM_BIN="${CUSTOM_BIN:-$HOME/cosmos3-venv/bin-custom}"
NVCC13_BIN="${NVCC13_BIN:-$HOME/.local/lib/python3.10/site-packages/nvidia/cu13/bin}"
PTXAS_BIN="${PTXAS_BIN:-$HOME/cosmos3-venv/lib/python3.13/site-packages/nvidia/cu13/bin}"

# ── Subcommand dispatch ───────────────────────────────────────────────
case "${1:-start}" in
    --stop|stop)
        if [[ -f "$VLLM_PID_FILE" ]]; then
            pid=$(cat "$VLLM_PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                echo "[stop] killing vllm PID $pid"
                kill "$pid" 2>/dev/null || true
                sleep 2
                kill -9 "$pid" 2>/dev/null || true
                rm -f "$VLLM_PID_FILE"
                echo "[stop] done"
            else
                echo "[stop] PID $pid not running (stale PID file)"
                rm -f "$VLLM_PID_FILE"
            fi
        else
            echo "[stop] no PID file at $VLLM_PID_FILE"
        fi
        exit 0
        ;;
    --status|status)
        if [[ -f "$VLLM_PID_FILE" ]] && kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
            echo "[status] vllm running, PID $(cat "$VLLM_PID_FILE")"
            curl -sf "http://localhost:$VLLM_PORT/v1/models" 2>&1 | head -c 200
            echo
        else
            echo "[status] vllm not running"
            exit 1
        fi
        exit 0
        ;;
esac

# ── Pre-flight ────────────────────────────────────────────────────────
echo "=== start_cosmos3_server.sh (Qwen3-VL-2B AWQ-4bit) ==="
echo "venv:  $VLLM_VENV_BIN"
echo "model: $VLLM_MODEL"
echo "port:  $VLLM_PORT"

if [[ ! -x "$VLLM_VENV_BIN/python" ]]; then
    echo "" >&2
    echo "ERROR: vllm venv not found at $VLLM_VENV_BIN" >&2
    echo "  Create it with:" >&2
    echo "    uv venv ~/cosmos3-venv --python 3.13" >&2
    echo "    source ~/cosmos3-venv/bin/activate" >&2
    echo "    uv pip install --torch-backend=auto \"vllm>=0.23.0\" \"transformers>=5.11.0\"" >&2
    echo "  Or point at an existing vllm install via VLLM_VENV_BIN." >&2
    exit 1
fi

# nvidia cu13 nvcc symlink (system nvcc 11.5 doesn't support sm_120).
# Skip silently if the cu13 path doesn't exist (other GPUs don't need it).
if [[ -x "$NVCC13_BIN/nvcc" ]]; then
    mkdir -p "$CUSTOM_BIN"
    ln -sf "$NVCC13_BIN/nvcc" "$CUSTOM_BIN/nvcc" 2>/dev/null || true
    if [[ -x "$CUSTOM_BIN/nvcc" ]]; then
        echo "cu13 nvcc: $($CUSTOM_BIN/nvcc --version | tail -1)"
    fi
else
    CUSTOM_BIN=""
fi

# ptxas for triton JIT (RTX 5070 / Blackwell sm120).
if [[ -x "$PTXAS_BIN/ptxas" ]]; then
    echo "ptxas: $($PTXAS_BIN/ptxas --version 2>&1 | head -1)"
else
    PTXAS_BIN=""
fi

# already running?
if [[ -f "$VLLM_PID_FILE" ]] && kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
    echo "vllm already running, PID $(cat "$VLLM_PID_FILE")"
    echo "to restart: bash $0 --stop"
    exit 0
fi

# free GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 || echo 0)
    GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo 0)
    if [[ "$GPU_TOTAL_MIB" -gt 0 ]]; then
        GPU_FREE_GB=$(awk -v t="$GPU_TOTAL_MIB" -v u="$GPU_USED_MIB" 'BEGIN { printf "%.1f", (t-u)/1024 }')
        echo "GPU free: ${GPU_FREE_GB} GB"
        if (( $(echo "$GPU_FREE_GB < 5.0" | bc -l 2>/dev/null || echo 0) )); then
            echo "WARN: only ${GPU_FREE_GB} GB free; AWQ-4bit needs ~3-4 GB (weights + KV + vision)"
            echo "      kill X server or other GPU consumers to free more"
        fi
    fi
fi

# ── Build env ─────────────────────────────────────────────────────────
# PATH order: custom (cu13 nvcc) → nvidia cu13 bin (for cu13 toolchain) → system
EXTRA_PATH=""
[[ -n "$CUSTOM_BIN" ]] && EXTRA_PATH="$CUSTOM_BIN"
[[ -n "$NVCC13_BIN" ]] && EXTRA_PATH="$EXTRA_PATH:$NVCC13_BIN"
[[ -n "$PTXAS_BIN" ]] && EXTRA_PATH="$EXTRA_PATH:$PTXAS_BIN"
export PATH="$EXTRA_PATH:$PATH"

# Point triton at the cu13 ptxas if we have one (system /usr/local/cuda/bin/ptxas
# is missing on the Blackwell rigs).
if [[ -n "$PTXAS_BIN" ]]; then
    export TRITON_PTXAS_PATH="$PTXAS_BIN/ptxas"
fi

# HF_TOKEN: legacy token file is the canonical source. INI parser in
# stored_tokens can mis-match (key name vs token value); fall back to legacy.
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN HF_HUB_TOKEN="$HF_TOKEN"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"  # allow fetching auxiliary models on first run

# ── Start ─────────────────────────────────────────────────────────────
# Flag notes (2026-08-12, Seeed  RTX 5070):
#   --reasoning-parser qwen3         reason2 is a Qwen3-VL; reasoning chain in <think> tags
#   --max-model-len 2560             8 GB VRAM budget; raise on bigger GPUs
#   --gpu-memory-utilization 0.65    leave 35% for X server / CUDA context / fragmentation
#   --limit-mm-per-prompt image=0    vision SigLIP + per-image activations OOMs otherwise
#   --enforce-eager                  skip CUDA graphs (saves ~0.5 GB; sm120 graph capture flaky)
#   --attention-backend FLASH_ATTN   flashinfer 0.6.16 has a sm120 incompatibility
#   --host 0.0.0.0                   bind all interfaces (wrc_demo talks to localhost)
#   --tool-call-parser hermes         Qwen3-Coder XML tool protocol (matches hermes SDK)
VLLM_FLAGS=(
    --model "$VLLM_MODEL"
    --port "$VLLM_PORT"
    --max-model-len "$VLLM_MAX_LEN"
    --gpu-memory-utilization "$VLLM_GPU_UTIL"
    --enforce-eager
    --enable-auto-tool-choice
    --tool-call-parser hermes
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
    --attention-backend FLASH_ATTN
    --host 0.0.0.0
)

echo "[start] vllm serve $VLLM_MODEL (max_len=$VLLM_MAX_LEN, gpu_util=$VLLM_GPU_UTIL)"
echo "[log]   $VLLM_LOG_FILE"

if [[ "${1:-}" == "--bg" ]]; then
    nohup "$VLLM_VENV_BIN/python" -m vllm.entrypoints.openai.api_server \
        "${VLLM_FLAGS[@]}" \
        > "$VLLM_LOG_FILE" 2>&1 &
    echo $! > "$VLLM_PID_FILE"
    echo "[bg] PID $(cat "$VLLM_PID_FILE")"
    echo "[bg] wait ~60s, then: bash $0 --status"
else
    exec "$VLLM_VENV_BIN/python" -m vllm.entrypoints.openai.api_server \
        "${VLLM_FLAGS[@]}"
fi
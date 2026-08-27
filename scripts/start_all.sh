#!/usr/bin/env bash
# One-shot bring-up: vLLM + wrc_demo (interactive).
#
# Usage:
#   bash scripts/start_all.sh                            # mock stack, no sudo
#   bash scripts/start_all.sh --real                     # real hardware (needs sudo)
#   bash scripts/start_all.sh --real --task "wave"       # real + single task
#   bash scripts/start_all.sh --stop                      # stop everything
#   bash scripts/start_all.sh --status                    # check status
#
# Architecture (ADR-0007, updated 2026-08-14 to use Qwen3-VL):
#   cosmos3-venv (uv)        wrc_demo (DM env)
#   ┌──────────────┐         ┌──────────────┐
#   │   vLLM 0.27  │  HTTP   │  wrc_demo    │
#   │  Qwen3-VL-2B │◄────────┤  OpenAICompat│
#   │   AWQ-INT4   │ :8080  │  Client      │
#   └──────────────┘         └──────────────┘
#
# vLLM lives in its own uv venv because cu13 + transformers v5 + vllm
# 0.27 are incompatible with the wrc_demo DM env (cu12, older stacks).
# They talk via the OpenAI HTTP API on localhost:8080.
#
# Default profile is `local_qwen3_vl` (Qwen3-VL-2B-Instruct AWQ-4bit,
# AWQ INT4 fits on 8 GB and natively supports tool calling). Override
# with `--llm <profile>` after editing the DEMO_ARGS line below.
#
# See ADR-0007 / docs/SPEC-local-llm.md for details.

set -euo pipefail

# Resolve the vLLM script relative to this script's location so the launcher
# works regardless of where the repo is checked out. Override with WRC_VLLM_SCRIPT.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_SCRIPT="${WRC_VLLM_SCRIPT:-$SCRIPT_DIR/start_cosmos3_server.sh}"
VLLM_PID_FILE="/tmp/vllm_cosmos3.pid"
VLLM_LOG="/tmp/vllm_cosmos3.log"
DEMO_PID_FILE="/tmp/wrc_demo.pid"
DEMO_LOG="/tmp/wrc_demo.log"

# ── Subcommand dispatch ───────────────────────────────────────────
case "${1:-start}" in
    --stop|stop)
        # Stop demo first, then vllm
        if [[ -f "$DEMO_PID_FILE" ]]; then
            pid=$(cat "$DEMO_PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                echo "[stop] killing wrc_demo PID $pid"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$DEMO_PID_FILE"
        fi
        pkill -f 'wrc_demo.apps.demo' 2>/dev/null || true
        pkill -f 'wrc_demo.apps.mcp' 2>/dev/null || true
        if [[ -x "$VLLM_SCRIPT" ]]; then
            bash "$VLLM_SCRIPT" --stop 2>&1 | head -3
        fi
        echo "[stop] done"
        exit 0
        ;;
    --status|status)
        echo "=== vLLM (cosmos3-llm @ :8080) ==="
        if [[ -f "$VLLM_PID_FILE" ]] && kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
            echo "  running, PID $(cat "$VLLM_PID_FILE")"
            curl -sf http://localhost:8080/v1/models 2>&1 | head -c 200
            echo
        else
            echo "  NOT running"
        fi
        echo ""
        echo "=== wrc_demo ==="
        if [[ -f "$DEMO_PID_FILE" ]] && kill -0 "$(cat "$DEMO_PID_FILE")" 2>/dev/null; then
            echo "  running, PID $(cat "$DEMO_PID_FILE")"
        else
            echo "  NOT running"
        fi
        exit 0
        ;;
esac

# ── Start mode ─────────────────────────────────────────────────────
MODE="${1:-mock}"   # --real, mock, or first arg = task string
shift || true
TASK=""
NEED_SUDO=0
CAMERAS="mock"
ARM="mock"

case "$MODE" in
    --real|real)
        NEED_SUDO=1
        CAMERAS="orbbec_overhead"
        ARM="rebot_rs"
        ;;
    *)
        # Treat $MODE as the task string (one-shot mode)
        TASK="$MODE"
        ;;
esac

# Optional positional --task
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "  wrc_demo + Qwen3-VL-AWQ-4bit one-shot launcher"
echo "============================================================"
echo "  mode:    $([[ $NEED_SUDO -eq 1 ]] && echo 'real hardware' || echo 'mock stack')"
echo "  camera:  $CAMERAS"
echo "  arm:     $ARM"
echo "  task:    ${TASK:-<interactive>}"
echo ""

# ── Step 1: ensure vLLM is running ───────────────────────────────
echo "[1/3] check vLLM..."
if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "  vLLM already running (skipped)"
else
    echo "  vLLM not running, starting..."
    # The legacy script name "start_cosmos3_server.sh" is preserved because
    # operation docs and many user scripts reference it; it now serves the
    # Qwen3-VL AWQ model (see ADR-0007 + §17/§18 of wrc_demo_操作文档.md).
    MAX_LEN=8192 GPU_UTIL=0.85 bash "$VLLM_SCRIPT" --bg
    # Wait for readiness
    for i in {1..40}; do
        sleep 3
        if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
            echo "  vLLM ready (after ${i}*3s)"
            break
        fi
        if [[ $i -eq 40 ]]; then
            echo "  vLLM failed to start in 120s; see $VLLM_LOG"
            exit 1
        fi
    done
fi

# ── Step 2: launch wrc_demo ─────────────────────────────────────
echo ""
echo "[2/3] start wrc_demo..."
RUN_DIR="${WRC_RUN_DIR:-$REPO_DIR/runs/live_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

# Pick a Python interpreter. Prefer $WRC_DEMO_PY; fall back to the conda DM env
# path (project convention on the demo rig); fall back to whatever `python` is on
# PATH (e.g. uv-managed .venv) so the launcher also works on a fresh checkout.
WRC_DEMO_PY="${WRC_DEMO_PY:-${CONDA_PREFIX:-}/bin/python}"
if [[ -z "${CONDA_PREFIX:-}" ]] && [[ ! -x "$WRC_DEMO_PY" ]]; then
    WRC_DEMO_PY="$(command -v python)"
fi

# Default LLM profile is local_qwen3_vl (current production model — Qwen3-VL-2B
# AWQ-4bit served by start_cosmos3_server.sh). Override with --llm <profile>.
DEMO_CMD="$WRC_DEMO_PY -u -m wrc_demo.apps.demo \
    --cameras $CAMERAS --arm $ARM --llm local_qwen3_vl \
    --no-view"

if [[ -n "$TASK" ]]; then
    DEMO_CMD="$DEMO_CMD --no-serve --task \"$TASK\""
else
    DEMO_CMD="$DEMO_CMD --interactive"
fi

# Build demo args as a bash array (preserves quoting of $TASK).
DEMO_ARGS=(--cameras "$CAMERAS" --arm "$ARM" --llm local_qwen3_vl --no-view)
if [[ -n "$TASK" ]]; then
    DEMO_ARGS+=(--no-serve --task "$TASK")
else
    DEMO_ARGS+=(--interactive)
fi

if [[ $NEED_SUDO -eq 1 ]]; then
    # Real hardware: needs sudo for CAN bus + WRC_STREAM=1, WRC_VIEW=1.
    # `nohup sudo` can't prompt for a password (no tty). Require the
    # operator to have cached sudo credentials first via `sudo -v` in an
    # interactive terminal; we verify non-interactively with `sudo -n true`.
    if ! sudo -n true 2>/dev/null; then
        echo ""
        echo "  ERROR: --real needs sudo (CAN bus, etc.) but no cached credentials."
        echo "  Run this in an interactive terminal first to cache sudo:"
        echo ""
        echo "      sudo -v"
        echo ""
        echo "  Then re-run: bash $0 --real ${TASK:+--task \"$TASK\"}"
        exit 1
    fi
    nohup sudo env HOME="$HOME" YOLO_OFFLINE=True \
        PYTHONPATH="$REPO_DIR/src" \
        WRC_RUN_DIR="$RUN_DIR" WRC_STREAM=1 WRC_VIEW=1 \
        "$WRC_DEMO_PY" -u -m wrc_demo.apps.demo \
        "${DEMO_ARGS[@]}" > "$DEMO_LOG" 2>&1 &
else
    nohup env HOME="$HOME" YOLO_OFFLINE=True \
        PYTHONPATH="$REPO_DIR/src" \
        WRC_RUN_DIR="$RUN_DIR" \
        "$WRC_DEMO_PY" -u -m wrc_demo.apps.demo \
        "${DEMO_ARGS[@]}" > "$DEMO_LOG" 2>&1 &
fi

DEMO_PID=$!
echo $DEMO_PID > "$DEMO_PID_FILE"
echo "  wrc_demo PID: $DEMO_PID"
echo "  log:          $DEMO_LOG"
echo "  run dir:      $RUN_DIR"

# ── Step 3: wait + summary ───────────────────────────────────────
echo ""
echo "[3/3] wait for demo to start..."
for i in {1..30}; do
    sleep 2
    if ! kill -0 $DEMO_PID 2>/dev/null; then
        echo "  ERROR: wrc_demo died early. Last 30 lines of log:"
        tail -30 "$DEMO_LOG" 2>&1
        exit 1
    fi
    # Check log for "task>" or completed task report
    if grep -q "task> \|task report\|success: True\|step budget" "$DEMO_LOG" 2>/dev/null; then
        break
    fi
done

echo ""
echo "============================================================"
echo "  ✓ READY"
echo "============================================================"
echo "  wrc_demo:    http://localhost:8090/  (dashboard, if --no-view not set)"
echo "  vLLM:        http://localhost:8080/v1  (Qwen3-VL-2B-AWQ-4bit via start_cosmos3_server.sh)"
echo "  log:         $DEMO_LOG"
echo "  tail:        tail -f $DEMO_LOG"
echo "  status:      bash $0 --status"
echo "  stop:        bash $0 --stop"
echo "============================================================"

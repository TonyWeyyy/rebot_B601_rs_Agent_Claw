#!/usr/bin/env bash
# Backwards-compatible alias for start_reason2_server.sh.
#
# History (2026-08-12): this script used to launch nvidia/Cosmos3-Edge 4B BF16
# via vLLM. After evaluating Cosmos-Reason2-2B, Qwen3-VL FP8, and Qwen3-VL
# AWQ-4bit, the project settled on the AWQ-4bit variant (the only one that
# fits in 8 GB and supports tool calling natively — see ADR-0007 / 0008 and
# §17-§19 of wrc_demo_操作文档.md).
#
# Both this script and start_reason2_server.sh are kept because operation
# docs reference both names. The real implementation now lives in
# start_reason2_server.sh; this file is a thin wrapper so existing commands
# like `bash scripts/start_cosmos3_server.sh --bg` keep working.
#
# Once running, wrc_demo can use either LLM profile:
#     python -m wrc_demo.apps.demo --llm local_qwen3_vl ...
# (local_cosmos3 was retired — its model and yaml have been removed; see
# the cleaned-up configs/llm/ directory.)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/start_reason2_server.sh" "$@"
#!/usr/bin/env python3
"""Download / verify the bundled-model artifacts required by wrc-demo.

This script is the **blessed entry point** for getting model weights onto
disk. The ``models/`` directory is gitignored (see ``.gitignore``:
``models/*``), so a fresh clone has no weights. This script:

  1. Lists which weights are needed (per ``configs/demo.yaml``).
  2. Checks what's already on disk and reports size + mtime.
  3. Fetches the ones that can be auto-downloaded from public mirrors
     (HuggingFace, ultralytics).
  4. Prints manual-instruction hints for the ones that can't (custom
     YOLOE weights from rebot_grasp, qwen3-vl AWQ-4bit, etc.).

Usage::

    python scripts/setup_models.py                 # list / verify
    python scripts/setup_models.py --fetch       # also auto-download
    python scripts/setup_models.py --profile all # download everything
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

# ── Manifest ──────────────────────────────────────────────────────────────
# Each entry: (relpath, expected_min_bytes, source_hint, auto_fetch_url_or_None)
MANIFEST = [
    {
        "path": "qwen3-vl-2b-awq-4bit/config.json",
        "min_mb": 0,
        "use": "local LLM (default profile local_qwen3_vl); vLLM auto-downloads if missing",
        "auto": None,
        "manual": (
            "Either let vLLM auto-pull on first start_cosmos3_server.sh, or:
"
            "  huggingface-cli download QwenLM/Qwen3-VL-2B-Instruct-AWQ-4bit \
"
            "    --local-dir models/qwen3-vl-2b-awq-4bit \
"
            "    --local-dir-use-symlinks False
"
            "~2.4 GB total."
        ),
    },

    {
        "path": "yoloe-11s-seg.pt",
        "min_mb": 20,
        "use": "default YOLOE detector (configs/demo.yaml)",
        "auto": None,
        "manual": (
            "Copy from rebot_grasp/models/yoloe-11s-seg.pt (sibling project); "
            "ultralytics hub does NOT redistribute the text-prompt variant. "
            "License: AGPL-3.0 (Ultralytics)."
        ),
    },
    {
        "path": "yoloe-26l-seg.pt",
        "min_mb": 70,
        "use": "heavier YOLOE for better banana/cyan-toy recognition",
        "auto": None,
        "manual": "Same source as 11s; pick one or the other.",
    },
    {
        "path": "mobileclip2_b.ts",
        "min_mb": 200,
        "use": "YOLOE text encoder (torchscript)",
        "auto": "https://huggingface.co/qualcomm/MobileCLIP2/resolve/main/mobileclip2_b.pt?download=true",
        "manual": (
            "If the auto-fetch fails, download manually from the Qualcomm "
            "MobileCLIP2 HuggingFace repo. The .pt source must be exported "
            "to torchscript (.ts) using ``torch.jit.trace`` against YOLOE's "
            "expected input shape."
        ),
    },
    {
        "path": "mobileclip_blt.ts",
        "min_mb": 500,
        "use": "Larger YOLOE text encoder variant",
        "auto": None,
        "manual": "Same provenance as mobileclip2_b.ts; export from .pt.",
    },
]


def _human_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> bool:
    print(f"  fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            with dst.open("wb") as f:
                shutil.copyfileobj(r, f)
        print(f"  wrote {_human_mb(dst.stat().st_size)} -> {dst}")
        return True
    except Exception as e:
        print(f"  !! fetch failed: {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fetch", action="store_true", help="Auto-download what we can")
    p.add_argument("--profile", choices=["min", "all"], default="min",
                   help="min = default detector + smallest text encoder; all = everything")
    args = p.parse_args()

    MODELS.mkdir(exist_ok=True)

    n_ok = n_missing = n_fetched = 0
    for entry in MANIFEST:
        path = MODELS / entry["path"]
        min_bytes = entry["min_mb"] * 1024 * 1024
        print(f"\n• {entry['path']}  [{entry['min_mb']} MB minimum]  — {entry['use']}")
        if path.exists():
            size = path.stat().st_size
            if size >= min_bytes:
                print(f"  ✓ present ({_human_mb(size)})")
                n_ok += 1
                continue
            print(f"  ⚠ present but only {size} bytes (expected >= {min_bytes})")
        n_missing += 1
        if args.fetch and entry["auto"]:
            if _download(entry["auto"], path):
                n_fetched += 1
        else:
            print(f"  → manual: {entry['manual']}")

    print(f"\nSummary: {n_ok} present, {n_missing} missing"
          + (f", {n_fetched} fetched" if args.fetch else ""))
    if n_missing and not args.fetch:
        print("Re-run with --fetch to auto-download what we can; the rest")
        print("need manual steps (see hints above).")
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

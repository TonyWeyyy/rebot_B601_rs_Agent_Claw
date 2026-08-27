"""WRC agentic grasping demo.

Camera-agnostic perception -> short-horizon episodic memory -> LLM agent
orchestration -> safety-gated arm control, for the Seeed reBot DevArm B601
(RobStride build) driven from a DGX Spark.
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Vendored SDK: reBotArm_control_py ships in-tree under wrc_demo/sdk/ so the
# project is self-contained (no external absolute paths, no PYTHONPATH edits).
# Injecting here means any later `import reBotArm_control_py.xxx` works
# regardless of which module triggers the wrc_demo import first.
_SDK_ROOT = _REPO_ROOT / "sdk"
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))


# ── ultralytics text-encoder pin ─────────────────────────────────────────────
# YOLOE's text encoder (``mobileclip2_b.ts`` / ``mobileclip_blt.ts``) is
# resolved by ultralytics as a CWD-relative path: if the CWD doesn't already
# contain the file, ``attempt_download_asset`` falls through to a GitHub
# download that stalls the watcher on a slow link. The detector WEIGHTS are
# already pinned to an absolute ${repo}/models/... path; this patch mirrors
# that for the text encoder so demo launch is CWD-independent.
_PINNED_TEXT_ENCODERS = {"mobileclip2_b.ts", "mobileclip_blt.ts"}
try:
    from ultralytics.utils import downloads as _ultralytics_downloads

    _orig_attempt_download_asset = _ultralytics_downloads.attempt_download_asset

    def _attempt_download_asset_pinned(file, *args, **kwargs):
        from pathlib import Path as _P
        name = _P(str(file)).name
        if name in _PINNED_TEXT_ENCODERS:
            for candidate in (_REPO_ROOT / "models" / name, _REPO_ROOT / name):
                if candidate.exists():
                    return str(candidate)
        return _orig_attempt_download_asset(file, *args, **kwargs)

    _ultralytics_downloads.attempt_download_asset = _attempt_download_asset_pinned
except ImportError:
    pass  # ultralytics not installed (mock stack / unit tests)


del Path, sys

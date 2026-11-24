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


# ── ultralytics text-encoder pin (LAZY) ─────────────────────────────────────
# YOLOE's text encoder (``mobileclip2_b.ts`` / ``mobileclip_blt.ts``) is
# resolved by ultralytics as a CWD-relative path: if the CWD doesn't already
# contain the file, ``attempt_download_asset`` falls through to a GitHub
# download that stalls the watcher on a slow link. The detector WEIGHTS are
# already pinned to an absolute ${repo}/models/... path; this patch mirrors
# that for the text encoder so demo launch is CWD-independent.
#
# IMPORTANT: this import is **lazy** (`_patch_ultralytics_text_encoder_pin`
# runs the first time any caller touches `attempt_download_asset`). Doing
# the `import ultralytics` at wrc_demo package-load would force every
# `import wrc_demo` to load torch — and if the active Python interpreter
# has a torch with a CUDA ABI that doesn't match the host's
# `libcudart.so.<N>` (e.g. torch cu13 + system without cu13 runtime),
# the whole `wrc_demo` package becomes importable even from pure-mock
# test paths like `tests/conftest.py::load_demo_config`. Lazy-import keeps
# the mock stack completely free of torch + ultralytics.
_PINNED_TEXT_ENCODERS = {"mobileclip2_b.ts", "mobileclip_blt.ts"}
_orig_attempt_download_asset = None
_ultralytics_patch_applied = False


def _patch_ultralytics_text_encoder_pin(mod=None):
    """Monkey-patch ultralytics' asset downloader (lazy, idempotent).

    Loads `ultralytics.utils.downloads` once (or patches the passed module
    object directly from the import hook), swaps in the CWD-independent
    resolver for the pinned text encoders, and never touches it again.
    Silently no-ops if ultralytics isn't installed (mock stack / unit
    tests where the detector is MockDetector anyway).
    """
    global _orig_attempt_download_asset, _ultralytics_patch_applied
    if _ultralytics_patch_applied:
        return True
    if mod is None:
        try:
            from ultralytics.utils import downloads as mod
        except ImportError:
            return False  # ultralytics not installed; nothing to patch
    if not hasattr(mod, "attempt_download_asset"):
        return False
    _ultralytics_patch_applied = True
    _orig_attempt_download_asset = mod.attempt_download_asset

    def _attempt_download_asset_pinned(file, *args, **kwargs):
        from pathlib import Path as _P
        name = _P(str(file)).name
        if name in _PINNED_TEXT_ENCODERS:
            for candidate in (_REPO_ROOT / "models" / name, _REPO_ROOT / name):
                if candidate.exists():
                    return str(candidate)
        return _orig_attempt_download_asset(file, *args, **kwargs)

    mod.attempt_download_asset = _attempt_download_asset_pinned
    return True


def _install_ultralytics_import_hook() -> None:
    """Install a lightweight sys.meta_path hook that applies the text-encoder
    pin ONLY when ``ultralytics.utils.downloads`` is actually imported.

    Zero overhead and zero torch/ultralytics imports on the mock stack;
    automatic pinning on any path that imports ultralytics.
    """
    if "ultralytics.utils.downloads" in sys.modules:
        _patch_ultralytics_text_encoder_pin(sys.modules["ultralytics.utils.downloads"])
        return

    import importlib.machinery

    class _UltralyticsPinFinder:
        _active = False

        @classmethod
        def find_spec(cls, fullname, path, target=None):
            if fullname != "ultralytics.utils.downloads" or cls._active:
                return None
            cls._active = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                cls._active = False
            if spec is None or spec.loader is None:
                return spec
            orig_loader = spec.loader

            class _PatchedLoader:
                def create_module(self, s):
                    return orig_loader.create_module(s) if hasattr(orig_loader, "create_module") else None

                def exec_module(self, module):
                    orig_loader.exec_module(module)
                    _patch_ultralytics_text_encoder_pin(module)

            spec.loader = _PatchedLoader()
            return spec

    if not any(getattr(f, "__name__", "") == "_UltralyticsPinFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _UltralyticsPinFinder)


_install_ultralytics_import_hook()


def _attempt_download_asset_passthrough(file, *args, **kwargs):
    """Fallback when ultralytics is unavailable: just return the file path.

    YOLOE weights are loaded by OpenVocabDetector.detect() which already
    uses an absolute ${repo}/models/... path, so the text encoder CWD
    fallback only matters in interactive workflows where ultralytics is
    present. If ultralytics isn't installed, the calling code path
    will fail with its own FileNotFoundError on the encoded weights.
    """
    return file


def _safe_attempt_download_asset(file, *args, **kwargs):
    """Public entrypoint: try to patch ultralytics, else passthrough.

    Callers (e.g. OpenVocabDetector) wrap their detector init in a guard
    that catches the ImportError from the actual `import ultralytics`,
    so this function only runs once the caller has already verified
    ultralytics is loadable. We still defensively try/except here.
    """
    if _patch_ultralytics_text_encoder_pin():
        try:
            from ultralytics.utils import downloads as _dl
            return _dl.attempt_download_asset(file, *args, **kwargs)
        except Exception:
            return _orig_attempt_download_asset(file, *args, **kwargs) if _orig_attempt_download_asset else file
    return _attempt_download_asset_passthrough(file, *args, **kwargs)


del Path, sys

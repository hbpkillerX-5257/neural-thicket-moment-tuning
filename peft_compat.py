"""Work around Kaggle PEFT + torchao version conflicts.

Kaggle currently ships torchao 0.10 while recent peft requires >=0.16. When the
old package is importable, peft raises inside the LoRA dispatcher instead of
falling back to plain nn.Linear adapters. For our FP16 LoRA runs we only need
the standard Linear path, so an incompatible torchao can be ignored safely.

Important: peft.tuners.lora.torchao binds ``is_torchao_available`` into its own
module globals at import time, so patching only peft.import_utils is not enough.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, Version


def _torchao_version() -> str | None:
    try:
        import torchao
    except ImportError:
        return None
    return getattr(torchao, "__version__", "0")


def _force_torchao_unavailable() -> None:
    def _unavailable() -> bool:
        return False

    import peft.import_utils as peft_import_utils

    peft_import_utils.is_torchao_available = _unavailable  # type: ignore[assignment]

    # Patch every already-imported peft module that bound the helper by name.
    import sys

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("peft"):
            continue
        if hasattr(module, "is_torchao_available"):
            setattr(module, "is_torchao_available", _unavailable)


def disable_incompatible_torchao(min_version: str = "0.16.0") -> bool:
    """Return True if torchao dispatch was disabled for this process."""
    installed = _torchao_version()
    if installed is None:
        return False

    try:
        too_old = Version(installed) < Version(min_version)
    except InvalidVersion:
        too_old = True
    if not too_old:
        return False

    _force_torchao_unavailable()
    return True

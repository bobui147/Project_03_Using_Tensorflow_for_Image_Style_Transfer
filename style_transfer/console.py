"""Cross-platform console helpers."""

from __future__ import annotations

import sys
from typing import TextIO


def _reconfigure(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def configure_utf8_console() -> None:
    """Ensure Vietnamese CLI messages work in legacy Windows terminals."""

    _reconfigure(sys.stdout)
    _reconfigure(sys.stderr)


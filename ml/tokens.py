"""Read tokens without ever holding them in a notebook cell or config. Fails closed."""

from __future__ import annotations

import os
from typing import Callable, Optional


class MissingSecretError(RuntimeError):
    """A required token is absent or empty; the run must stop."""


def _colab_getter() -> Optional[Callable[[str], Optional[str]]]:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None

    def get(name: str) -> Optional[str]:
        try:
            return userdata.get(name)
        except Exception:  # Colab raises if the secret is missing or access isn't granted
            return None

    return get


def require_secret(name: str, getter: Optional[Callable[[str], Optional[str]]] = None) -> str:
    """Return the secret's value, or raise MissingSecretError. Never returns an empty string.

    Order: an explicit getter (tests), then Colab Secrets, then the environment.
    """
    if not name or not name.strip():
        raise MissingSecretError("secret name is empty")
    sources = [getter] if getter is not None else [g for g in (_colab_getter(), os.environ.get) if g]
    for source in sources:
        value = source(name)
        if isinstance(value, str) and value.strip():
            return value
    raise MissingSecretError(f"required secret {name!r} is missing or empty; refusing to continue")

"""Gate ML3c helpers for the full fine-tune. Pure functions, tested on CPU; the VM driver is scripts/ml3c_full_vm.py."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

REQUIRED = ("max_units", "rate_units_per_hour", "staging_config", "nbac_base_url", "nbac_checksums", "experiment_name",
            "tracking_uri", "start_from", "normalisation", "max_epochs", "early_stopping_patience", "lr", "batch_size",
            "precision", "seed")


class FullRunError(ValueError):
    """The full-run settings or inputs are not safe to start with."""


def check_full_config(full: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED if k not in full]
    if missing:
        raise FullRunError(f"full run config is missing {missing}")
    for key in ("max_units", "rate_units_per_hour", "lr"):
        v = full[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise FullRunError(f"full.{key} must be a positive number")
    for key in ("max_epochs", "early_stopping_patience", "batch_size", "seed"):
        v = full[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "seed" else 1):
            raise FullRunError(f"full.{key} must be a {'non-negative' if key == 'seed' else 'positive'} integer")
    if full["start_from"] not in ("burnscars_checkpoint", "backbone"):
        raise FullRunError("full.start_from must be burnscars_checkpoint or backbone")
    if full["normalisation"] not in ("training_chips", "base_model"):
        raise FullRunError("full.normalisation must be training_chips or base_model")
    if not str(full["tracking_uri"]).startswith("https://"):
        raise FullRunError("full.tracking_uri must be https")
    return dict(full)


def max_train_hours(full: Mapping[str, Any], reserve_units: float) -> float:
    """Hours of GPU time the training may use: (cap - units already used or reserved) / rate. Fails closed at <= 0."""
    hours = (float(full["max_units"]) - float(reserve_units)) / float(full["rate_units_per_hour"])
    if hours <= 0:
        raise FullRunError(f"no budget left: cap {full['max_units']} units, reserved {reserve_units}")
    return hours


def lightning_max_time(hours: float) -> str:
    """Lightning Trainer max_time as 'DD:HH:MM:SS', rounded down to whole minutes."""
    minutes = int(hours * 60)
    if minutes <= 0:
        raise FullRunError("max_time under one minute")
    days, rem = divmod(minutes, 24 * 60)
    h, m = divmod(rem, 60)
    return f"{days:02d}:{h:02d}:{m:02d}:00"


def read_checksums(path: str | Path) -> dict[str, str]:
    """'<sha256>  <file name>' lines. Fails closed on malformed lines or duplicate names."""
    out: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            raise FullRunError(f"{path}:{n}: expected '<sha256>  <name>'")
        if "/" in parts[1] or parts[1] in out:
            raise FullRunError(f"{path}:{n}: bad or duplicate file name {parts[1]!r}")
        out[parts[1]] = parts[0]
    if not out:
        raise FullRunError(f"{path}: no checksums")
    return out


def verify_files(folder: str | Path, expected: Mapping[str, str]) -> list[str]:
    """Every expected file present with its SHA-256. Returns the verified names; fails closed otherwise."""
    folder = Path(folder)
    problems = []
    for name, sha in sorted(expected.items()):
        path = folder / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha:
            problems.append(f"{name}: SHA-256 mismatch")
    if problems:
        raise FullRunError("NBAC files do not match the pinned checksums:\n" + "\n".join(problems))
    return sorted(expected)

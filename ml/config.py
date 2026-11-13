"""Load and check a training config.

A config names everything a run depends on. Loading fails closed: a missing key, an unpinned base model or a
bad seed raises ConfigError before any data or GPU is touched.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REQUIRED_KEYS = ("name", "task", "seed", "base_model", "data_manifest")
# Only a full commit hash is immutable. Branch names, refs and tags can move, so they are refused.
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
MANIFEST_DIR = "data_manifest"


class ConfigError(ValueError):
    """The config is missing something a reproducible run needs."""


def check_manifest_path(value: str) -> PurePosixPath:
    """A config's data_manifest must be a repo-relative JSON file directly under data_manifest/."""
    if "\\" in value:
        raise ConfigError(f"'data_manifest' must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if (
        str(pure) != value  # only the plain spelling, e.g. no './', '//' or trailing '/'
        or "\x00" in value
        or pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) != 2
        or pure.parts[0] != MANIFEST_DIR
        or pure.suffix != ".json"
    ):
        raise ConfigError(f"'data_manifest' must look like '{MANIFEST_DIR}/<name>.json', not {value!r}")
    return pure


def config_sha256(path: str | Path) -> str:
    """SHA-256 of the config file's bytes, as logged with each run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and check the keys every run needs."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")

    for key in ("name", "task", "data_manifest"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ConfigError(f"{path}: '{key}' must be a non-empty string")
    check_manifest_path(data["data_manifest"])

    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"{path}: 'seed' must be a non-negative integer")

    base = data["base_model"]
    if not isinstance(base, dict):
        raise ConfigError(f"{path}: 'base_model' must be a mapping with 'id' and 'revision'")
    model_id = base.get("id")
    revision = base.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError(f"{path}: 'base_model.id' must be a non-empty string")
    if not isinstance(revision, str) or not COMMIT_SHA.fullmatch(revision):
        raise ConfigError(
            f"{path}: 'base_model.revision' must be a full 40-character lowercase commit hash, not {revision!r}"
        )

    return data

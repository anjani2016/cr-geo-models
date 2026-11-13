"""The record every training or evaluation run logs (to MLflow and the gate record).

A run must name its code, data and model versions. Building the record fails closed if the code isn't committed,
so no published result can come from uncommitted code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ml.config import MANIFEST_DIR, check_manifest_path, config_sha256, load_config
from ml.manifest import load_manifest, manifest_sha256


class RunMetadataError(RuntimeError):
    """The run can't be tied to a committed state of the code."""


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RunMetadataError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def commit_sha(repo_dir: str | Path, allow_dirty: bool = False) -> str:
    """HEAD commit SHA. Refuses when tracked files have uncommitted changes, unless allow_dirty (exploration only)."""
    repo_dir = Path(repo_dir)
    sha = _git(repo_dir, "rev-parse", "HEAD")
    if not allow_dirty and _git(repo_dir, "status", "--porcelain", "--untracked-files=no"):
        raise RunMetadataError("tracked files have uncommitted changes; commit first or pass allow_dirty for exploration")
    return sha


def build_run_metadata(
    repo_dir: str | Path,
    config_path: str | Path,
    gpu_type: str,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Collect the reproducibility record for one run. The manifest path comes from the config."""
    if not isinstance(gpu_type, str) or not gpu_type.strip():
        raise RunMetadataError("gpu_type must be named (e.g. 'L4', 'A100', 'cpu')")
    repo_dir = Path(repo_dir)
    config = load_config(config_path)
    rel = check_manifest_path(config["data_manifest"])
    manifest_dir = (repo_dir / MANIFEST_DIR).resolve()
    manifest_path = (repo_dir / Path(*rel.parts)).resolve()
    if manifest_path.parent != manifest_dir:
        raise RunMetadataError(f"data manifest resolves outside {MANIFEST_DIR}/: {config['data_manifest']}")
    manifest = load_manifest(manifest_path)
    return {
        "commit_sha": commit_sha(repo_dir, allow_dirty=allow_dirty),
        "dirty": bool(allow_dirty),
        "config_name": config["name"],
        "config_sha256": config_sha256(config_path),
        "task": config["task"],
        "seed": config["seed"],
        "base_model_id": config["base_model"]["id"],
        "base_model_revision": config["base_model"]["revision"],
        "dataset": manifest["dataset"],
        "data_manifest_sha256": manifest_sha256(manifest),
        "gpu_type": gpu_type.strip(),
    }

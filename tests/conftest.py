import json
import subprocess
from pathlib import Path

import pytest

from ml.manifest import build_entries


def write_config(path: Path, **overrides) -> Path:
    config = {
        "name": "example",
        "task": "segmentation",
        "seed": 42,
        "base_model": {"id": "Example-Org/example-model", "revision": "0123456789abcdef0123456789abcdef01234567"},
        "data_manifest": "data_manifest/example.json",
    }
    config.update(overrides)
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "staged"
    (root / "chips").mkdir(parents=True)
    (root / "chips" / "a.bin").write_bytes(b"alpha")
    (root / "chips" / "b.bin").write_bytes(b"bravo")
    return root


@pytest.fixture
def manifest(data_root: Path) -> dict:
    return {
        "dataset": "example-dataset",
        "source_url": "https://example.org/data",
        "licence": "CC-BY-4.0",
        "attribution": "Example Org",
        "retrieved_on": "2026-11-12",
        "files": build_entries(data_root, ["chips/a.bin", "chips/b.bin"]),
    }


@pytest.fixture
def git_repo(tmp_path: Path, manifest: dict) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data_manifest").mkdir()
    (repo / "data_manifest" / "example.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_config(repo / "config.yaml")
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.org"],
        ["config", "user.name", "Test"],
        ["add", "."],
        ["commit", "-q", "-m", "init"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return repo

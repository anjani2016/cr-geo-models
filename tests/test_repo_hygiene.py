"""Checks on what gets committed to this public repo."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),  # Hugging Face token shape
    re.compile(r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API key shape
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_have_no_outputs(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], f"{path.name}: clear outputs before committing"
            assert cell.get("execution_count") is None


def test_notebook_list_is_not_empty():
    assert NOTEBOOKS, "expected at least the Colab launcher"


def _tracked_or_candidate_files():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return [ROOT / line for line in result.stdout.splitlines() if line]


def test_no_token_shaped_strings_in_committable_files():
    hits = []
    for path in _tracked_or_candidate_files():
        if not path.is_file() or path.suffix in {".png", ".jpg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in TOKEN_PATTERNS:
            if pattern.search(text) and path.name != "test_repo_hygiene.py":
                hits.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    assert not hits, "token-shaped strings found:\n" + "\n".join(hits)


@pytest.mark.parametrize("name", ["model.ckpt", "data/x.tif", "mlruns/0/meta.yaml", ".env", ".env.local", "svc-service-account.json"])
def test_gitignore_blocks_data_and_secrets(name):
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index", name], check=False
    )
    if result.returncode == 128:
        pytest.skip("not a git checkout")
    assert result.returncode == 0, f"{name} is not ignored"

import subprocess

import pytest

from ml.manifest import manifest_sha256
from ml.run_metadata import RunMetadataError, build_run_metadata, commit_sha


def test_record_names_code_data_and_model(git_repo, manifest):
    record = build_run_metadata(git_repo, git_repo / "config.yaml", gpu_type="L4")
    head = subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert record["commit_sha"] == head
    assert record["dirty"] is False
    assert record["base_model_revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert record["data_manifest_sha256"] == manifest_sha256(manifest)
    assert record["seed"] == 42 and record["gpu_type"] == "L4"


def test_uncommitted_change_refuses(git_repo):
    (git_repo / "config.yaml").write_text((git_repo / "config.yaml").read_text() + "\n# edit\n")
    with pytest.raises(RunMetadataError, match="uncommitted"):
        build_run_metadata(git_repo, git_repo / "config.yaml", gpu_type="L4")


def test_allow_dirty_marks_record(git_repo):
    (git_repo / "config.yaml").write_text((git_repo / "config.yaml").read_text() + "\n# edit\n")
    record = build_run_metadata(git_repo, git_repo / "config.yaml", gpu_type="cpu", allow_dirty=True)
    assert record["dirty"] is True


def test_untracked_files_do_not_block(git_repo):
    (git_repo / "scratch.txt").write_text("x")
    assert commit_sha(git_repo)


def test_not_a_git_repo_fails(tmp_path):
    with pytest.raises(RunMetadataError):
        commit_sha(tmp_path)


@pytest.mark.parametrize("gpu", ["", "  ", None])
def test_gpu_type_required(git_repo, gpu):
    with pytest.raises(RunMetadataError, match="gpu_type"):
        build_run_metadata(git_repo, git_repo / "config.yaml", gpu_type=gpu)


def test_manifest_symlink_out_of_data_manifest_is_refused(git_repo, tmp_path, manifest):
    import json
    import os

    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps(manifest))
    link = git_repo / "data_manifest" / "linked.json"
    os.symlink(outside, link)
    from tests.conftest import write_config

    write_config(git_repo / "config.yaml", data_manifest="data_manifest/linked.json")
    with pytest.raises(RunMetadataError, match="outside"):
        build_run_metadata(git_repo, git_repo / "config.yaml", gpu_type="L4", allow_dirty=True)

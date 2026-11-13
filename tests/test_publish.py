import os

import pytest

from ml.manifest import file_sha256
from ml.publish import StageItem, StagingError, stage_upload


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    (d / "README.md").write_text("# card\n")
    (d / "w.safetensors").write_bytes(b"\x00\x01weights")
    (d / "c.yaml").write_text("a: 1\n")
    return d


def _items(src):
    return [StageItem("README.md", src / "README.md"),
            StageItem("model.safetensors", src / "w.safetensors", file_sha256(src / "w.safetensors")),
            StageItem("config/c.yaml", src / "c.yaml")]


def test_stages_allow_list_and_sums(src, tmp_path):
    out = tmp_path / "out"
    sums = stage_upload(_items(src), out)
    names = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert names == ["README.md", "SHA256SUMS", "config/c.yaml", "model.safetensors"]
    assert sums["model.safetensors"] == file_sha256(src / "w.safetensors")
    assert (out / "SHA256SUMS").read_text().count("\n") == 3


def test_hash_mismatch_fails_closed(src, tmp_path):
    items = _items(src)
    (src / "w.safetensors").write_bytes(b"\x00\x02weights")  # one byte flipped after the hash was taken
    with pytest.raises(StagingError, match="mismatch"):
        stage_upload(items, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_binary_without_hash_refused(src, tmp_path):
    with pytest.raises(StagingError, match="needs an expected"):
        stage_upload([StageItem("model.safetensors", src / "w.safetensors")], tmp_path / "out")


@pytest.mark.parametrize("dest", ["../x.md", "/abs.md", ".env", "a/.env", "a/b/c.md", "", "a\\b.md", "SHA256SUMS"])
def test_bad_destinations_refused(src, tmp_path, dest):
    with pytest.raises(StagingError):
        stage_upload([StageItem(dest, src / "README.md")], tmp_path / "out")


def test_symlink_source_refused(src, tmp_path):
    link = src / "link.md"
    os.symlink(src / "README.md", link)
    with pytest.raises(StagingError, match="symlink"):
        stage_upload([StageItem("README.md", link)], tmp_path / "out")


def test_missing_source_refused(src, tmp_path):
    with pytest.raises(StagingError, match="missing"):
        stage_upload([StageItem("norm_stats.json", src / "nope.json")], tmp_path / "out")


def test_non_empty_out_dir_refused(src, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / ".DS_Store").write_bytes(b"x")
    with pytest.raises(StagingError, match="not empty"):
        stage_upload(_items(src), out)


def test_token_shaped_string_refused(src, tmp_path):
    (src / "README.md").write_text("token " + "hf_" + "a" * 34 + "\n")
    with pytest.raises(StagingError, match="token-shaped"):
        stage_upload(_items(src), tmp_path / "out")


def test_duplicate_destination_refused(src, tmp_path):
    with pytest.raises(StagingError, match="duplicate"):
        stage_upload([StageItem("README.md", src / "README.md"), StageItem("README.md", src / "c.yaml")], tmp_path / "out")


def test_bad_hash_format_refused(src, tmp_path):
    with pytest.raises(StagingError, match="64 lowercase"):
        stage_upload([StageItem("m.safetensors", src / "w.safetensors", "B471")], tmp_path / "out")

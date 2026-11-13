import json
import os

import pytest

from ml.manifest import ManifestError, build_entries, load_manifest, manifest_sha256, verify


def test_verify_passes_on_matching_files(manifest, data_root):
    verify(manifest, data_root)


def test_verify_fails_on_tampered_file(manifest, data_root):
    (data_root / "chips" / "a.bin").write_bytes(b"alphx")  # same size, different bytes
    with pytest.raises(ManifestError, match="SHA-256"):
        verify(manifest, data_root)


def test_verify_fails_on_size_change_and_missing_file(manifest, data_root):
    (data_root / "chips" / "a.bin").write_bytes(b"alpha-longer")
    (data_root / "chips" / "b.bin").unlink()
    with pytest.raises(ManifestError) as err:
        verify(manifest, data_root)
    assert "size" in str(err.value) and "missing" in str(err.value)


@pytest.mark.parametrize("bad", ["../outside.bin", "/etc/passwd", "chips/../../x", "", "chips\\a.bin", "chips/a\x00.bin"])
def test_paths_escaping_root_are_rejected(manifest, data_root, bad):
    manifest["files"][0]["path"] = bad
    with pytest.raises(ManifestError):
        verify(manifest, data_root)


def test_symlink_pointing_outside_root_is_rejected(manifest, data_root, tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"alpha")
    target = data_root / "chips" / "a.bin"
    target.unlink()
    os.symlink(outside, target)
    with pytest.raises(ManifestError, match="outside"):
        verify(manifest, data_root)


def test_build_entries_rejects_non_files(data_root):
    with pytest.raises(ManifestError):
        build_entries(data_root, ["chips"])
    with pytest.raises(ManifestError):
        build_entries(data_root, ["../x"])


def _write(tmp_path, manifest):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("field", ["dataset", "source_url", "licence", "attribution", "retrieved_on", "files"])
def test_load_manifest_requires_fields(tmp_path, manifest, field):
    del manifest[field]
    with pytest.raises(ManifestError, match=field):
        load_manifest(_write(tmp_path, manifest))


def test_load_manifest_rejects_blank_licence_empty_files_and_duplicates(tmp_path, manifest):
    blank = dict(manifest, licence="  ")
    with pytest.raises(ManifestError, match="licence"):
        load_manifest(_write(tmp_path, blank))
    with pytest.raises(ManifestError, match="files"):
        load_manifest(_write(tmp_path, dict(manifest, files=[])))
    dup = dict(manifest, files=[manifest["files"][0], manifest["files"][0]])
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, dup))
    extra = dict(manifest, files=[dict(manifest["files"][0], note="x")])
    with pytest.raises(ManifestError, match="exactly"):
        load_manifest(_write(tmp_path, extra))


def test_manifest_checksum_is_order_independent_and_content_sensitive(manifest):
    reordered = dict(reversed(list(manifest.items())))
    assert manifest_sha256(manifest) == manifest_sha256(reordered)
    assert manifest_sha256(manifest) != manifest_sha256(dict(manifest, licence="CC0-1.0"))


def test_build_entries_refuses_symlink_outside_root(data_root, tmp_path):
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"outside")
    os.symlink(outside, data_root / "chips" / "link.bin")
    with pytest.raises(ManifestError, match="outside"):
        build_entries(data_root, ["chips/link.bin"])


@pytest.mark.parametrize("size", [3.0, 5.0, True, False, -1, "5", None])
def test_bad_byte_counts_fail(tmp_path, manifest, data_root, size):
    manifest["files"][0]["bytes"] = size
    with pytest.raises(ManifestError, match="bytes"):
        load_manifest(_write(tmp_path, manifest))
    with pytest.raises(ManifestError, match="bytes"):
        verify(manifest, data_root)


def test_true_size_cannot_pass_for_a_one_byte_file(tmp_path, manifest, data_root):
    (data_root / "chips" / "one.bin").write_bytes(b"x")
    entry = build_entries(data_root, ["chips/one.bin"])[0]
    manifest["files"] = [dict(entry, bytes=True)]
    with pytest.raises(ManifestError, match="bytes"):
        verify(manifest, data_root)


@pytest.mark.parametrize("digest", ["abc", "A" * 64, "g" * 64, "0" * 63, "0" * 65, 123, None])
def test_bad_sha256_fails(tmp_path, manifest, digest):
    manifest["files"][0]["sha256"] = digest
    with pytest.raises(ManifestError, match="sha256"):
        load_manifest(_write(tmp_path, manifest))

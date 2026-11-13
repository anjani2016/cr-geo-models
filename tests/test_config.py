import pytest

from ml.config import ConfigError, config_sha256, load_config
from tests.conftest import write_config


def test_valid_config_loads(tmp_path):
    config = load_config(write_config(tmp_path / "c.yaml"))
    assert config["seed"] == 42
    assert config["base_model"]["revision"] == "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize("key", ["name", "task", "seed", "base_model", "data_manifest"])
def test_missing_required_key_fails(tmp_path, key):
    path = write_config(tmp_path / "c.yaml")
    import yaml

    data = yaml.safe_load(path.read_text())
    del data[key]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=key):
        load_config(path)


@pytest.mark.parametrize("revision", ["main", "MAIN", "latest", "", "  ", None, 123])
def test_unpinned_base_model_revision_fails(tmp_path, revision):
    path = write_config(tmp_path / "c.yaml", base_model={"id": "Example-Org/m", "revision": revision})
    with pytest.raises(ConfigError, match="revision"):
        load_config(path)


@pytest.mark.parametrize("seed", [-1, 1.5, "42", True, None])
def test_bad_seed_fails(tmp_path, seed):
    with pytest.raises(ConfigError, match="seed"):
        load_config(write_config(tmp_path / "c.yaml", seed=seed))


@pytest.mark.parametrize("value", ["", "   ", 7])
def test_empty_or_non_string_name_fails(tmp_path, value):
    with pytest.raises(ConfigError, match="name"):
        load_config(write_config(tmp_path / "c.yaml", name=value))


def test_non_mapping_and_bad_yaml_fail(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)
    path.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_yaml_tags_are_not_executed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("!!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_checksum_changes_with_content(tmp_path):
    a = write_config(tmp_path / "a.yaml")
    b = write_config(tmp_path / "b.yaml", seed=43)
    assert config_sha256(a) != config_sha256(b)
    assert len(config_sha256(a)) == 64


@pytest.mark.parametrize(
    "revision",
    [
        "refs/heads/main",
        "origin/main",
        "refs/remotes/origin/main",
        "v1.0",
        "0123456789abcdef",  # short hash
        "0123456789ABCDEF0123456789ABCDEF01234567",  # uppercase
        "0123456789abcdef0123456789abcdef012345678",  # 41 chars
        " 0123456789abcdef0123456789abcdef01234567",
    ],
)
def test_branch_ref_tag_and_short_revisions_fail(tmp_path, revision):
    path = write_config(tmp_path / "c.yaml", base_model={"id": "Example-Org/m", "revision": revision})
    with pytest.raises(ConfigError, match="revision"):
        load_config(path)


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/x.json",
        "../x.json",
        "data_manifest/../x.json",
        "data_manifest/sub/x.json",
        "other/x.json",
        "data_manifest/x.yaml",
        "data_manifest\\x.json",
        "x.json",
        "./data_manifest/x.json",
        "data_manifest//x.json",
        "data_manifest/x.json/",
        "data_manifest/x\x00.json",
    ],
)
def test_manifest_path_outside_data_manifest_fails(tmp_path, value):
    with pytest.raises(ConfigError, match="data_manifest"):
        load_config(write_config(tmp_path / "c.yaml", data_manifest=value))

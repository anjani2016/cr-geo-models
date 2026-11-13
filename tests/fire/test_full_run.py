import hashlib
from pathlib import Path

import pytest
import yaml

from ml.fire.full_run import FullRunError, check_full_config, lightning_max_time, max_train_hours, read_checksums, verify_files

ROOT = Path(__file__).resolve().parents[2]
FULL = yaml.safe_load((ROOT / "configs/ml3_burn_scar.yaml").read_text())["full"]


def test_shipped_full_config_is_valid_and_capped():
    cfg = check_full_config(FULL)
    assert cfg["max_units"] == 40 and cfg["start_from"] == "burnscars_checkpoint" and cfg["tracking_uri"].startswith("https://")


@pytest.mark.parametrize("key,value", [("max_units", 0), ("max_units", True), ("lr", -1), ("max_epochs", 0), ("max_epochs", 2.5),
                                       ("seed", -1), ("start_from", "scratch"), ("normalisation", "imagenet"),
                                       ("tracking_uri", "http://mlflow.example.org")])
def test_bad_full_config_fails(key, value):
    with pytest.raises(FullRunError):
        check_full_config(dict(FULL, **{key: value}))


def test_missing_key_fails():
    bad = dict(FULL)
    del bad["nbac_checksums"]
    with pytest.raises(FullRunError, match="nbac_checksums"):
        check_full_config(bad)


def test_time_cap_from_units():
    assert max_train_hours({"max_units": 40, "rate_units_per_hour": 2.0}, reserve_units=4) == pytest.approx(18.0)
    with pytest.raises(FullRunError, match="no budget"):
        max_train_hours({"max_units": 10, "rate_units_per_hour": 2.0}, reserve_units=10)


@pytest.mark.parametrize("hours,expected", [(18.0, "00:18:00:00"), (25.5, "01:01:30:00"), (0.5, "00:00:30:00")])
def test_lightning_max_time(hours, expected):
    assert lightning_max_time(hours) == expected


def test_lightning_max_time_refuses_under_a_minute():
    with pytest.raises(FullRunError):
        lightning_max_time(0.01)


def test_shipped_nbac_checksums_parse():
    sums = read_checksums(ROOT / FULL["nbac_checksums"])
    assert len(sums) == 12 and all(n.startswith("NBAC_") and n.endswith("_20260513.zip") for n in sums)


@pytest.mark.parametrize("line", ["abc  x.zip", "0" * 64 + "  ../x.zip", "0" * 64, ("A" * 64) + "  x.zip"])
def test_bad_checksum_lines_fail(tmp_path, line):
    p = tmp_path / "s.txt"
    p.write_text(line + "\n")
    with pytest.raises(FullRunError):
        read_checksums(p)


def test_verify_files_ok_missing_and_tampered(tmp_path):
    (tmp_path / "a.zip").write_bytes(b"hello")
    (tmp_path / "b.zip").write_bytes(b"world")
    expected = {"a.zip": hashlib.sha256(b"hello").hexdigest(), "b.zip": hashlib.sha256(b"world").hexdigest()}
    assert verify_files(tmp_path, expected) == ["a.zip", "b.zip"]
    (tmp_path / "b.zip").write_bytes(b"w0rld")
    with pytest.raises(FullRunError, match="b.zip: SHA-256 mismatch"):
        verify_files(tmp_path, expected)
    with pytest.raises(FullRunError, match="c.zip: missing"):
        verify_files(tmp_path, dict(expected, **{"c.zip": "0" * 64}))

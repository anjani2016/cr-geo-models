from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from ml.fire.train_prep import TrainPrepError, band_stats, chip_pairs, make_subset, terratorch_config

ROOT = Path(__file__).resolve().parents[2]
BASE = yaml.safe_load((ROOT / "configs/vendor/prithvi_eo_v2_300m_burnscars_config.yaml").read_text())
TRAINING = {"max_epochs": 1, "batch_size": 4, "experiment_name": "e", "tracking_uri": "file:/tmp/mlruns", "output_dir": "/tmp/out"}


def _chip(folder, stem, value, label_value=0, ignore_rows=0):
    folder.mkdir(parents=True, exist_ok=True)
    t = from_origin(0, 64, 1, 1)
    img = np.stack([np.full((8, 8), value * (b + 1), dtype="float32") for b in range(6)])
    lab = np.full((8, 8), label_value, dtype="int16")
    lab[:ignore_rows] = -1
    if ignore_rows:
        img[:, :ignore_rows] = 99.0  # must not reach the statistics
    for suffix, arr, dtype in (("img", img, "float32"), ("label", lab[None], "int16")):
        with rasterio.open(folder / f"{stem}_{suffix}.tif", "w", driver="GTiff", height=8, width=8, count=arr.shape[0],
                           dtype=dtype, transform=t, crs="EPSG:32616") as d:
            d.write(arr)


def test_band_stats_use_valid_pixels_only(tmp_path):
    _chip(tmp_path / "train", "2021-1_00", 0.1, ignore_rows=4)
    _chip(tmp_path / "train", "2021-2_00", 0.3)
    stats = band_stats(tmp_path / "train")
    # valid pixels: 32 at 0.1*(b+1), 64 at 0.3*(b+1); the 99.0 ignore rows are excluded
    expected = [(32 * 0.1 + 64 * 0.3) * (b + 1) / 96 for b in range(6)]
    assert np.allclose(stats["means"], expected, rtol=1e-6)
    assert stats["pixels"] == 96 and stats["chips"] == 2 and all(s > 0 for s in stats["stds"])


def test_band_stats_failures(tmp_path):
    with pytest.raises(TrainPrepError, match="no training chips"):
        band_stats(tmp_path)
    _chip(tmp_path / "flat", "2021-1_00", 0.1)
    with pytest.raises(TrainPrepError, match="zero spread"):
        band_stats(tmp_path / "flat")
    _chip(tmp_path / "allignore", "2021-1_00", 0.1, ignore_rows=8)
    with pytest.raises(TrainPrepError, match="no valid pixels"):
        band_stats(tmp_path / "allignore")


def test_unmatched_chip_fails(tmp_path):
    _chip(tmp_path / "train", "2021-1_00", 0.1)
    (tmp_path / "train" / "2021-1_00_label.tif").unlink()
    with pytest.raises(TrainPrepError, match="unmatched"):
        chip_pairs(tmp_path / "train")


def test_subset_takes_whole_fires_and_is_deterministic(tmp_path):
    src = tmp_path / "src"
    for split in ("train", "val", "test"):
        for fire in range(5):
            for n in range(3):
                _chip(src / split, f"2021-{fire}_{n:02d}", 0.1 + fire * 0.01 + n * 0.001)
    a = make_subset(src, tmp_path / "a", {"train": 4, "test": 2})
    b = make_subset(src, tmp_path / "b", {"train": 4, "test": 2})
    assert a == b
    assert len(a["train"]) == 6 and len(a["test"]) == 3  # whole fires of 3 chips
    fires = {s.rsplit("_", 1)[0] for s in a["train"]}
    assert all(sum(s.startswith(f + "_") for s in a["train"]) == 3 for f in fires)
    assert not (tmp_path / "a" / "val").exists()


@pytest.mark.parametrize("counts", [{"train": 0}, {"train": True}, {"holdout": 3}, {"train": "4"}])
def test_subset_bad_requests(tmp_path, counts):
    _chip(tmp_path / "src" / "train", "2021-1_00", 0.1)
    with pytest.raises(TrainPrepError):
        make_subset(tmp_path / "src", tmp_path / "dst", counts)


def test_subset_refuses_non_empty_destination(tmp_path):
    _chip(tmp_path / "src" / "train", "2021-1_00", 0.1)
    (tmp_path / "dst").mkdir()
    (tmp_path / "dst" / "keep.txt").write_text("x")
    with pytest.raises(TrainPrepError, match="not empty"):
        make_subset(tmp_path / "src", tmp_path / "dst", {"train": 1})


def test_terratorch_config_rewrites_data_and_logging_without_touching_base():
    before = yaml.safe_dump(BASE)
    cfg = terratorch_config(BASE, "/content/work/subset/", [0.1] * 6, [0.05] * 6, dict(TRAINING, limit_train_batches=6, seed=7))
    init = cfg["data"]["init_args"]
    assert init["train_data_root"] == "/content/work/subset/train" and init["test_data_root"] == "/content/work/subset/test"
    assert "train_split" not in init and init["img_grep"] == "*_img.tif" and init["label_grep"] == "*_label.tif"
    assert init["means"] == [0.1] * 6 and init["batch_size"] == 4
    assert cfg["trainer"]["logger"]["class_path"].endswith("MLFlowLogger")
    assert cfg["trainer"]["limit_train_batches"] == 6 and cfg["seed_everything"] == 7
    assert any(c["class_path"] == "ModelCheckpoint" for c in cfg["trainer"]["callbacks"])
    assert cfg["model"]["init_args"]["ignore_index"] == -1
    assert yaml.safe_dump(BASE) == before  # base config not mutated


@pytest.mark.parametrize("means,stds", [([0.1] * 5, [0.1] * 6), ([0.1] * 6, [0.1] * 5 + [0.0])])
def test_terratorch_config_rejects_bad_normalisation(means, stds):
    with pytest.raises(TrainPrepError):
        terratorch_config(BASE, "/x", means, stds, TRAINING)


def test_terratorch_config_requires_training_keys():
    bad = dict(TRAINING)
    del bad["tracking_uri"]
    with pytest.raises(TrainPrepError, match="tracking_uri"):
        terratorch_config(BASE, "/x", [0.1] * 6, [0.1] * 6, bad)


def test_shipped_run_config_is_pinned_and_loads():
    from ml.config import load_config

    cfg = load_config(ROOT / "configs/ml3_burn_scar.yaml")
    assert cfg["base_model"]["revision"] == "a3f2c410e45b8ac7417976614528a872f024d831"
    assert cfg["smoke"]["max_units"] == 10


def test_macos_appledouble_files_are_ignored(tmp_path):
    _chip(tmp_path / "train", "2021-1_00", 0.1)
    _chip(tmp_path / "train", "2021-2_00", 0.3)
    (tmp_path / "train" / "._2021-1_00_img.tif").write_bytes(b"\x00\x05\x16\x07junk")
    (tmp_path / "train" / "._2021-1_00_label.tif").write_bytes(b"\x00\x05\x16\x07junk")
    assert [p[0].name for p in chip_pairs(tmp_path / "train")] == ["2021-1_00_img.tif", "2021-2_00_img.tif"]
    assert band_stats(tmp_path / "train")["chips"] == 2

"""Gate ML3 preparation: normalisation statistics, a small subset for smoke tests, and the TerraTorch config.

Statistics come from training chips only (labels != -1), so validation and test data never shape the normalisation.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ml.fire.chips import IGNORE

SPLITS = ("train", "val", "test")
IMG_SUFFIX, LABEL_SUFFIX = "_img.tif", "_label.tif"


class TrainPrepError(ValueError):
    """Inputs for training preparation are missing or inconsistent."""


def chip_pairs(split_dir: str | Path) -> list[tuple[Path, Path]]:
    """(image, label) pairs in one split folder, sorted. Fails closed on an image without its label or the reverse."""
    split_dir = Path(split_dir)
    images = {p.name[: -len(IMG_SUFFIX)]: p for p in split_dir.glob(f"*{IMG_SUFFIX}")}
    labels = {p.name[: -len(LABEL_SUFFIX)]: p for p in split_dir.glob(f"*{LABEL_SUFFIX}")}
    if images.keys() != labels.keys():
        raise TrainPrepError(f"{split_dir}: unmatched chips {sorted(images.keys() ^ labels.keys())[:5]}")
    return [(images[k], labels[k]) for k in sorted(images)]


def band_stats(train_dir: str | Path) -> dict[str, Any]:
    """Per-band mean and standard deviation over valid pixels (label != -1) of every training chip (float64 sums)."""
    import rasterio

    pairs = chip_pairs(train_dir)
    if not pairs:
        raise TrainPrepError(f"{train_dir}: no training chips")
    total = sq = None
    count = 0
    for img_path, lab_path in pairs:
        with rasterio.open(img_path) as img, rasterio.open(lab_path) as lab:
            bands = img.read().astype(np.float64)
            valid = lab.read(1) != IGNORE
        if bands.shape[1:] != valid.shape:
            raise TrainPrepError(f"{img_path.name}: image and label sizes differ")
        values = bands[:, valid]
        if total is None:
            total, sq = np.zeros(bands.shape[0]), np.zeros(bands.shape[0])
        elif values.shape[0] != total.shape[0]:
            raise TrainPrepError(f"{img_path.name}: band count {values.shape[0]} differs from {total.shape[0]}")
        total += values.sum(axis=1)
        sq += (values**2).sum(axis=1)
        count += values.shape[1]
    if count == 0:
        raise TrainPrepError("no valid pixels in the training chips")
    mean = total / count
    std = np.sqrt(np.maximum(sq / count - mean**2, 0.0))
    if (std <= 0).any():
        raise TrainPrepError(f"a band has zero spread: std={std.tolist()}")
    return {"means": mean.tolist(), "stds": std.tolist(), "pixels": int(count), "chips": len(pairs)}


def _fire_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]  # '<year>-<id>_<nn>' -> '<year>-<id>'


def make_subset(src_root: str | Path, dst_root: str | Path, counts: Mapping[str, int]) -> dict[str, list[str]]:
    """Copy about counts[split] chips per split, taking whole fires in a fixed hash order (so no fire is split up).

    Deterministic: the same staged data gives the same subset. Refuses to write into a non-empty destination.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    if dst_root.exists() and any(dst_root.iterdir()):
        raise TrainPrepError(f"{dst_root} is not empty")
    chosen: dict[str, list[str]] = {}
    for split, want in counts.items():
        if split not in SPLITS or isinstance(want, bool) or not isinstance(want, int) or want <= 0:
            raise TrainPrepError(f"bad subset request {split!r}: {want!r}")
        pairs = chip_pairs(src_root / split)
        by_fire: dict[str, list[tuple[Path, Path]]] = {}
        for img, lab in pairs:
            by_fire.setdefault(_fire_of(img.name[: -len(IMG_SUFFIX)]), []).append((img, lab))
        order = sorted(by_fire, key=lambda f: hashlib.sha256(f.encode()).hexdigest())
        picked: list[tuple[Path, Path]] = []
        for fire in order:
            if len(picked) >= want:
                break
            picked.extend(by_fire[fire])
        if not picked:
            raise TrainPrepError(f"{src_root / split}: no chips to pick")
        (dst_root / split).mkdir(parents=True, exist_ok=True)
        for img, lab in picked:
            shutil.copy2(img, dst_root / split / img.name)
            shutil.copy2(lab, dst_root / split / lab.name)
        chosen[split] = [img.name[: -len(IMG_SUFFIX)] for img, _ in picked]
    return chosen


def terratorch_config(
    base: Mapping[str, Any],
    data_root: str,
    means: list[float],
    stds: list[float],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """A TerraTorch config from the vendored base: our data folders, normalisation and training limits, MLflow logging."""
    if len(means) != 6 or len(stds) != 6 or any(s <= 0 for s in stds):
        raise TrainPrepError("need six means and six positive stds in the model's band order")
    for key in ("max_epochs", "batch_size", "experiment_name", "tracking_uri", "output_dir"):
        if key not in training:
            raise TrainPrepError(f"training.{key} is required")
    cfg = copy.deepcopy(dict(base))
    init = cfg["data"]["init_args"]
    for split in SPLITS:
        init[f"{split}_data_root"] = f"{data_root.rstrip('/')}/{split}"
        init.pop(f"{split}_split", None)  # our folders are already split by fire and region
    init.update(img_grep=f"*{IMG_SUFFIX}", label_grep=f"*{LABEL_SUFFIX}", means=list(means), stds=list(stds),
                batch_size=int(training["batch_size"]), num_workers=int(training.get("num_workers", 2)))
    trainer = cfg["trainer"]
    trainer["max_epochs"] = int(training["max_epochs"])
    trainer["default_root_dir"] = training["output_dir"]
    for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches"):
        if key in training:
            trainer[key] = training[key]
    if "precision" in training:
        trainer["precision"] = training["precision"]
    trainer["logger"] = {
        "class_path": "lightning.pytorch.loggers.MLFlowLogger",
        "init_args": {"experiment_name": training["experiment_name"], "tracking_uri": training["tracking_uri"],
                      "run_name": training.get("run_name")},
    }
    trainer["callbacks"] = [c for c in trainer.get("callbacks", []) if c.get("class_path") != "ModelCheckpoint"] + [
        {"class_path": "ModelCheckpoint", "init_args": {"monitor": "val/loss", "save_last": True,
                                                         "dirpath": f"{training['output_dir'].rstrip('/')}/checkpoints"}}
    ]
    cfg["seed_everything"] = int(training.get("seed", cfg.get("seed_everything", 0)))
    return cfg


def write_yaml(cfg: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")
    return path

"""Run burn-scar-ca v1 on one Sentinel-2 L2A chip.

    pip install terratorch==1.2.14 rasterio safetensors
    python inference_example.py chip.tif out_mask.tif

Input: a GeoTIFF with six bands in this order: BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2 (Sentinel-2 L2A B02, B03, B04, B8A, B11, B12),
as surface reflectance on a 0-1 scale, at 30 m. Chips were 512 x 512 pixels in training.
Output: a uint8 GeoTIFF, 1 = burned, 0 = not burned. This is a model estimate (Modeled), not a measurement.
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from safetensors.torch import load_file
from terratorch.tasks import SemanticSegmentationTask

HERE = Path(__file__).resolve().parent


def build_model() -> SemanticSegmentationTask:
    base = yaml.safe_load((HERE / "config" / "prithvi_eo_v2_300m_burnscars_config.yaml").read_text())
    init = dict(base["model"]["init_args"])
    init["model_args"] = dict(init["model_args"], backbone_pretrained=False)  # every weight comes from the file below
    task = SemanticSegmentationTask(**init)
    state = load_file(str(HERE / "burn_scar_ca_v1_fp16.safetensors"))
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}  # stored in fp16; run in fp32
    task.load_state_dict(state, strict=True)
    return task.eval()


def predict(task: SemanticSegmentationTask, image: np.ndarray) -> np.ndarray:
    """image: (6, H, W) float reflectance 0-1. Returns (H, W) uint8 with 1 = burned."""
    stats = json.loads((HERE / "norm_stats.json").read_text())
    mean = np.asarray(stats["means"], dtype="float32")[:, None, None]
    std = np.asarray(stats["stds"], dtype="float32")[:, None, None]
    x = np.nan_to_num(image.astype("float32"), nan=0.0)  # as in training: no-data replaced by 0, then normalised
    x = torch.from_numpy((x - mean) / std)[None]
    with torch.no_grad():
        logits = task(x).output
    return logits.argmax(dim=1)[0].numpy().astype("uint8")


def main(src: str, dst: str) -> None:
    with rasterio.open(src) as r:
        image, profile = r.read(), r.profile
    if image.shape[0] != 6:
        raise SystemExit(f"expected 6 bands, got {image.shape[0]}")
    mask = predict(build_model(), image)
    profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(dst, "w", **profile) as w:
        w.write(mask, 1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])

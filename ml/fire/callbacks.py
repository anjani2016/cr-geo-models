"""Training callbacks (gate ML3b). Lightning is imported lazily so the CPU tests don't need it.

PredictionImages: after each validation epoch, renders a few validation chips as
[false colour | NBAC outline | predicted outline] and logs the PNGs to the run's MLflow artifacts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

YELLOW, CYAN = (255, 255, 0), (0, 255, 255)
FALSE_COLOUR = (5, 3, 2)  # SWIR2, NIR-narrow, red in the model's band order: burns show red/brown


def _stretch(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = band[valid] if valid.any() else band.ravel()
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    return np.clip((band - lo) / (hi - lo + 1e-9) * 255, 0, 255)


def _edges(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(mask, dtype=bool)
    m = mask
    edge[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return edge


def render_comparison(image: np.ndarray, label: np.ndarray, prediction: np.ndarray, gap: int = 6) -> np.ndarray:
    """uint8 (3, H, 3W + 2*gap): false colour; false colour + NBAC outline (yellow); false colour + predicted outline (cyan).
    Pixels labelled -1 (ignore) are shown grey in the outline panels."""
    if image.ndim != 3 or image.shape[0] < 6:
        raise ValueError(f"image must be (bands>=6, H, W), got {image.shape}")
    if label.shape != image.shape[1:] or prediction.shape != image.shape[1:]:
        raise ValueError("image, label and prediction sizes differ")
    valid = label != -1
    base = np.stack([_stretch(image[b].astype(np.float64), valid) for b in FALSE_COLOUR])
    panels = [base.copy()]
    for mask, colour in ((label == 1, YELLOW), (prediction == 1, CYAN)):
        panel = base.copy()
        panel[:, ~valid] = 128
        edge = _edges(mask)
        for c in range(3):
            panel[c][edge] = colour[c]
        panels.append(panel)
    spacer = np.full((3, image.shape[1], gap), 255.0)
    out = np.concatenate([panels[0], spacer, panels[1], spacer, panels[2]], axis=2)
    return out.astype(np.uint8)


def write_png(array: np.ndarray, path: str | Path) -> Path:
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", driver="PNG", width=array.shape[2], height=array.shape[1], count=3, dtype="uint8") as dst:
            dst.write(array)
    return path


def _logits(output: Any):
    return getattr(output, "output", output)


try:  # Only on the training machine (lightning installed); the CPU test environment has no lightning.
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover - exercised on Colab
    Callback = object  # type: ignore[assignment,misc]


class PredictionImages(Callback):  # type: ignore[misc,valid-type]
    """Logs `n_images` rendered validation chips to MLflow each `every_n_epochs` epochs. Never stops training:
    a failure to render or upload is printed and skipped."""

    def __init__(self, n_images: int = 4, every_n_epochs: int = 1):
        super().__init__()
        self.n_images, self.every_n_epochs = int(n_images), int(every_n_epochs)
        self._batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0 and self._batch is None:
            self._batch = {k: v for k, v in batch.items() if k in ("image", "mask")}

    def on_validation_epoch_end(self, trainer, pl_module):
        batch, self._batch = self._batch, None
        epoch = trainer.current_epoch
        if batch is None or trainer.sanity_checking or epoch % self.every_n_epochs:
            return
        try:
            import torch

            with torch.no_grad():
                logits = _logits(pl_module(batch["image"]))
            preds = logits.argmax(dim=1).cpu().numpy()
            images = batch["image"].float().cpu().numpy()
            masks = batch["mask"].cpu().numpy()
            logger = trainer.logger
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(min(self.n_images, images.shape[0])):
                    png = write_png(render_comparison(images[i], masks[i], preds[i]), Path(tmp) / f"val_{i:02d}.png")
                    logger.experiment.log_artifact(logger.run_id, str(png), artifact_path=f"predictions/epoch_{epoch:03d}")
        except Exception as exc:  # never kill a long GPU run over a picture
            print(f"[PredictionImages] skipped epoch {epoch}: {type(exc).__name__}: {exc}", flush=True)

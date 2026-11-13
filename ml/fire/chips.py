"""Chip grids over a fire and label rasters in the base model's convention (1 burned, 0 not burned, -1 ignore)."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry.base import BaseGeometry

BURNED, NOT_BURNED, IGNORE = 1, 0, -1


class ChipError(ValueError):
    """The chip request doesn't make sense."""


def chip_origins(bounds: tuple[float, float, float, float], chip_px: int, res_m: float, max_chips: int) -> list[tuple[float, float]]:
    """Top-left corners (projected metres) of chips that tile a fire's bounds, centred on it.

    Refuses rather than silently trimming when a fire needs more than max_chips.
    """
    if chip_px <= 0 or res_m <= 0 or max_chips <= 0:
        raise ChipError("chip_px, res_m and max_chips must be positive")
    minx, miny, maxx, maxy = bounds
    if not (maxx > minx and maxy > miny):
        raise ChipError(f"empty bounds {bounds}")
    size = chip_px * res_m
    nx, ny = math.ceil((maxx - minx) / size), math.ceil((maxy - miny) / size)
    if nx * ny > max_chips:
        raise ChipError(f"fire needs {nx * ny} chips, more than max_chips={max_chips}")
    left = (minx + maxx) / 2 - nx * size / 2
    top = (miny + maxy) / 2 + ny * size / 2
    return [(left + i * size, top - j * size) for j in range(ny) for i in range(nx)]


def chip_transform(origin: tuple[float, float], res_m: float) -> Affine:
    return Affine(res_m, 0.0, origin[0], 0.0, -res_m, origin[1])


def _burn(geoms: Iterable[BaseGeometry], value: int, transform: Affine, chip_px: int) -> np.ndarray:
    shapes = [(g, value) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((chip_px, chip_px), dtype=bool)
    raster = features.rasterize(shapes, out_shape=(chip_px, chip_px), transform=transform, fill=0, dtype="int16", all_touched=False)
    return raster == value


def label_chip(
    burned: Iterable[BaseGeometry],
    transform: Affine,
    chip_px: int,
    invalid: np.ndarray,
    ignore: Iterable[BaseGeometry] = (),
) -> np.ndarray:
    """int16 label: 1 inside a burned polygon of the fire year; -1 where imagery is invalid or inside an `ignore`
    polygon (e.g. recent earlier fires, whose scars look burned but are not this year's); 0 elsewhere.
    Invalid imagery wins over burned; burned wins over ignore."""
    if invalid.shape != (chip_px, chip_px) or invalid.dtype != bool:
        raise ChipError(f"invalid mask must be bool ({chip_px}, {chip_px}), got {invalid.dtype} {invalid.shape}")
    label = np.full((chip_px, chip_px), NOT_BURNED, dtype=np.int16)
    label[_burn(ignore, 1, transform, chip_px)] = IGNORE
    label[_burn(burned, 1, transform, chip_px)] = BURNED
    label[invalid] = IGNORE
    return label


def label_summary(label: np.ndarray) -> dict[str, int]:
    return {
        "burned": int((label == BURNED).sum()),
        "not_burned": int((label == NOT_BURNED).sum()),
        "ignore": int((label == IGNORE).sum()),
    }

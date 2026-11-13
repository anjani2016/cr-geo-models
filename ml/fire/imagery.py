"""Sentinel-2 L2A helpers: reflectance scaling, invalid-pixel masks, scene choice.

All pure functions on plain dicts and numpy arrays, so they test without network access.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


class ImageryError(ValueError):
    """A STAC item or array can't be turned into model input safely."""


def band_scaling(item: Mapping[str, Any], asset: str) -> tuple[float, float, Optional[float]]:
    """(scale, offset, nodata) for one asset, from the item's raster:bands. Fails closed if scale or offset is absent."""
    assets = item.get("assets") or {}
    if asset not in assets:
        raise ImageryError(f"{item.get('id')}: asset {asset!r} missing")
    bands = assets[asset].get("raster:bands") or []
    if len(bands) != 1:
        raise ImageryError(f"{item.get('id')}: asset {asset!r} needs exactly one raster:bands entry")
    band = bands[0]
    scale, offset = band.get("scale"), band.get("offset")
    for name, value in (("scale", scale), ("offset", offset)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ImageryError(f"{item.get('id')}: asset {asset!r} has no usable {name} ({value!r})")
    if scale <= 0:
        raise ImageryError(f"{item.get('id')}: asset {asset!r} scale must be positive")
    nodata = band.get("nodata")
    return float(scale), float(offset), (None if nodata is None else float(nodata))


def to_reflectance(dn: np.ndarray, scale: float, offset: float, nodata: Optional[float]) -> np.ndarray:
    """DN × scale + offset as float32; nodata pixels become NaN."""
    out = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    if nodata is not None:
        out[dn == nodata] = np.nan
    return out


def invalid_mask(scl: np.ndarray, invalid_classes: Sequence[int], bands: np.ndarray) -> np.ndarray:
    """True where the scene classification is an invalid class or any band is NaN."""
    if bands.ndim != 3 or bands.shape[1:] != scl.shape:
        raise ImageryError(f"bands {bands.shape} and SCL {scl.shape} don't line up")
    return np.isin(scl, list(invalid_classes)) | np.isnan(bands).any(axis=0)


def search_window(fire_end: Optional[date], year: int, days_after: int, fallback_month_day: str) -> tuple[date, date]:
    """Dates to look for a post-fire scene. Without an end date, start at the configured month-day of the fire year."""
    if isinstance(days_after, bool) or not isinstance(days_after, int) or days_after <= 0:
        raise ImageryError("days_after must be a positive integer")
    start = fire_end or date.fromisoformat(f"{year}-{fallback_month_day}")
    return start, start + timedelta(days=days_after)


def choose_scene(items: Iterable[Mapping[str, Any]], start: date, end: date, max_cloud: float) -> Optional[Mapping[str, Any]]:
    """The item with the lowest cloud cover inside [start, end] and under max_cloud; ties go to the earlier date, then the ID."""
    best = None
    for item in items:
        props = item.get("properties") or {}
        cloud = props.get("eo:cloud_cover")
        when = props.get("datetime")
        if cloud is None or when is None:
            continue
        day = datetime.fromisoformat(str(when).replace("Z", "+00:00")).date()
        if not (start <= day <= end) or float(cloud) >= max_cloud:
            continue
        key = (float(cloud), day, str(item.get("id")))
        if best is None or key < best[0]:
            best = (key, item)
    return None if best is None else best[1]

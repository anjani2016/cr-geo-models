"""Choose fires from NBAC records and give each one a fixed split.

Splits are by fire and region, never by chip: a fire inside a held-out region box is always `test`; the rest go to
`train` or `val` by a stable hash of the fire ID, so the same fire lands in the same split on every run.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional

SPLITS = ("train", "val", "test")


class SelectionError(ValueError):
    """The NBAC records or the selection config can't be used safely."""


@dataclass(frozen=True)
class Fire:
    fire_id: str
    year: int
    area_ha: float
    start: Optional[date]
    end: Optional[date]
    lon: float
    lat: float
    region: str
    split: str


def _box(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SelectionError(f"region {name!r}: bbox must be [west, south, east, north]")
    west, south, east, north = (float(v) for v in value)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise SelectionError(f"region {name!r}: bbox {value} is not a valid lon/lat box")
    return west, south, east, north


def _in_box(lon: float, lat: float, box: tuple[float, float, float, float]) -> bool:
    west, south, east, north = box
    return west <= lon <= east and south <= lat <= north


def _to_date(value: Any) -> Optional[date]:
    if value in (None, "", 0) or value != value:  # value != value catches NaN
        return None
    if isinstance(value, datetime):  # includes pandas Timestamp; must come before the date check
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text.replace("/", "-"))
    except ValueError:
        if len(text) == 8 and text.isdigit():  # YYYYMMDD
            return date(int(text[:4]), int(text[4:6]), int(text[6:]))
        raise SelectionError(f"unreadable date {value!r}")


def stable_fraction(fire_id: str) -> float:
    """A number in [0, 1) fixed by the fire ID (SHA-256), used for the train/val split."""
    digest = hashlib.sha256(fire_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def check_config(cfg: Mapping[str, Any]) -> None:
    for key in ("fields", "years", "min_area_ha", "regions", "test_regions", "val_fraction"):
        if key not in cfg:
            raise SelectionError(f"selection config is missing {key!r}")
    for key in ("fire_id", "year", "area_ha"):
        if not cfg["fields"].get(key):
            raise SelectionError(f"selection.fields.{key} must name an NBAC attribute")
    regions = cfg["regions"]
    if not isinstance(regions, dict) or not regions:
        raise SelectionError("selection.regions must map names to bboxes")
    for name, box in regions.items():
        _box(box, name)
    unknown = set(cfg["test_regions"]) - set(regions)
    if unknown:
        raise SelectionError(f"test_regions not in regions: {sorted(unknown)}")
    if not cfg["test_regions"]:
        raise SelectionError("at least one region must be held out for test")
    fraction = cfg["val_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise SelectionError("val_fraction must be between 0 and 1")
    if isinstance(cfg["min_area_ha"], bool) or not isinstance(cfg["min_area_ha"], (int, float)) or cfg["min_area_ha"] < 0:
        raise SelectionError("min_area_ha must be a non-negative number")


def select_fires(records: Iterable[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[Fire]:
    """Records are NBAC attributes plus 'lon'/'lat' of a point inside each fire (WGS84)."""
    check_config(cfg)
    fields = cfg["fields"]
    years = {int(y) for y in cfg["years"]}
    regions = {name: _box(box, name) for name, box in cfg["regions"].items()}
    test_regions = set(cfg["test_regions"])
    chosen: dict[str, Fire] = {}
    for record in records:
        missing = [fields[k] for k in ("fire_id", "year", "area_ha") if fields[k] not in record]
        missing += [k for k in ("lon", "lat") if k not in record]
        if missing:
            raise SelectionError(f"NBAC record is missing {missing}; check selection.fields against the file")
        year = int(record[fields["year"]])
        area = float(record[fields["area_ha"]])
        if not math.isfinite(area) or area < 0:
            raise SelectionError(f"fire {record[fields['fire_id']]} ({year}) has an unusable area {area!r}")
        if year not in years or area < cfg["min_area_ha"]:
            continue
        lon, lat = float(record["lon"]), float(record["lat"])
        region = next((name for name, box in regions.items() if _in_box(lon, lat, box)), None)
        if region is None:
            continue
        fire_id = f"{year}-{record[fields['fire_id']]}"
        if fire_id in chosen:
            raise SelectionError(f"duplicate fire ID {fire_id}")
        if region in test_regions:
            split = "test"
        else:
            split = "val" if stable_fraction(fire_id) < cfg["val_fraction"] else "train"
        chosen[fire_id] = Fire(
            fire_id=fire_id,
            year=year,
            area_ha=area,
            start=_to_date(record.get(fields.get("start") or "", None)),
            end=_to_date(record.get(fields.get("end") or "", None)),
            lon=lon,
            lat=lat,
            region=region,
            split=split,
        )
    return sorted(chosen.values(), key=lambda f: f.fire_id)

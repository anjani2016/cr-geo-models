"""Gate ML1 orchestrator: NBAC fires + Sentinel-2 L2A -> 512 x 512 training chips, labels and a manifest.

    python -m ml.fire.stage --config configs/ml1_fire_staging.yaml --nbac <NBAC .shp/.zip> [--nbac ...] --out <dir> [--dry-run] [--limit N]

--dry-run reads only the NBAC file and prints the fires and chip counts; it touches no imagery. A real run reads Sentinel-2
pixels over the network (STAC + cloud-optimised GeoTIFFs), so it is started only with the owner's OK.
Heavy libraries (geopandas, pystac-client) are imported inside functions so the CPU tests don't need them.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ml.fire.chips import ChipError, chip_origins, chip_transform, label_chip, label_summary
from ml.fire.imagery import band_scaling, choose_scene, invalid_mask, search_window, to_reflectance
from ml.fire.selection import Fire, fire_key, select_fires

REQUIRED_SECTIONS = ("model_bands", "selection", "imagery", "chips", "source")


class StageConfigError(ValueError):
    """The staging config is incomplete."""


def load_stage_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise StageConfigError(f"{path}: top level must be a mapping")
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise StageConfigError(f"{path}: missing sections {missing}")
    bands = cfg["model_bands"]
    if not isinstance(bands, list) or len(bands) != 6 or any(set(b) != {"model", "asset"} for b in bands):
        raise StageConfigError("model_bands must list six {model, asset} pairs in the model's band order")
    for key in ("stac_url", "collection", "scl_asset", "invalid_scl_classes", "days_after_fire", "fallback_start", "max_cloud"):
        if key not in cfg["imagery"]:
            raise StageConfigError(f"imagery.{key} is required")
    for key in ("chip_px", "res_m", "max_chips_per_fire"):
        if key not in cfg["chips"]:
            raise StageConfigError(f"chips.{key} is required")
    for key in ("dataset", "source_url", "licence", "attribution", "retrieved_on"):
        if not str(cfg["source"].get(key, "")).strip():
            raise StageConfigError(f"source.{key} is required (it goes into the data manifest)")
    return cfg


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def merge_split_fires(
    records: list[dict[str, Any]], geoms: dict[Any, Any], fields: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[Any, Any]]:
    """NBAC splits a fire that crosses a provincial or park boundary into one row per jurisdiction, sharing YEAR and
    NFIREID. Merge them: union of shapes, summed area, earliest start, latest end, a new point inside the union."""
    import shapely

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(fire_key(record[fields["year"]], record[fields["fire_id"]]), []).append(record)
    merged, merged_geoms = [], {}
    for key, rows in groups.items():
        if len(rows) == 1:
            merged.append(rows[0])
            merged_geoms[rows[0]["_row"]] = geoms[rows[0]["_row"]]
            continue
        geom = shapely.union_all([geoms[r["_row"]] for r in rows])
        point = geom.representative_point()
        out = dict(rows[0])
        out[fields["area_ha"]] = sum(float(r[fields["area_ha"]]) for r in rows)
        for name, pick in (("start", min), ("end", max)):
            col = fields.get(name)
            if col:
                dates = [_plain_date(r.get(col)) for r in rows]
                dates = [d for d in dates if d is not None]
                out[col] = pick(dates) if dates else None
        out.update(lon=point.x, lat=point.y, _merged_rows=len(rows))
        merged.append(out)
        merged_geoms[out["_row"]] = geom
    return merged, merged_geoms


def _plain_date(value: Any) -> Any:
    from ml.fire.selection import _to_date

    return _to_date(value)


def read_nbac(paths: Iterable[str | Path], fields: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[Any, Any]]:
    """NBAC records (attributes + lon/lat of a point inside each fire) and their WGS84 geometries, keyed by (file, row).
    Rows of one fire split across jurisdictions are merged (merge_split_fires)."""
    import geopandas as gpd  # staging-only dependency

    records, geoms = [], {}
    for n, path in enumerate(paths):
        gdf = gpd.read_file(path).to_crs(4326)
        for idx, row in gdf.iterrows():
            point = row.geometry.representative_point()
            record = {k: v for k, v in row.items() if k != "geometry"}
            record.update(lon=point.x, lat=point.y, _row=(n, idx))
            records.append(record)
            geoms[(n, idx)] = row.geometry
    return merge_split_fires(records, geoms, fields)


def plan_chips(
    fires: Iterable[Fire],
    geoms_by_id: Mapping[str, Any],
    cfg: Mapping[str, Any],
    all_fires: Iterable[tuple[int, Any]] = (),
) -> list[dict[str, Any]]:
    """For each fire: UTM zone, chip origins, and the polygons that set its labels.

    `all_fires` is every NBAC polygon read, as (year, WGS84 geometry). Polygons of the same year that touch the chip area are
    labelled burned; polygons from the `prior_years_ignore` years before are labelled ignore (-1). Pure (shapely + pyproj).
    """
    import shapely
    from pyproj import Transformer
    from shapely.geometry import box

    chips = cfg["chips"]
    prior = int(cfg["selection"].get("prior_years_ignore", 0))
    pool = [(int(y), g) for y, g in all_fires if g is not None and not g.is_empty]
    plans = []
    for fire in fires:
        epsg = _utm_epsg(fire.lon, fire.lat)
        to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
        geom_utm = shapely.transform(geoms_by_id[fire.fire_id], to_utm, interleaved=False)
        try:
            origins = chip_origins(geom_utm.bounds, chips["chip_px"], chips["res_m"], chips["max_chips_per_fire"])
        except ChipError as exc:  # e.g. a fire too large for max_chips_per_fire: skipped and reported, not trimmed
            plans.append({"fire": fire, "skipped": str(exc), "origins": []})
            continue
        size = chips["chip_px"] * chips["res_m"]
        area = box(min(o[0] for o in origins), min(o[1] for o in origins) - size,
                   max(o[0] for o in origins) + size, max(o[1] for o in origins))
        west, south, east, north = geoms_by_id[fire.fire_id].bounds
        near = [(y, g) for y, g in pool
                if g.bounds[0] <= east + 1 and g.bounds[2] >= west - 1 and g.bounds[1] <= north + 1 and g.bounds[3] >= south - 1]
        burned, ignore = [geom_utm], []
        for year, geom in near:
            if year == fire.year or fire.year - prior <= year < fire.year:
                g = shapely.transform(geom, to_utm, interleaved=False)
                if g.intersects(area) and not g.equals(geom_utm):
                    (burned if year == fire.year else ignore).append(g)
        plans.append({"fire": fire, "epsg": epsg, "geom_utm": geom_utm, "origins": origins, "burned": burned, "ignore": ignore})
    return plans


def _read_band(href: str, epsg: int, transform, chip_px: int, resampling) -> Any:
    import numpy as np
    import rasterio
    from rasterio.warp import reproject

    out = np.zeros((chip_px, chip_px), dtype="float32")
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out, dst_transform=transform, dst_crs=f"EPSG:{epsg}",
            resampling=resampling, src_nodata=src.nodata, dst_nodata=src.nodata if src.nodata is not None else 0,
        )
    return out


def stage(plans: list[dict[str, Any]], cfg: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Read imagery for each planned chip, write image and label GeoTIFFs, return the manifest (not yet written)."""
    import numpy as np
    import rasterio
    import shapely
    from pyproj import Transformer
    from pystac_client import Client
    from rasterio.enums import Resampling
    from shapely.geometry import box

    from ml.manifest import build_entries

    img_cfg, chips = cfg["imagery"], cfg["chips"]
    client = Client.open(img_cfg["stac_url"])
    written, log = [], []
    for plan in plans:
        fire: Fire = plan["fire"]
        if plan.get("skipped"):
            log.append({"fire": fire.fire_id, "status": "skipped", "reason": plan["skipped"]})
            continue
        start, end = search_window(fire.end, fire.year, img_cfg["days_after_fire"], img_cfg["fallback_start"])
        to_wgs84 = Transformer.from_crs(plan["epsg"], 4326, always_xy=True).transform
        size = chips["chip_px"] * chips["res_m"]
        chip_areas = [shapely.transform(box(x, y - size, x + size, y), to_wgs84, interleaved=False) for x, y in plan["origins"]]
        bbox = list(shapely.union_all(chip_areas).bounds)
        items = [i.to_dict() for i in client.search(collections=[img_cfg["collection"]], bbox=bbox,
                                                    datetime=f"{start.isoformat()}/{end.isoformat()}").items()]
        for n, (origin, area) in enumerate(zip(plan["origins"], chip_areas)):
            # One scene per chip, and only a tile whose footprint covers the whole chip (edge-of-swath tiles can be mostly empty).
            scene = choose_scene(items, start, end, img_cfg["max_cloud"], must_cover=area)
            if scene is None:
                log.append({"fire": fire.fire_id, "chip": n, "status": "no clear covering scene", "window": [str(start), str(end)]})
                continue
            scaling = [band_scaling(scene, b["asset"]) for b in cfg["model_bands"]]
            transform = chip_transform(origin, chips["res_m"])
            dn = [_read_band(scene["assets"][b["asset"]]["href"], plan["epsg"], transform, chips["chip_px"], Resampling.bilinear)
                  for b in cfg["model_bands"]]
            bands = np.stack([to_reflectance(d, *s) for d, s in zip(dn, scaling)])
            scl = _read_band(scene["assets"][img_cfg["scl_asset"]]["href"], plan["epsg"], transform, chips["chip_px"], Resampling.nearest)
            invalid = invalid_mask(scl.astype("int16"), img_cfg["invalid_scl_classes"], bands)
            label = label_chip(plan["burned"], transform, chips["chip_px"], invalid, ignore=plan["ignore"])
            valid_fraction = float((label != -1).mean())
            if valid_fraction < chips.get("min_valid_fraction", 0.0):
                log.append({"fire": fire.fire_id, "chip": n, "scene": scene["id"], "status": "too few valid pixels",
                            "valid_fraction": round(valid_fraction, 3)})
                continue
            stem = f"{fire.split}/{fire.fire_id}_{n:02d}"
            for suffix, array, dtype in (("img", np.nan_to_num(bands, nan=0.0), "float32"), ("label", label[None], "int16")):
                path = out_dir / f"{stem}_{suffix}.tif"
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", height=chips["chip_px"], width=chips["chip_px"], count=array.shape[0],
                                   dtype=dtype, crs=f"EPSG:{plan['epsg']}", transform=transform, compress="deflate") as dst:
                    dst.write(array.astype(dtype))
                written.append(str(path.relative_to(out_dir)))
            log.append({"fire": fire.fire_id, "chip": n, "scene": scene["id"], "split": fire.split, **label_summary(label)})
    src = cfg["source"]
    manifest = {k: src[k] for k in ("dataset", "source_url", "licence", "attribution", "retrieved_on")}
    manifest["files"] = build_entries(out_dir, written)
    return {"manifest": manifest, "log": log}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--nbac", required=True, action="append", help="Local NBAC .shp or .zip; repeat for several years")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Stage at most N fires (0 = all)")
    args = parser.parse_args(argv)

    cfg = load_stage_config(args.config)
    records, geoms = read_nbac(args.nbac, cfg["selection"]["fields"])
    fires = select_fires(records, cfg["selection"])
    fields = cfg["selection"]["fields"]
    row_of = {fire_key(r[fields["year"]], r[fields["fire_id"]]): r["_row"] for r in records}
    if args.limit:
        fires = fires[: args.limit]
    all_fires = [(int(r[fields["year"]]), geoms[r["_row"]]) for r in records]
    plans = plan_chips(fires, {f.fire_id: geoms[row_of[f.fire_id]] for f in fires}, cfg, all_fires)
    summary = {s: sum(len(p["origins"]) for p in plans if p["fire"].split == s) for s in ("train", "val", "test")}
    skipped = [p for p in plans if p.get("skipped")]
    print(json.dumps({"fires": len(plans) - len(skipped), "skipped": len(skipped), "chips_by_split": summary}, indent=1))
    if args.dry_run:
        for p in plans:
            f = p["fire"]
            note = f"SKIPPED: {p['skipped']}" if p.get("skipped") else f"{len(p['origins'])} chips"
            print(f.fire_id, f.region, f.split, round(f.area_ha), "ha", note, f.end or "no end date")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    result = stage(plans, cfg, args.out)
    (args.out / "staging_log.json").write_text(json.dumps(result["log"], indent=1, default=str), encoding="utf-8")
    if not result["manifest"]["files"]:
        raise SystemExit("no chips written (see staging_log.json); no manifest written")
    (args.out / "manifest.json").write_text(json.dumps(result["manifest"], indent=1), encoding="utf-8")
    print(f"wrote {len(result['manifest']['files'])} files; log entries {len(result['log'])}; run date {date.today()}")


if __name__ == "__main__":
    main()

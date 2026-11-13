from pathlib import Path

import pytest
import yaml
from shapely.geometry import box

from ml.fire.selection import select_fires
from ml.fire.stage import StageConfigError, load_stage_config, plan_chips

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "ml1_fire_staging.yaml"


def test_shipped_config_loads_and_bands_match_model_order():
    cfg = load_stage_config(CONFIG)
    assert [b["model"] for b in cfg["model_bands"]] == ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    assert cfg["chips"]["chip_px"] == 512


@pytest.mark.parametrize("drop", ["model_bands", "selection", "imagery", "chips", "source"])
def test_missing_section_fails(tmp_path, drop):
    cfg = yaml.safe_load(CONFIG.read_text())
    del cfg[drop]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(StageConfigError):
        load_stage_config(path)


def test_blank_licence_or_five_bands_fail(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["source"]["licence"] = "  "
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(StageConfigError, match="licence"):
        load_stage_config(path)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["model_bands"] = cfg["model_bands"][:5]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(StageConfigError, match="six"):
        load_stage_config(path)


def test_plan_labels_same_year_burned_and_prior_years_ignore():
    cfg = load_stage_config(CONFIG)
    fire_geom = box(-85.10, 48.00, -85.00, 48.06)           # ~7 km x 7 km, in ontario_boreal_south
    neighbour_same_year = box(-85.00, 48.00, -84.98, 48.02)  # touches the chip area
    old_scar = box(-85.12, 48.04, -85.08, 48.06)             # two years earlier
    ancient = box(-85.12, 48.00, -85.10, 48.02)              # ten years earlier: not ignored
    far_away = box(-70.0, 50.0, -69.9, 50.1)
    rec = {"NFIREID": 7, "YEAR": 2023, "POLY_HA": 4000, "AG_SDATE": None, "AG_EDATE": None, "lon": -85.05, "lat": 48.03}
    fires = select_fires([rec], cfg["selection"])
    assert fires and fires[0].split in {"train", "val"}
    pool = [(2023, fire_geom), (2023, neighbour_same_year), (2021, old_scar), (2013, ancient), (2023, far_away)]
    plan = plan_chips(fires, {fires[0].fire_id: fire_geom}, cfg, pool)[0]
    assert plan["epsg"] == 32616
    assert len(plan["burned"]) == 2 and len(plan["ignore"]) == 1
    assert len(plan["origins"]) == 1


def test_fire_too_large_is_skipped_and_reported_not_fatal():
    cfg = load_stage_config(CONFIG)
    small = box(-85.10, 48.00, -85.00, 48.06)
    huge = box(-90.0, 47.0, -84.0, 49.9)
    recs = [
        {"NFIREID": 1, "YEAR": 2023, "POLY_HA": 4000, "AG_SDATE": None, "AG_EDATE": None, "lon": -85.05, "lat": 48.03},
        {"NFIREID": 2, "YEAR": 2023, "POLY_HA": 900000, "AG_SDATE": None, "AG_EDATE": None, "lon": -87.0, "lat": 48.5},
    ]
    fires = select_fires(recs, cfg["selection"])
    plans = plan_chips(fires, {"2023-1": small, "2023-2": huge}, cfg, [])
    by_id = {p["fire"].fire_id: p for p in plans}
    assert "max_chips" in by_id["2023-2"]["skipped"] and by_id["2023-2"]["origins"] == []
    assert len(by_id["2023-1"]["origins"]) == 1


def test_split_fire_rows_are_merged():
    from datetime import date

    from ml.fire.stage import merge_split_fires

    fields = {"fire_id": "NFIREID", "year": "YEAR", "area_ha": "POLY_HA", "start": "AG_SDATE", "end": "AG_EDATE"}
    a, b, c = box(-85.10, 48.0, -85.05, 48.05), box(-85.05, 48.0, -85.00, 48.05), box(-80, 47, -79.9, 47.1)
    records = [
        {"NFIREID": 284.0, "YEAR": 2021.0, "POLY_HA": 19367.0, "AG_SDATE": "2021-07-10", "AG_EDATE": "2021-09-14", "_row": 1, "lon": 0, "lat": 0},
        {"NFIREID": 284.0, "YEAR": 2021.0, "POLY_HA": 2999.0, "AG_SDATE": "2021-07-08", "AG_EDATE": None, "_row": 2, "lon": 0, "lat": 0},
        {"NFIREID": 5.0, "YEAR": 2021.0, "POLY_HA": 900.0, "AG_SDATE": None, "AG_EDATE": None, "_row": 3, "lon": -79.95, "lat": 47.05},
    ]
    merged, geoms = merge_split_fires(records, {1: a, 2: b, 3: c}, fields)
    assert len(merged) == 2
    fire = next(r for r in merged if r["NFIREID"] == 284.0)
    assert fire["POLY_HA"] == 22366.0 and fire["_merged_rows"] == 2
    assert fire["AG_SDATE"] == date(2021, 7, 8) and fire["AG_EDATE"] == date(2021, 9, 14)
    assert geoms[fire["_row"]].area == pytest.approx(a.union(b).area)
    assert -85.10 < fire["lon"] < -85.00
    ids = [f.fire_id for f in select_fires(merged, load_stage_config(CONFIG)["selection"])]
    assert ids == ["2021-284", "2021-5"]  # merged rows no longer trip the duplicate check

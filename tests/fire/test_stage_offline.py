"""Runs ml.fire.stage.stage end to end with local GeoTIFFs and a stand-in STAC client (no network)."""

import sys
import types
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ml.fire.selection import select_fires
from ml.fire.stage import load_stage_config, plan_chips, stage
from ml.manifest import verify

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "ml1_fire_staging.yaml"
ASSETS = ["blue", "green", "red", "nir08", "swir16", "swir22"]


def _write(path, array, transform, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=dtype, crs="EPSG:32616", transform=transform, nodata=0) as dst:
        dst.write(array.astype(dtype), 1)


def test_offline_stage_writes_chips_labels_and_a_verifiable_manifest(tmp_path, monkeypatch):
    cfg = load_stage_config(CONFIG)
    cfg["chips"]["chip_px"] = 64  # small for the test; same code path
    fire_geom = box(-85.06, 48.01, -85.04, 48.02)
    rec = {"NFIREID": 7, "YEAR": 2023, "POLY_HA": 4000, "AG_SDATE": None, "AG_EDATE": "2023-08-20", "lon": -85.05, "lat": 48.015}
    fires = select_fires([rec], cfg["selection"])
    plan = plan_chips(fires, {fires[0].fire_id: fire_geom}, cfg, [(2023, fire_geom)])

    # Source rasters at 20 m covering the chip, in the fire's UTM zone (16N). DN 2000 -> reflectance 0.1.
    (ox, oy) = plan[0]["origins"][0]
    transform = from_origin(ox - 1000, oy + 1000, 20, 20)
    n = int((64 * 30 + 2000) / 20)
    hrefs = {}
    for name in ASSETS:
        p = tmp_path / f"{name}.tif"
        _write(p, np.full((n, n), 2000), transform, "uint16")
        hrefs[name] = str(p)
    scl = np.full((n, n), 4)
    scl[55:65, :] = 9  # a strip of cloud just inside the chip's top edge (the chip starts 50 rows down)
    _write(tmp_path / "scl.tif", scl, transform, "uint8")
    hrefs["scl"] = str(tmp_path / "scl.tif")

    band = {"scale": 0.0001, "offset": -0.1, "nodata": 0}
    item = {"id": "S2_TEST", "properties": {"datetime": "2023-08-25T16:00:00Z", "eo:cloud_cover": 5},
            "assets": {k: {"href": v, "raster:bands": [band]} for k, v in hrefs.items()}}

    class FakeSearch:
        def items(self):
            return [types.SimpleNamespace(to_dict=lambda: item)]

    class FakeClient:
        @staticmethod
        def open(url):
            return types.SimpleNamespace(search=lambda **kw: FakeSearch())

    monkeypatch.setitem(sys.modules, "pystac_client", types.SimpleNamespace(Client=FakeClient))
    out = tmp_path / "out"
    result = stage(plan, cfg, out)

    files = [f["path"] for f in result["manifest"]["files"]]
    assert len(files) == 2 and files[0].endswith("_img.tif") and files[1].endswith("_label.tif")
    verify(result["manifest"], out)
    with rasterio.open(out / files[0]) as img, rasterio.open(out / files[1]) as lab:
        bands, label = img.read(), lab.read(1)
        assert img.count == 6 and img.dtypes[0] == "float32" and img.crs.to_epsg() == 32616
    assert np.allclose(bands[:, label != -1], 0.1, atol=1e-4)
    log = result["log"][0]
    assert log["scene"] == "S2_TEST" and log["burned"] > 0 and log["ignore"] > 0 and log["not_burned"] > 0
    assert result["manifest"]["licence"].startswith("Labels: Open Government Licence - Canada")


def test_no_clear_scene_is_logged_not_written(tmp_path, monkeypatch):
    cfg = load_stage_config(CONFIG)
    fire_geom = box(-85.06, 48.01, -85.04, 48.02)
    rec = {"NFIREID": 7, "YEAR": 2023, "POLY_HA": 4000, "AG_SDATE": None, "AG_EDATE": "2023-08-20", "lon": -85.05, "lat": 48.015}
    fires = select_fires([rec], cfg["selection"])
    plan = plan_chips(fires, {fires[0].fire_id: fire_geom}, cfg, [(2023, fire_geom)])
    cloudy = {"id": "S2_CLOUDY", "properties": {"datetime": "2023-08-25T16:00:00Z", "eo:cloud_cover": 90}, "assets": {}}
    fake = types.SimpleNamespace(Client=types.SimpleNamespace(open=lambda url: types.SimpleNamespace(
        search=lambda **kw: types.SimpleNamespace(items=lambda: [types.SimpleNamespace(to_dict=lambda: cloudy)]))))
    monkeypatch.setitem(sys.modules, "pystac_client", fake)
    out = tmp_path / "out"
    out.mkdir()
    result = stage(plan, cfg, out)
    assert result["manifest"]["files"] == []
    assert result["log"][0]["status"] == "no clear scene"
    assert list(out.iterdir()) == []

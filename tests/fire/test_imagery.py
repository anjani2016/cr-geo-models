from datetime import date

import numpy as np
import pytest

from ml.fire.imagery import ImageryError, band_scaling, choose_scene, invalid_mask, search_window, to_reflectance


def item(scale=0.0001, offset=-0.1, nodata=0, extra=None, applied=False):
    band = {"scale": scale, "offset": offset, "nodata": nodata}
    band.update(extra or {})
    props = {} if applied is None else {"earthsearch:boa_offset_applied": applied}
    return {"id": "S2X", "properties": props, "assets": {"blue": {"raster:bands": [band]}}}


def test_scaling_read_from_item():
    assert band_scaling(item(), "blue") == (0.0001, -0.1, 0.0)


def test_offset_already_applied_is_not_applied_again():
    assert band_scaling(item(applied=True), "blue") == (0.0001, 0.0, 0.0)
    assert band_scaling(item(offset=0, applied=None), "blue") == (0.0001, 0.0, 0.0)


@pytest.mark.parametrize("applied", [None, "yes", 1])
def test_ambiguous_offset_flag_fails_closed(applied):
    with pytest.raises(ImageryError):
        band_scaling(item(applied=applied), "blue")


@pytest.mark.parametrize("scale,offset", [(None, -0.1), (0.0001, None), (0, -0.1), (-1, 0), (float("nan"), 0), (True, 0), ("0.0001", 0)])
def test_missing_or_bad_scaling_fails_closed(scale, offset):
    with pytest.raises(ImageryError):
        band_scaling(item(scale, offset), "blue")


def test_missing_asset_or_band_list_fails():
    with pytest.raises(ImageryError, match="missing"):
        band_scaling(item(), "red")
    bad = {"id": "S2X", "assets": {"blue": {"raster:bands": []}}}
    with pytest.raises(ImageryError, match="exactly one"):
        band_scaling(bad, "blue")


def test_reflectance_and_nodata():
    dn = np.array([[0, 1000], [3000, 11000]], dtype=np.uint16)
    out = to_reflectance(dn, 0.0001, -0.1, 0.0)
    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert out[1, 0] == pytest.approx(0.2, abs=1e-6)
    assert out.dtype == np.float32


def test_invalid_mask_combines_scl_and_nan():
    scl = np.array([[4, 9], [4, 4]])
    bands = np.zeros((6, 2, 2), dtype=np.float32)
    bands[2, 1, 1] = np.nan
    mask = invalid_mask(scl, [8, 9], bands)
    assert mask.tolist() == [[False, True], [False, True]]
    with pytest.raises(ImageryError):
        invalid_mask(np.zeros((3, 3)), [9], bands)


def test_search_window():
    assert search_window(date(2023, 8, 15), 2023, 45, "09-01") == (date(2023, 8, 15), date(2023, 9, 29))
    assert search_window(None, 2023, 10, "09-01") == (date(2023, 9, 1), date(2023, 9, 11))
    for bad in (0, -5, True, 1.5):
        with pytest.raises(ImageryError):
            search_window(None, 2023, bad, "09-01")


def scene(sid, day, cloud):
    return {"id": sid, "properties": {"datetime": f"{day}T16:00:00Z", "eo:cloud_cover": cloud}}


def test_choose_scene_lowest_cloud_inside_window():
    items = [scene("a", "2023-09-02", 15), scene("b", "2023-09-05", 3), scene("c", "2023-10-30", 0), scene("d", "2023-09-06", 25)]
    assert choose_scene(items, date(2023, 9, 1), date(2023, 9, 30), 20)["id"] == "b"


def test_choose_scene_ties_and_none():
    items = [scene("z", "2023-09-05", 3), scene("a", "2023-09-05", 3), scene("b", "2023-09-03", 3)]
    assert choose_scene(items, date(2023, 9, 1), date(2023, 9, 30), 20)["id"] == "b"
    assert choose_scene([scene("a", "2023-09-02", 50)], date(2023, 9, 1), date(2023, 9, 30), 20) is None
    assert choose_scene([{"id": "x", "properties": {}}], date(2023, 9, 1), date(2023, 9, 30), 20) is None


def test_choose_scene_requires_full_coverage_when_asked():
    from shapely.geometry import box

    chip = box(-85.1, 48.0, -85.0, 48.1)
    covering = dict(scene("cover", "2023-09-05", 10), geometry=box(-86, 47, -84, 49).__geo_interface__)
    partial = dict(scene("edge", "2023-09-05", 0), geometry=box(-85.05, 47, -84, 49).__geo_interface__)
    no_geom = scene("nogeom", "2023-09-05", 0)
    assert choose_scene([covering, partial, no_geom], date(2023, 9, 1), date(2023, 9, 30), 20, must_cover=chip)["id"] == "cover"
    assert choose_scene([partial, no_geom], date(2023, 9, 1), date(2023, 9, 30), 20, must_cover=chip) is None

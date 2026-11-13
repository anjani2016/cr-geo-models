import copy

import pytest

from ml.fire.selection import SelectionError, select_fires, stable_fraction

CFG = {
    "fields": {"fire_id": "NFIREID", "year": "YEAR", "area_ha": "POLY_HA", "start": "AG_SDATE", "end": "AG_EDATE"},
    "years": [2023],
    "min_area_ha": 500,
    "val_fraction": 0.5,
    "regions": {"west": [-100.0, 45.0, -80.0, 55.0], "east": [-80.0, 45.0, -60.0, 55.0]},
    "test_regions": ["east"],
}


def rec(fid, year=2023, ha=1000.0, lon=-90.0, lat=50.0, end="2023-08-15"):
    return {"NFIREID": fid, "YEAR": year, "POLY_HA": ha, "AG_SDATE": "2023-06-01", "AG_EDATE": end, "lon": lon, "lat": lat}


def test_filters_by_year_area_and_region():
    fires = select_fires([rec(1), rec(2, year=2022), rec(3, ha=10), rec(4, lon=-120.0)], CFG)
    assert [f.fire_id for f in fires] == ["2023-1"]


def test_held_out_region_is_always_test():
    fires = select_fires([rec(i, lon=-70.0) for i in range(20)], CFG)
    assert {f.split for f in fires} == {"test"}


def test_other_regions_never_test_and_split_is_stable():
    records = [rec(i) for i in range(200)]
    first = {f.fire_id: f.split for f in select_fires(records, CFG)}
    second = {f.fire_id: f.split for f in select_fires(list(reversed(records)), CFG)}
    assert first == second
    assert set(first.values()) == {"train", "val"}


def test_stable_fraction_is_deterministic_and_in_range():
    assert stable_fraction("2023-1") == stable_fraction("2023-1")
    assert all(0 <= stable_fraction(f"x{i}") < 1 for i in range(100))


def test_dates_parse_or_fail():
    fire = select_fires([rec(1, end="20230815")], CFG)[0]
    assert str(fire.end) == "2023-08-15"
    assert select_fires([rec(1, end=None)], CFG)[0].end is None
    with pytest.raises(SelectionError, match="date"):
        select_fires([rec(1, end="mid-August")], CFG)


def test_missing_nbac_field_fails_closed():
    bad = rec(1)
    del bad["POLY_HA"]
    with pytest.raises(SelectionError, match="POLY_HA"):
        select_fires([bad], CFG)


def test_duplicate_fire_id_fails():
    with pytest.raises(SelectionError, match="duplicate"):
        select_fires([rec(1), rec(1)], CFG)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("regions"),
        lambda c: c.update(test_regions=[]),
        lambda c: c.update(test_regions=["nowhere"]),
        lambda c: c.update(val_fraction=0),
        lambda c: c.update(val_fraction=1.0),
        lambda c: c.update(val_fraction=True),
        lambda c: c.update(min_area_ha=-1),
        lambda c: c["regions"].update(bad=[-80.0, 45.0, -100.0, 55.0]),
        lambda c: c["regions"].update(bad=[0, 0, 1]),
        lambda c: c["fields"].update(fire_id=""),
    ],
)
def test_bad_config_fails(mutate):
    cfg = copy.deepcopy(CFG)
    mutate(cfg)
    with pytest.raises(SelectionError):
        select_fires([rec(1)], cfg)


def test_datetime_and_nat_end_dates_become_plain_dates():
    from datetime import date, datetime

    fire = select_fires([rec(1, end=datetime(2023, 8, 20, 13, 5))], CFG)[0]
    assert type(fire.end) is date and fire.end == date(2023, 8, 20)
    assert select_fires([rec(1, end=float("nan"))], CFG)[0].end is None


@pytest.mark.parametrize("area", [float("nan"), float("inf"), -5.0])
def test_unusable_area_fails_closed(area):
    with pytest.raises(SelectionError, match="area"):
        select_fires([rec(1, ha=area)], CFG)


def test_decimal_nbac_ids_and_years_become_integers():
    record = rec(2.0)
    record["YEAR"] = 2023.0
    assert select_fires([record], CFG)[0].fire_id == "2023-2"


def test_fire_key_matches_select_fires_ids():
    from ml.fire.selection import fire_key

    assert fire_key(2023.0, 2.0) == "2023-2" == select_fires([dict(rec(2.0), YEAR=2023.0)], CFG)[0].fire_id
    assert fire_key(2023, "ON-17") == "2023-ON-17"

import numpy as np
import pytest
from shapely.geometry import box

from ml.fire.chips import BURNED, IGNORE, NOT_BURNED, ChipError, chip_origins, chip_transform, label_chip, label_summary


def test_origins_cover_and_centre_the_fire():
    origins = chip_origins((0, 0, 20000, 10000), 512, 30, 16)  # chip = 15,360 m
    assert len(origins) == 2
    xs = sorted(o[0] for o in origins)
    assert xs[0] < 0 and xs[1] + 15360 > 20000
    assert origins[0][1] > 10000 - 15360 / 2


def test_too_many_chips_refused_and_bad_inputs():
    with pytest.raises(ChipError, match="max_chips"):
        chip_origins((0, 0, 100000, 100000), 512, 30, 4)
    for args in [((0, 0, 0, 10), 512, 30, 4), ((0, 0, 10, 10), 0, 30, 4), ((0, 0, 10, 10), 512, -1, 4), ((0, 0, 10, 10), 512, 30, 0)]:
        with pytest.raises(ChipError):
            chip_origins(*args)


def test_label_burned_ignore_and_invalid_priority():
    px, res = 10, 10.0
    t = chip_transform((0.0, 100.0), res)  # covers x 0..100, y 0..100
    burned = [box(0, 50, 50, 100)]          # top-left quarter
    ignore = [box(0, 0, 100, 100)]          # whole chip from an earlier fire
    invalid = np.zeros((px, px), dtype=bool)
    invalid[0, 0] = True
    label = label_chip(burned, t, px, invalid, ignore=ignore)
    assert label[0, 0] == IGNORE            # invalid wins over burned
    assert label[1, 1] == BURNED            # burned wins over ignore
    assert label[9, 9] == IGNORE            # ignore elsewhere
    assert label_summary(label) == {"burned": 24, "not_burned": 0, "ignore": 76}


def test_label_without_polygons_is_not_burned():
    label = label_chip([], chip_transform((0.0, 10.0), 1.0), 10, np.zeros((10, 10), dtype=bool))
    assert (label == NOT_BURNED).all() and label.dtype == np.int16


@pytest.mark.parametrize("mask", [np.zeros((9, 10), dtype=bool), np.zeros((10, 10), dtype=np.uint8)])
def test_bad_invalid_mask_fails(mask):
    with pytest.raises(ChipError):
        label_chip([], chip_transform((0.0, 10.0), 1.0), 10, mask)

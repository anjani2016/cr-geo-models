import numpy as np
import pytest
import rasterio

from ml.fire.callbacks import CYAN, YELLOW, render_comparison, write_png


def _inputs(h=16, w=16):
    rng = np.random.default_rng(0)
    image = rng.normal(size=(6, h, w)).astype("float32")
    label = np.zeros((h, w), dtype="int16")
    label[4:12, 4:12] = 1
    label[0, :] = -1
    pred = np.zeros((h, w), dtype="int64")
    pred[2:10, 2:10] = 1
    return image, label, pred


def test_render_three_panels_with_outlines():
    image, label, pred = _inputs()
    out = render_comparison(image, label, pred, gap=6)
    assert out.shape == (3, 16, 16 * 3 + 12) and out.dtype == np.uint8
    nbac_panel, pred_panel = out[:, :, 22:38], out[:, :, 44:60]
    assert (nbac_panel[:, 4, 4] == YELLOW).all()       # NBAC outline corner
    assert (pred_panel[:, 2, 2] == CYAN).all()          # prediction outline corner
    assert (nbac_panel[:, 0, 8] == 128).all()           # ignore pixels shown grey
    assert (out[:, :, 16:22] == 255).all()              # white spacer


@pytest.mark.parametrize("bad", ["bands", "label", "pred"])
def test_render_rejects_mismatched_inputs(bad):
    image, label, pred = _inputs()
    if bad == "bands":
        image = image[:3]
    elif bad == "label":
        label = label[:8]
    else:
        pred = pred[:, :8]
    with pytest.raises(ValueError):
        render_comparison(image, label, pred)


def test_render_handles_all_ignore_chip():
    image, label, pred = _inputs()
    label[:] = -1
    assert render_comparison(image, label, pred).shape[0] == 3


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_write_png_roundtrip(tmp_path):
    arr = render_comparison(*_inputs())
    path = write_png(arr, tmp_path / "x.png")
    with rasterio.open(path) as d:
        assert d.count == 3 and (d.read() == arr).all()

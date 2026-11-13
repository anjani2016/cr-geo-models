# Gate ML1 plan: wildfire data staging

**Parent plan:** lake twin `docs/plans/plan_open_lake_models_colab.md` revision 3, §3 ML1. **Owner, 2026-11-12, verbatim:** "Sign off, start ML1".
**Branch:** `feat/ml1-fire-data-staging`. **Implementer and reviewer:** Claude Code (self-review per the owner's 2026-11-12 instruction in `AGENTS.md`).
**Colab units:** none. Staging runs on CPU (locally or on a Colab CPU runtime).

## Goal
Turn Canadian fire perimeters and Sentinel-2 imagery into 512 × 512 training chips in the same form as the base model's training data, so ML3
can fine-tune `Prithvi-EO-2.0-300M-BurnScars` on Canadian fires.

## Facts checked for this plan (2026-11-12)

| Fact | Value | How checked |
|---|---|---|
| Base model revision | `a3f2c410e45b8ac7417976614528a872f024d831`; licence apache-2.0 | Hugging Face API `models/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` |
| Model input | Bands in order `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`; reflectance on a 0–1 scale (config means 0.033–0.232); classes `Not burned`, `Burn scar`; `ignore_index: -1`; 512 × 512 chips | `burn_scars_config.yaml` at that revision |
| Base model training data | HLS Burn Scars dataset, licence cc-by-4.0 | Hugging Face API, dataset revision `1864285e25010d346a842e4f068b1a1d4248ed6d` |
| Fire perimeters | NBAC, yearly zips (e.g. `NBAC_2023_20260513.zip`, 1972–2025), release 2026-05-13 | Directory listing at `cwfis.cfs.nrcan.gc.ca/downloads/nbac/` |
| NBAC licence and citation | "Open Government Licence - Canada"; citation "Canadian Forest Service. National Burned Area Composite (NBAC). Natural Resources Canada, Canadian Forest Service, Northern Forestry Centre, Edmonton, Alberta. https://cwfis.cfs.nrcan.gc.ca." | NBAC metadata PDF, `pdftotext` |
| Imagery | Element84 Earth Search `sentinel-2-l2a`. Assets `blue, green, red` at 10 m and `nir08, swir16, swir22` at 20 m, each with `raster:bands` scale 0.0001 and offset −0.1, plus `scl` | One item read from the STAC API (`S2B_17UNQ_20230929_0_L2A`) |
| Burn severity (CanLaBS v2) | One Canada-wide 30 m GeoTIFF of dNBR (1985–2024); readme states no licence | CanLaBS `_readme.txt` |

## Design decisions (and why)
1. **Labels: burned / not burned from NBAC only.** That's what the base model predicts. Severity classes from CanLaBS are **deferred**: no licence is
   stated, it is one very large raster, and the readme gives no severity thresholds. Revisit before ML3 adds severity. *Scope change from plan rev 3 (ML1 step 4).*
2. **Imagery: Sentinel-2 L2A, not HLS.** It is open with no login, and the pipeline already uses it (SAT2). HLS needs an Earthdata account the owner
   would have to create. The six bands map one-to-one onto the model's bands. All bands are resampled to **30 m** to match HLS. *Scope change from plan rev 3 (ML1 step 3: "HLS or Sentinel-2").*
3. **Reflectance = DN × scale + offset, read from each item's `raster:bands`.** Fails closed if scale or offset is missing.
4. **Invalid pixels get label −1:** Sentinel-2 Scene Classification classes 0 (no data), 1 (saturated/defective), 3 (cloud shadow),
   8 and 9 (cloud medium/high probability), 10 (thin cirrus) and 11 (snow) `[verify: ESA SCL class table]`, plus any band at its nodata value.
   The class list lives in the config.
5. **Post-fire scene:** the clearest scene between the fire's end date and a set number of days after it. If NBAC has no end date, the window
   starts on the first of September of the fire year (stated in the config).
6. **Splits by fire and region, never by chip.** Every fire inside a held-out region box goes to `test`. Other fires are assigned to train or val
   by a stable hash of the fire ID, so splits don't change between runs.
7. **Output:** per chip, a 6-band float32 GeoTIFF and an int16 label GeoTIFF, plus a manifest (`ml.manifest`) with source, licence and attribution.
   Written to Drive or local disk, never git.

## Files
- `ml/fire/selection.py`: read NBAC records, check required fields (fail closed), filter, assign regions and splits
- `ml/fire/imagery.py`: scale/offset from STAC items, reflectance conversion, SCL invalid mask, scene choice
- `ml/fire/chips.py`: chip grid over a fire, label rasterising with −1 for invalid pixels
- `ml/fire/stage.py`: command-line orchestrator (network and heavy I/O; lazy imports; `--dry-run` lists fires and chips without downloading imagery)
- `configs/ml1_fire_staging.yaml`; `data_manifest/README.md` (unchanged); `requirements-dev.txt` (adds numpy, rasterio, shapely, pyproj);
  `requirements-stage.txt` (pystac-client, geopandas; staging only)
- `tests/fire/`: unit tests on synthetic arrays and records; no network
- Gate record and index row

## Blast radius
New modules only. `ml/config.py`, `ml/manifest.py`, `ml/tokens.py`, `ml/run_metadata.py` and the launcher are unchanged. No lake twin files change.
CI installs the new dev dependencies.

## Not in this gate
- Running the staging over real data. That downloads NBAC zips (tens of MB per year; 181 MB for 2023) and reads Sentinel-2 imagery. **It needs the owner's OK first** (file downloads).
- Training (ML3) and severity classes.

## Done means
- `pytest -q` passes in a clean venv.
- My own negative probes are recorded.
- `--dry-run` works on a small synthetic NBAC file.
- The gate record is written from git.

# Gate ML1: wildfire data staging

- **Repo / branch:** `cr-geo-models`, `feat/ml1-fire-data-staging` (from `c04b89b`)
- **Commit SHA:** 618b7fb1784b9e2d392bca851087f99ddc26da6b (change commit)
- **Status:** committed, self-reviewed, signed off
- **Plan:** `docs/plans/ml1_fire_data_staging.md` (this branch); parent: lake twin `plan_open_lake_models_colab.md` rev 3 §3 ML1
- **Owner direction (2026-11-12), verbatim:** "Sign off, start ML1"
- **Roles:** Implementer: Claude Code. Reviewer: Claude Code self-review per the owner's instruction recorded in `AGENTS.md`. Owner sign-off: 2026-11-12.
- **Colab units used:** 0

## Scope
Code that turns NBAC fire perimeters and Sentinel-2 L2A imagery into 512 × 512 chips and labels in the base model's format
(6 bands as reflectance on a 0–1 scale; labels 1 burned, 0 not burned, −1 ignore), with fixed splits by fire and region, plus a checksum manifest.
**No real data was staged.** A real run downloads NBAC zips and reads Sentinel-2 imagery, so it waits for the owner's OK.

## Scope changes from plan revision 3 (recorded in the ML1 plan)
1. **Severity classes deferred.** Labels are burned / not burned from NBAC only. The CanLaBS readme states no licence, the data is one Canada-wide raster, and the readme gives no class thresholds.
2. **Sentinel-2 L2A instead of HLS.** It is open with no login, and the six bands map one-to-one onto the model's bands. Everything is resampled to 30 m to match HLS.

## Files (from `git status --short --untracked-files=all`)
- `ml/fire/selection.py`: NBAC record checks, year / area / region filters, held-out test region, hash-based train/val split
- `ml/fire/imagery.py`: scale and offset read from each STAC item (fails closed if absent), reflectance, scene-classification invalid mask, search window, scene choice
- `ml/fire/chips.py`: chip grid (refuses fires above `max_chips_per_fire`), labels with priority invalid > burned > ignore > not burned
- `ml/fire/stage.py`: command line with `--dry-run`; reads one or more NBAC files; same-year neighbouring fires labelled burned; scars from the 3 previous years labelled ignore; oversized fires skipped and reported; writes no manifest when no chips are written
- `configs/ml1_fire_staging.yaml`: all gate choices in one place, marked "gate choice" or `[verify]`
- `requirements-dev.txt` (now adds numpy, rasterio, shapely, pyproj and their dependencies); `requirements-stage.txt` (new: geopandas, pystac-client and dependencies). Both pinned from clean Python 3.12 venvs with `pip freeze`
- `tests/fire/`: 5 test files, including an offline end-to-end run with local GeoTIFFs and a stand-in STAC client
- `docs/plans/ml1_fire_data_staging.md`, `docs/audit/gate_index.md` (ML1 row), this record

## Facts checked (2026-11-12)
- Base model: revision `a3f2c410e45b8ac7417976614528a872f024d831`, licence apache-2.0 (Hugging Face API).
- Input bands and normalisation: from `burn_scars_config.yaml` at that revision.
- NBAC: licence and citation read from its metadata PDF (`pdftotext`). Release `20260513` yearly zips listed on the CWFIS download page.
- Sentinel-2 asset names, scale 0.0001 and offset −0.1: read from item `S2B_17UNQ_20230929_0_L2A` on Earth Search.

## Checks run
- `.venv/bin/pytest -q` in a clean venv built from `requirements-dev.txt` only, with no geopandas or pystac-client: **161 passed**. CI will use the same setup.
- Dry run on a synthetic 4-polygon NBAC-style shapefile (staging venv): 3 fires selected, 2 train chips and 1 test chip; the 50 ha fire was filtered out.

## Self-review probes (Claude Code)
| Probe | Result | Action |
|---|---|---|
| NBAC end date passed as a `datetime` (how geopandas can return it) | **Crashed** scene choice (`TypeError: can't compare datetime.datetime to datetime.date`) | Fixed: dates become plain dates; test added |
| NaN area | **Passed the min-area filter** | Fixed: NaN, infinite or negative area fails closed; tests added |
| One fire larger than `max_chips_per_fire` | **Aborted the whole run** | Fixed: that fire is skipped and reported; test added |
| An NBAC file without `POLY_HA` | Failed closed, naming the field | OK |
| Same NBAC file passed twice | Failed closed: "duplicate fire ID 2023-7" | OK |
| Fire exactly on a region boundary | First region in config order wins (deterministic) | OK |
| STAC item without scale or offset, or with a bool, NaN, string or ≤ 0 value | Refused (tests) | OK |
| Every scene too cloudy | Logged "no clear scene"; nothing written; the command line writes no manifest | OK |
| Offline end-to-end run | Reflectance 0.1 from DN 2000 on valid pixels; cloud strip → −1; manifest verifies | OK; my first fixture put the cloud outside the chip, fixed in the test |

## Known limits
- **Not run on real data.** NBAC attribute names in the config (`NFIREID`, `YEAR`, `POLY_HA`, `AG_SDATE`, `AG_EDATE`) are `[verify]`. A dry run on the real file will fail closed if they're wrong.
- **Boundary leakage:** a train fire near the held-out Quebec box can have chips that reach into it. Splits are by fire, but pixels near the boundary could appear in both train and test chips.
- Reflectance below 0 (dark water with the −0.1 offset) is kept as is, not clipped.
- Sentinel-2 imagery isn't identical to HLS (the base model's training data). ML3 measures the effect against a zero-shot baseline.
- The network path (STAC search, cloud-optimised GeoTIFF reads) is tested only with stand-ins.

## Reviewer notes
no review text on file (self-reviewed per the owner's instruction)

## Sign-off
Owner committed on 2026-11-12. Owner's words, verbatim: "Sign off, run pilot".

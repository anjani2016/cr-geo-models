# Gate ML1a: pilot staging run and the fixes it forced

- **Repo / branch:** `cr-geo-models`, `feat/ml1-fire-data-staging` (after ML1 sign-off `c79a167`)
- **Commit SHA:** 778c285277092af7e4ec41cfb92617d554a387d2 (change commit)
- **Status:** built, self-reviewed, sign-off pending
- **Owner direction (2026-11-12), verbatim:** "Sign off, run pilot"
- **Roles:** Implementer and reviewer: Claude Code (self-review per `AGENTS.md`). Owner sign-off: pending.
- **Colab units used:** 0 (ran locally on CPU)

## What ran
- **Downloaded** (owner-approved) into `data/nbac/`, which git ignores:
  - `NBAC_2021_20260513.zip`: 35,416,989 bytes, SHA-256 `86df2814b9213b342bcd09aabc94a5e5342ef789090ab013857ba6930db8e8cf`
  - `NBAC_2018_20260513.zip`: 27,756,288 bytes, SHA-256 `9ce81934c6e6400ee6735fbfe8da8465f67bfd593b29601ead561835dd80ebde`
  - Both have 19 fields, including `YEAR`, `NFIREID`, `POLY_HA`, `AG_SDATE` and `AG_EDATE`, so the config's field names are confirmed (`pyogrio.read_info`). 2021 has 1,733 features; 2018 has 1,716.
- **Dry run on 2021 with 2018 as prior-year scars:**
  - 138 fires selected, 4 skipped as too large for `max_chips_per_fire`.
  - 231 chips planned: train 180, val 48, test 3.
- **Staging limited to 5 fires** (`--limit 5`): 18 chips written. The manifest verifies (`ml.manifest.verify`).
  - Pixels: burned 876,399; not burned 3,829,497; ignore 12,696.
  - Run time: 109.55 s with the per-chip scene fix (`/usr/bin/time`).
  - These figures come from the second-to-last run. The final run, with the offset fix, wrote the same 18 chips with the same label counts.

## Problems the pilot found, and fixes
| # | Found | Evidence | Fix |
|---|---|---|---|
| 1 | NBAC splits a fire that crosses a provincial or park boundary into rows that share `NFIREID`. The duplicate check stopped the run | 1,733 rows, 1,723 unique `NFIREID`, 20 rows in duplicate groups (e.g. fire 284: ON 19,367 ha and MB 2,999 ha) | `stage.merge_split_fires`: union of shapes, summed area, earliest start, latest end. Test added |
| 2 | Fire IDs came out as decimals (`2021-1.0`), and staging mapped IDs to rows with different formatting | NBAC stores `YEAR` and `NFIREID` as decimals | `selection.fire_key`, shared by selection and staging. Tests added |
| 3 | 3 of 5 fires were 100% "ignore" | Chip centre outside the chosen tile's footprint; tile `S2B_14VNL_20210930_0_L2A` has `s2:nodata_pixel_percentage` 73.5 | One scene per chip, chosen only among tiles whose footprint covers the whole chip (`choose_scene(must_cover=...)`). Chips with less than `min_valid_fraction` (0.5, gate choice) usable pixels are not written. Tests added. After the fix: 18 of 18 chips written, ignore 12,696 of 4.7 million pixels |
| 4 | Reflectance came out negative (visible band means −0.046 to −0.032) | The baseline 03.01 and 05.00 copies of `S2B_14UMG_20210926` have near-equal raw values (blue median 116 vs 118; NIR 1,188 vs 1,206), so the −0.1 offset listed for 05.00 is already in its pixels (`earthsearch:boa_offset_applied: true`) | `band_scaling` uses offset 0 when that flag is true, the listed offset when false, and refuses a non-zero offset with no flag, or a non-boolean flag. Tests added, including `1`, which Python treats as equal to `True` and my first version let through |

## Result after the fixes (final pilot run)
- Mean reflectance over valid pixels, band order Blue, Green, Red, NIR-narrow, SWIR1, SWIR2: **0.020, 0.032, 0.034, 0.120, 0.105, 0.067**.
  The base model's config means are 0.033, 0.057, 0.059, 0.232, 0.197, 0.119, so our autumn boreal chips are darker. ML3 should decide whether to
  normalise with statistics from our own training chips.
- Normalised Burn Ratio from NIR-narrow and SWIR2: **−0.042 on burned pixels, 0.35 on unburned pixels**, the expected burn signal. Labels and imagery line up.
- Quick-look of chip `2021-1047_05`: `ml1a_pilot_quicklook_2021-1047_05.png` in this folder. False colour (SWIR2, NIR, red) with the NBAC outline in yellow: the outline follows the burned (red/pink) area.

## Checks
- `.venv/bin/pytest -q` (clean venv from `requirements-dev.txt`): **170 passed**.

## Known limits carried into ML3
- **Test split is thin.** 2021 gives only 3 test chips (Quebec held out). Before ML3, stage more years, or choose a larger or second held-out region (owner decides).
- **Downloads for a full run** (CWFIS listing): NBAC 2019 27 MB, 2020 7 MB, 2022 22 MB, 2023 181 MB, 2024 71 MB, plus 2016–2017 (13 MB, 30 MB) for prior-year scars.
  2021 and 2018 are already downloaded. Needs the owner's OK.
- The offset rule is based on one pair of scenes plus the flag's meaning `[verify: Element84 Earth Search documentation for boa_offset_applied]`.
- Chips whose fire straddles two tiles, with no single tile covering the chip, are logged as "no clear covering scene" and dropped. Mosaicking is not built.

## Reviewer notes
no review text on file (self-reviewed per the owner's instruction)

## Sign-off
pending

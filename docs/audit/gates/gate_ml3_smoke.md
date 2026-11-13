# Gate ML3 (part 1): training build and Colab GPU smoke test

- **Repo / branch:** `cr-geo-models`, `feat/ml3-burn-scar-finetune`
- **Commits:**
  - `84c6a9b` (ML1a years 2017–2025)
  - `e9bb15d` (ML3 build)
  - `f30a696` (smoke fixes)
  - this record's commit: pending
- **Status:** smoke test done; self-reviewed; owner review pending. The full fine-tune is **not** started.
- **Plan:** `docs/plans/ml3_burn_scar_finetune.md`
- **Owner decisions (2026-11-12), verbatim:** "Build + small GPU test (Recommended)"; "Allow colab commands (Recommended)"; on monitoring: "will we use mlflow here to monitor the training?"
- **Roles:** Implementer and reviewer: Claude Code (self-review per `AGENTS.md`). Owner sign-off: pending.

## Units
| | |
|---|---|
| Session | `ml3smoke`, NVIDIA L4 23034 MiB, 22:35 → 22:57 EST |
| Balance (`colab usage`) | 599.79 → 599.16 = **0.63 units** (Colab's display at 22:57; rate shown 1.94/hr) |
| Cap | 10 units (owner) |
| Session stopped | yes; `colab sessions`: "No active sessions found on server." |

## What ran (from `smoke_summary.json`, downloaded)
- **Environment on the VM:** terratorch 1.2.14, torch 2.14.1+cu130, lightning 2.6.6, mlflow 3.18.0, Python 3.13.16.
  Pinned into `requirements-ml.txt`; the full `pip freeze` is in `docs/colab_env/2026-11-12_l4_terratorch_pip_freeze.txt`.
- **Data:** 48 chips (24 train, 8 val, 16 test, whole fires) picked by `make_subset` from the staging run while it was still in progress.
  Archive SHA-256 `ec1221d6…bdd8`, checked on the VM after reassembly. Chip list: `data/ml3_smoke/subset_chips.json` (git-ignored).
- **Training-chip band means** (valid pixels, 5,841,648 pixels): 0.0255, 0.0405, 0.0390, 0.1489, 0.1044, 0.0631.
  The base model's means are 0.0333, 0.0570, 0.0589, 0.2323, 0.1973, 0.1194, so our chips are darker, as ML1a found.
- **Base checkpoint:** `Prithvi_EO_V2_300M_BurnScars.pt` (1,297,798,380 bytes) at revision `a3f2c410…d831`.

### Results on the 16 Canadian test chips
| Run | Burn-scar IoU | Not-burned IoU | mIoU | Burn-scar class accuracy | Not-burned class accuracy |
|---|---|---|---|---|---|
| Zero-shot US BurnScars checkpoint, its own normalisation (baseline) | 0.584 | 0.833 | 0.708 | 0.997 | 0.834 |
| Smoke fine-tune: Prithvi backbone + new decoder, 6 steps, our normalisation | 0.289 | 0.421 | 0.355 | 1.000 | 0.421 |

**How to read this:** the fine-tune row is **not a result**. It checks that the training and test code paths work end to end. Six steps from a fresh
decoder is expected to do worse than the trained US model. The baseline is real but rests on 16 chips from 2 or 3 fires.

It shows the US model **over-predicts burns** on Canadian chips: it finds almost every burned pixel (0.997), but calls 17% of unburned pixels burned.

## MLflow
- `data/ml3_smoke/mlflow.db` (SQLite, git-ignored) holds 2 runs named `finetune-smoke`, both FINISHED (`sqlite3` query): fit (36 metric rows) and test (12).
- **The zero-shot run did not log to MLflow.** The trainer built by `LightningInferenceModel` has no logger. Its metrics are only in `smoke_summary.json`.
  Fix for the full run: log the baseline through the same `terratorch test` path.

## Problems found and fixed during the smoke test
| # | Problem | Evidence | Fix |
|---|---|---|---|
| 1 | macOS `tar` added AppleDouble `._*` files; `band_stats` tried to open them | `RasterioIOError: '/content/work/subset/train/._2017-1216_00_img.tif' not recognized` | `chip_pairs` skips `._*` files (test added); pack with `COPYFILE_DISABLE=1` (`f30a696`) |
| 2 | MLflow 3.18 refuses the plain-file store | `MlflowException: The filesystem tracking backend … is in maintenance mode` | SQLite store `sqlite:////content/work/mlflow.db` (`f30a696`) |
| 3 | Uploading the 273 MB archive failed | `colab upload`: "500 Server Error" for the whole archive. A 0.7 MB file and 40 MB parts went through | Split into 40 MB parts, reassembled and SHA-256-checked on the VM |
| 4 | `colab exec` hung twice and never reached the kernel | `colab status` still showed the earlier execution as last | Ran the steps with `colab ssh -s … -- <cmd>` instead. Only one SSH connection is allowed at a time (HTTP 429 when a second one tried) |

## Checks
- Local `.venv/bin/pytest -q` (clean venv from `requirements-dev.txt`): **185 passed**.

## Decisions for the owner before the full run
1. **Starting weights:** fine-tune from the **BurnScars checkpoint** (US-trained, decoder included) rather than from the general Prithvi backbone.
   The smoke run used the backbone path only to test the code.
2. **Normalisation:** keep the base model's statistics (as trained) or use our training-chip statistics. Fine-tuning can use either; ours suit Canadian chips.
3. **Moving the full dataset (~7–8 GB) to Colab:** 40 MB parts over `colab upload` is slow from home. Options:
   (a) stage directly on a Colab CPU runtime, which reads Sentinel-2 from AWS at data-centre speed;
   (b) a Google Drive or GCS bucket, which needs a one-time interactive `colab drivemount` / `colab auth` by the owner;
   (c) keep uploading parts.
4. **Run size:** epochs and units cap for the full run.

## Reviewer Verdict & Owner Sign-Off Date
- **Reviewer Verdict:** no review text on file (self-review per the owner's instruction)
- **Owner Sign-Off Date:** pending

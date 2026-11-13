# Gate ML3c: full Canadian burn-scar fine-tune

- **Repo / branch:** `cr-geo-models`, `feat/ml3c-full-finetune`
- **Commits:**
  - `e9bb149` (driver)
  - `5c9e505` (pre-staged chips)
  - `2a9e6db` (best-weights export)
  - `1756c27` (test script)
  - this record: pending
- **Status:** a model trained and tested on the held-out split. Self-reviewed; owner sign-off pending. Nothing is published.
- **Owner direction (2026-11-12/13), verbatim:**
  - "Yes, start full run"
  - "ok, skip drive tonight. going to sleep"
  - credits: "we can go more years. we have 600 credits and they replenish. 200 every month. so not to worry about credits."
- **Roles:** Implementer and reviewer: Claude Code (self-review per `AGENTS.md`). Owner sign-off: pending.
- **Units:** 598.94 → 594.19 = **4.75 units** (`colab usage`) across three L4 sessions, against a 40-unit cap.

## Model
- **Base:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` @ `a3f2c410e45b8ac7417976614528a872f024d831` (Apache-2.0), trained with TerraTorch 1.2.14.
- **Training data:**
  - NBAC 2017–2025 perimeters on Sentinel-2 L2A, staged on the Mac (local run 7,577 s by `/usr/bin/time`): 1,024 chips (train 607, val 143, test 274), 5.4 GB.
  - Copied to Colab and verified file by file against the manifest (2,048 files).
- **Test split:** every fire in the `quebec_boreal` region box, held out from training.
- **Normalisation:** band means and standard deviations from 149,510,593 valid training-chip pixels.
- **Settings:** fine-tuned from the BurnScars weights, all weights trainable, AdamW at lr 1e-4, ReduceLROnPlateau, bf16 on an NVIDIA L4.

## What happened
| Session | Window (EST) | Outcome |
|---|---|---|
| `ml3cfull` | 23:54 → ~01:17 | Staging on the VM ran at about 2.6 chips a minute (VM in asia-southeast1, imagery in AWS us-west-2), so it was stopped and the Mac's chips were streamed over `colab ssh` instead. **Baseline logged:** run `8043f176…`. Fine-tune run `finetune-burnscars-ca-v0` reached epoch 14 (val/loss 0.053); then **Colab ended the runtime** (the session disappeared between 01:16 and 01:25). Checkpoints lost; run marked KILLED with an outcome tag |
| `ml3cfull2` | 01:27 → ~02:25 | Added `BestWeightsExport` (fp16 best weights written when val/loss improves) and a Mac pull every 15 minutes; keep-alive `colab status` every 5 minutes. Rerun `finetune-burnscars-ca-v0-r2` (run `745f8057…`) reached epoch 13 (best val/loss 0.0551 at epoch 12). **Runtime ended again** between 02:22 and 02:27, despite the keep-alive. The Mac had pulled the **epoch-10** weights (val/loss 0.0572) at 02:21. Run marked KILLED with an outcome tag |
| `ml3ceval` | 02:29 → 02:44 | Weights streamed (SHA-256 `b471b653…4381` equal on both ends), test chips verified against the manifest, test run logged: `finetune-burnscars-ca-v0-r2-best-epoch10-test` (run `25cc3f6a…`). Credentials removed; session stopped |

## Result on the held-out test split (274 chips), from `trainer.test`
| Metric | US BurnScars baseline (run `8043f176…`) | Canadian fine-tune, r2 epoch 10 (run `25cc3f6a…`) |
|---|---|---|
| Burn-scar IoU | 0.566 | **0.873** |
| Not-burned IoU | 0.846 | 0.972 |
| mIoU | 0.706 | **0.922** |
| Burn-scar class accuracy (recall) | 0.980 | 0.939 |
| Not-burned class accuracy | 0.849 | **0.984** |
| Boundary mIoU | 0.060 | 0.559 |
| F1 | 0.820 | 0.959 |

**How to read this:**
- The fine-tune removes most of the US model's false burns: not-burned accuracy rises from 0.849 to 0.984. It misses a few more burned pixels: recall falls from 0.980 to 0.939.
- The scores measure **agreement with NBAC perimeters**, not field-checked ground truth. NBAC counts unburned islands inside a perimeter as burned, and its edges can be off by a pixel or two.
- There is **one** held-out region (Quebec). That's one test of transfer, not proof the model works everywhere in Canada.
- The tested weights are from **epoch 10**, saved in fp16 and cast back to fp32. They aren't the run's best epoch (12) and the run didn't finish its early-stopping schedule. A longer run might do somewhat better.
- The test split was used only for the final test. No setting was tuned on it.
- Tier: model output is `Modeled (Simulation)`.

## Artifacts
- **Weights:** `data/ml3c_r2/best_weights_fp16.pt` (649 MB) and `.json` on the owner's Mac (git-ignored). Not on the MLflow server: its artifact store is a GCS bucket Colab can't write to (TASKS.md item).
- **Summaries:** `data/ml3c_r2/full_summary.json`, `data/ml3c_r2/eval_summary.json`, `data/ml3c_r2/norm_stats.json`.

## Finding: Colab CLI GPU runtimes ended after about 55–85 minutes
Both full-run runtimes vanished from `colab sessions` while training, with no error on the VM side we could see. A 5-minute `colab status` keep-alive didn't prevent the second one.
The cause is unknown `[verify: Colab's runtime limits for CLI-created sessions; whether a browser-attached session (`colab url`) or the Pro+ background-execution setting lasts longer]`.
Until that's known, long runs need the best-weights export and pull built here, or must be split into sessions under ~50 minutes.

## Checks
- `.venv/bin/pytest -q` (clean venv from `requirements-dev.txt`): **246 passed**.

## Reviewer Verdict & Owner Sign-Off Date
- **Reviewer Verdict:** no review text on file (self-review per the owner's instruction)
- **Owner Sign-Off Date:** pending

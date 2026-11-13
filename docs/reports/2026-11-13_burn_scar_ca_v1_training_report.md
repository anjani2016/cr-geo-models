# Training report: `burn-scar-ca` v1, a Canadian burned-area model (2026-11-13)

A plain account of which models were used, every run logged to MLflow, which runs were killed and why, and what the final model scores.
Written for the model card (ML7) and as source material for a LinkedIn post. Every number comes from a gate record or the Colab ledger in this repo;
the record is named next to it.

## 1. What the model does
Given a Sentinel-2 image after a fire, it marks each 30 m pixel (chips resampled to 30 m, as the base model was trained) as **burned** or **not burned**. It was fine-tuned on Canadian fires because the
starting model was trained on US fires and over-predicted burns on Canadian imagery.

Output tier: `Modeled (Simulation)`, a model estimate, not a measurement.

## 2. Models and tools used
| Role | What | Licence / version |
|---|---|---|
| Starting model | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`, revision `a3f2c410e45b8ac7417976614528a872f024d831`: a 300-million-parameter geospatial foundation model, already fine-tuned for US burn scars | Apache-2.0 |
| Training framework | TerraTorch 1.2.14 on PyTorch 2.14.1 and Lightning 2.6.6 | open source |
| Labels | NBAC (National Burned Area Composite) fire perimeters, release 20260513, fires 2017–2025 | Open Government Licence – Canada |
| Imagery | Sentinel-2 Level-2A surface reflectance, six bands matching Prithvi's inputs, from the Earth Search catalogue | Copernicus Sentinel data, free and open. Attribution: "Contains modified Copernicus Sentinel data 2017–2026" (wording from the Sentinel Data Legal Notice; years from the scenes of the 1,024 chips, checked in gate ML7) |
| Compute | Google Colab, NVIDIA L4 GPU (23 GB), driven from a laptop with the Colab CLI | |
| Tracking | MLflow 2.15 server: metrics live during training; the final model uploaded and registered afterwards | |

## 3. Data
- **1,024 image chips** (train 607, validation 143, test 274), 5.4 GB, built on a laptop (`gate_ml3c_full_finetune.md`).
- **Split by fire and region, never by random chip,** so neighbouring chips of one fire can't sit on both sides of the split.
- **Test region:** every fire inside a Quebec boreal box was held out from training and tuning. It was used once, for the final test.
- Labels: 1 = burned (inside an NBAC perimeter), 0 = not burned, −1 = ignore (cloud, no data).

## 4. Every run, in order (MLflow experiment `cr-geo-models-fire`)
| Run | State | What it was | Why it ended that way |
|---|---|---|---|
| `baseline-us-burnscars-zero-shot` | Finished | The US model, unchanged, on a 16-chip smoke subset. Burn-scar IoU 0.584, mIoU 0.708 (`gate_ml3b_mlflow_tracking.md`) | A check that live tracking works |
| `ml3b-check-finetune-1epoch` | Finished | One training epoch to test logging, tags and images end to end | A check, not a result. Its first attempt crashed on a tagging bug (numbers passed as tags), fixed before the rerun |
| `baseline-us-burnscars-full-test` | Finished | The US model, unchanged, on the 274 Quebec test chips. **Burn-scar IoU 0.566, mIoU 0.706** | The baseline the fine-tune has to beat |
| `finetune-burnscars-ca-v0` | **Killed** | Full fine-tune, first attempt. Reached epoch 14 (validation loss 0.053) | **Colab ended the GPU runtime** about 80 minutes after the session started. Checkpoints were on that machine and were lost |
| `finetune-burnscars-ca-v0-r2` | **Killed** | Full fine-tune, second attempt, with a new safeguard: the best weights so far were written out as they improved and pulled to the laptop every 15 minutes. Reached epoch 13 (best validation loss 0.0551 at epoch 12) | **Colab ended the runtime again** about an hour after the session started, despite a keep-alive every 5 minutes. The laptop had already pulled the **epoch-10** weights (validation loss 0.0572) |
| `finetune-burnscars-ca-v0-r2-best-epoch10-test` | Finished | The rescued epoch-10 weights, on a fresh GPU session, tested on the 274 Quebec chips | **The result** (section 5) |

Earlier, a smoke test logged two runs named `finetune-smoke` to a local database, not the server. They only proved the code ran (`gate_ml3_smoke.md`).

**Why the runtimes ended:** unknown. Nothing on the machine showed an error; the session simply disappeared from Colab's list. A keep-alive didn't help.
`[verify: Colab's limits for CLI-created GPU sessions]`. The practical answer was the safeguard: keep the best weights off the machine as training goes.

## 5. Result on the held-out Quebec fires (274 chips)
From `trainer.test`, runs `baseline-us-burnscars-full-test` and `finetune-burnscars-ca-v0-r2-best-epoch10-test` (`gate_ml3c_full_finetune.md`):

| Metric | US model (baseline) | Canadian fine-tune (v1) |
|---|---|---|
| Burn-scar IoU | 0.566 | **0.873** |
| Not-burned IoU | 0.846 | 0.972 |
| mIoU | 0.706 | **0.922** |
| Burned pixels found (recall) | 0.980 | 0.939 |
| Not-burned pixels correctly kept | 0.849 | **0.984** |
| Boundary mIoU | 0.060 | 0.559 |
| F1 | 0.820 | 0.959 |

**In plain words:** the US model found nearly every burned pixel but also called about 15% of unburned land burned. The Canadian fine-tune cut those false
burns to under 2%, at the cost of missing a few more burned pixels (recall 0.98 → 0.94). Overlap with the official perimeters rose from 0.57 to 0.87.

## 6. Limits (keep these in any post or card)
- Scores measure **agreement with NBAC perimeters**, not field-checked ground truth. NBAC counts unburned islands inside a perimeter as burned.
- **One** held-out region (Quebec boreal). That's one test of transfer, not proof it works everywhere in Canada.
- The tested weights are from **epoch 10**, not the run's best epoch, because the runtime ended. A longer run might score somewhat higher.
- Burned / not burned only. **Severity** (low / moderate / high) is not in this version.

## 7. Cost
Colab's own balance, before and after each session (`docs/colab_ledger.md`): 599.79 → 594.19, **5.60 compute units** for everything above,
including both killed runs.

## 8. What's registered and what gets published
- **Registered:** `burn-scar-ca` version 1 on the project's MLflow server. It holds the epoch-10 weights (fp16, 648,942,995 bytes, SHA-256 starting
  `b471b653`), the normalisation statistics, the configs and a README.
- **To publish (ML7):** the same weights and files on Hugging Face (`centauri-research2019`), plus a Zenodo DOI for the tagged code release, linked to ORCID.

## 9. Lessons (useful for a post)
1. **Start from a foundation model, then localise.** A US-trained model transferred poorly to Canadian boreal fires; ten epochs on Canadian labels fixed most of it.
2. **Split by fire and region,** or the test score flatters the model.
3. **Plan for the runtime to vanish.** Export the best weights during training and pull them off the machine.
4. **Move compute to data, or data to compute, on purpose.** Staging imagery on the Colab machine was slow (machine and imagery in different regions); staging on the laptop and streaming chips over was faster.
5. **Track everything.** Every run, including the killed ones, is in MLflow with its commit, base-model revision, data manifest and seed, so the result can be traced and repeated.

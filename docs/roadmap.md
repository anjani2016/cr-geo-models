# Roadmap

What this repo has built, what's next, and how the work runs. Updated at the end of each gate. Gate details live in `docs/audit/gates/`;
the rules for every gate are in `AGENTS.md`.

**Last updated:** 2026-11-13 (ML7 in progress).

## Status by gate

| Gate | What | State | Record |
|---|---|---|---|
| ML-R | Repository scaffold: config with pinned model revisions, data manifests, CPU tests in CI | Done, signed off | `gate_ml_r_scaffold.md` |
| ML1 / ML1a | Wildfire training data: NBAC fire perimeters on Sentinel-2 L2A, fires 2017–2025, split by fire and region, Quebec boreal held out for testing. 1,024 chips (train 607, val 143, test 274) | Done, signed off | `gate_ml1_fire_data_staging.md`, `gate_ml1a_pilot_fixes.md` |
| ML3 smoke / ML3b | 48-chip GPU smoke test on Colab; live MLflow tracking | Done, signed off | `gate_ml3_smoke.md`, `gate_ml3b_mlflow_tracking.md` |
| ML3c | Full fine-tune of Prithvi-EO-2.0-300M BurnScars on Canadian fires: **burned area** | Done, signed off. Registered as `burn-scar-ca` v1 on the MLflow server | `gate_ml3c_full_finetune.md` |
| ML3d | **Severity classes**: train the model to label each burned pixel low / moderate / high severity (an image-segmentation model change, not lake analysis) | Parked (owner, 2026-11-13) | |
| ML5 | Lake chlorophyll-a (and possibly water clarity) from Sentinel-2: gradient-boosting baseline, then a mixture density network; train on open spectral datasets, test on held-out lakes and years | Parked (owner, 2026-11-13) | |
| ML6 | Optional: forecasting backtest of lake time series with open foundation models | Parked (owner, 2026-11-13) | |
| ML7 | Publishing: Hugging Face model card and weights, Zenodo DOI for a tagged release, ORCID link | **In progress:** card, notice, weights (safetensors) and staging built; nothing published yet | `gate_ml7_publish.md` |

**Held-out result for `burn-scar-ca` v1** (Quebec boreal fires, 274 chips, from `trainer.test`): burn-scar IoU 0.873 against 0.566 for the US
BurnScars model it started from; mIoU 0.922 against 0.706. Scores measure agreement with NBAC perimeters, not field-checked ground truth, on one
held-out region. Output tier: `Modeled (Simulation)`.

## Next: publish `burn-scar-ca` v1 (ML7), then a LinkedIn post
Owner, 2026-11-13: "I would rather focus on publishing this model and write a linkedin psot". ML3d, ML5 and ML6 are **parked** until the owner reopens them.
Mapping lakes after fires is separate work in another repo and is not part of this repo's roadmap.

**ML7 scope (only this):**
1. **Model card** (`README.md` for the Hugging Face model repo): what it does, training data and regions, held-out test, metrics against the US baseline,
   limits, tier label, licences and attributions, citation. Source: `docs/reports/2026-11-13_burn_scar_ca_v1_training_report.md`.
2. **Files:** the registered v1 weights (fp16, SHA-256 starting `b471b653`), `norm_stats.json`, configs, and a short inference example. The weights
   are on the owner's machine at `data/ml3c_r2/model_v1/` (git-ignored); check the SHA-256 before upload.
3. **Checks before publishing:** Copernicus Sentinel attribution wording (confirmed in ML7: 2017–2026); NBAC (OGL-Canada) attribution text; Prithvi (Apache-2.0) notice and
   base-model citation; a claims sweep of the card (no "verified", "real-time" or partner names); licence for the fine-tuned weights (Apache-2.0 proposed, owner decides).
4. **Publishing (owner does or confirms each step):** owner creates the Hugging Face model repo under `centauri-research2019` and a write token;
   upload; tag a code release on GitHub; Zenodo archives the release and gives a DOI; owner adds the DOI to ORCID.
5. **LinkedIn post draft** from the training report, with the limits kept in. Draft only; the owner posts.

Not in ML7: new training, severity classes, chlorophyll-a, or anything in the lake platform.

## Compute used
From `docs/colab_ledger.md` (Colab's own balance, before and after each session): 599.79 → 594.19, **5.60 units** over five L4 sessions.

## How the work runs (lessons so far)
- **Colab sessions end early.** Colab CLI GPU runtimes ended after about 55–85 minutes. Long runs export the best weights as they improve and pull them
  to a local machine, or are split into sessions under about 50 minutes.
- **Stage imagery locally.** Staging Sentinel-2 chips on the Colab VM was slow (VM and imagery in different regions). Staging locally and streaming the
  chips to the VM in batches of about 250 MB over `colab ssh` worked.
- **`colab ssh` over `colab exec`.** `exec` hung at times; one SSH connection at a time.
- **MLflow:** metrics log live from Colab. The server's artifact store can't be written from Colab, so models are uploaded from a local machine.
- **Google Drive:** mounting from the CLI failed; not used yet.
- **Credentials** are copied to the VM with mode 600 and deleted before the session stops. They are never printed or committed.

## Rules that apply to every gate
See `AGENTS.md`: plan with blast radius → owner confirms → build → self-review with adversarial probes → commit → gate record → owner sign-off.
Only numbers from commands run in the gate are quoted. No person or organisation is named as a collaborator or data provider without written confirmation;
data publishers are credited only as their licences require. Nothing is pushed, tagged, merged or published without the owner's word.

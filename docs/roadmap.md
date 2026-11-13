# Roadmap

What this repo has built, what's next, and how the work runs. Updated at the end of each gate. Gate details live in `docs/audit/gates/`;
the rules for every gate are in `AGENTS.md`.

**Last updated:** 2026-11-13.

## Status by gate

| Gate | What | State | Record |
|---|---|---|---|
| ML-R | Repository scaffold: config with pinned model revisions, data manifests, CPU tests in CI | Done, signed off | `gate_ml_r_scaffold.md` |
| ML1 / ML1a | Wildfire training data: NBAC fire perimeters on Sentinel-2 L2A, fires 2017–2025, split by fire and region, Quebec boreal held out for testing. 1,024 chips (train 607, val 143, test 274) | Done, signed off | `gate_ml1_fire_data_staging.md`, `gate_ml1a_pilot_fixes.md` |
| ML3 smoke / ML3b | One-chip GPU test on Colab; live MLflow tracking | Done, signed off | `gate_ml3_smoke.md`, `gate_ml3b_mlflow_tracking.md` |
| ML3c | Full fine-tune of Prithvi-EO-2.0-300M BurnScars on Canadian fires: **burned area** | Done, signed off. Registered as `burn-scar-ca` v1 on the MLflow server | `gate_ml3c_full_finetune.md` |
| ML3d | **Severity classes** (burned area split into low / moderate / high severity from dNBR) | Not started | |
| ML5 | Lake chlorophyll-a (and possibly water clarity) from Sentinel-2: gradient-boosting baseline, then a mixture density network; train on open spectral datasets, test on held-out lakes and years | Not started | |
| ML6 | Optional: forecasting backtest of lake time series with open foundation models | Not started | |
| ML7 | Publishing: Hugging Face model card and weights, Zenodo DOI for a tagged release, ORCID link | Not started | |

**Held-out result for `burn-scar-ca` v1** (Quebec boreal fires, 274 chips, from `trainer.test`): burn-scar IoU 0.873 against 0.566 for the US
BurnScars model it started from; mIoU 0.922 against 0.706. Scores measure agreement with NBAC perimeters, not field-checked ground truth, on one
held-out region. Output tier: `Modeled (Simulation)`.

## Suggested next order (owner decides)
1. **ML7 for `burn-scar-ca` v1:** model card, weights and configs on Hugging Face; a tagged release archived on Zenodo.
   Open items before publishing: the Copernicus Sentinel attribution wording `[verify]`, the NBAC licence attribution text, a `claims-auditor`-style pass on the card,
   and the owner's OK for each publish.
2. **ML5:** chlorophyll-a retrieval. Uses Colab units that expire around 2026-11-21.
3. **ML3d:** severity classes.
4. **ML6:** only if the lake time series are dense enough.

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

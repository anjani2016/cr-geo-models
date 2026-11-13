# Gate ML3c plan: full Canadian burn-scar fine-tune

**Owner, 2026-11-12, verbatim:** "Yes, start full run". This accepts the recommended setup:
- start from the US BurnScars checkpoint
- normalise with Canadian training-chip statistics
- stage the data on the Colab GPU machine
- up to 50 epochs with early stopping
- cap at 40 units
- log the baseline as its own MLflow run
- log live to mlflow.centauri-research.com, experiment `cr-geo-models-fire`

Credits, earlier: "not to worry about credits".

**Driver:** `scripts/ml3c_full_vm.py`, which runs on the VM. **Settings:** `configs/ml3_burn_scar.yaml` `full:` section (checked by `ml.fire.full_run.check_full_config`).

## Steps on the VM
1. **Record and preflight:** commit, GPU and settings; the MLflow login must be accepted before anything else.
2. **NBAC:** download the 12 files for 2014–2025 from CWFIS and verify them against `data_manifest/nbac_20260513_sha256.txt`. A mismatch fails closed.
3. **Stage** with `ml.fire.stage` and `configs/ml1_fire_staging.yaml`: the same code and settings as the local run, reading Sentinel-2 from AWS.
4. **Statistics** from training chips only.
5. **Baseline:** the US BurnScars checkpoint with its own normalisation, on the full held-out Quebec test split. Logged as `baseline-us-burnscars-full-test`.
6. **Fine-tune:** loaded through the same path as the baseline, all weights trainable, AdamW at lr 1e-4, ReduceLROnPlateau, EarlyStopping (patience 15)
   and ModelCheckpoint on val/loss.
   - **Time cap:** `max_time` = (40 units − units used so far − 2 reserved) ÷ 1.94 units per hour.
   - **Logging:** `finetune-burnscars-ca-v0`, with prediction images saved on the VM.
7. **Test the best checkpoint** in the same run.

## Watching the cap from the Mac
`colab usage` about every 30 minutes. The run is stopped if the balance falls 40 units below the 598.94 starting point. The session is always stopped at the end.

## Risks
- **Session lifetime:** Colab may end a long-running CLI session `[verify]`. Metrics on the MLflow server survive; checkpoints on the VM would not.
- **Data:** VM staging may differ slightly from the local run if Earth Search changed its items (counts are compared afterwards).
- Results rest on fires from 2017–2025 with the Quebec region held out. A single held-out region is one test of how the model transfers, not proof that it does.

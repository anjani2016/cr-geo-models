# Gate ML3 plan: Canadian burn-scar model on Colab (this part: build and GPU smoke test)

**Parent plan:** lake twin `docs/plans/plan_open_lake_models_colab.md` revision 3, §3 ML3.
**Branch:** `feat/ml3-burn-scar-finetune` (from the ML1/ML1a branch, which it depends on).
**Implementer and reviewer:** Claude Code (self-review per `AGENTS.md`).

**Owner decisions (2026-11-12), verbatim:**
- Overnight scope: "Build + small GPU test (Recommended)". This means: finish staging, build and self-review the ML3 training code with CPU tests, then
  one short Colab GPU smoke test on about 50 chips (zero-shot baseline plus a few training steps), **capped at 10 units**. Full training waits for the owner.
- Colab permission: "Allow colab commands (Recommended)". `colab` runs outside the sandbox through local, uncommitted Claude settings.
- Credits: "we can go more years. we have 600 credits and they replenish. 200 every month. so not to worry about credits."
- Monitoring: the owner asked "mlflow or langsmith". MLflow, because LangSmith is for LLM application traces, not training loops.

## Facts used (checked 2026-11-12)
- **Base model:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` at revision `a3f2c410e45b8ac7417976614528a872f024d831` (Apache-2.0).
  - Its TerraTorch config is vendored as `configs/vendor/prithvi_eo_v2_300m_burnscars_config.yaml`, SHA-256 `9b40e6d8…f466`.
  - Its `inference.py` loads the checkpoint with `terratorch.cli_tools.LightningInferenceModel.from_config(config, checkpoint)`.
- **TerraTorch:** latest on PyPI is 1.2.14 (PyPI JSON API). Pinned from the resolved Colab install, not guessed.
- **Data format:** the base config reads chips with `GenericNonGeoSegmentationDataModule` (`img_grep`, `label_grep`, `no_label_replace: -1`), so ML1 chips are used as they are.

## What this gate builds
1. **`ml/fire/train_prep.py`**
   - `band_stats`: per-band mean and standard deviation over **valid pixels of training chips only** (labels ≠ −1), so no val/test statistics leak in.
   - `make_subset`: a deterministic small copy of the staged chips, by fire.
   - `terratorch_config`: writes a TerraTorch YAML from the vendored config with our data folders, normalisation, epochs, batch limits and an
     MLflow logger (file store).
2. **`configs/ml3_burn_scar.yaml`:** the run config (`ml.config` keys plus a `training:` section). The base-model revision is pinned.
3. **`scripts/ml3_smoke_vm.py`:** runs on the Colab VM.
   - Zero-shot test of the US BurnScars checkpoint on our Canadian test chips (the baseline), with the model's own normalisation.
   - Then a short fine-tune from the Prithvi backbone with our normalisation: 1 epoch, a few batches.
   - Writes `smoke_summary.json` and an MLflow file store.
4. **CPU tests** for every function in `train_prep.py`.

## Smoke test procedure (one Colab L4 session)
1. `colab usage`: record the balance in `docs/colab_ledger.md`.
2. Pack a commit-pinned `git archive` and a subset of about 48 chips (24 train, 8 val, 16 test).
3. Run `colab new -s ml3smoke --gpu L4` (T4 if L4 is refused), then `upload`, `install terratorch==1.2.14 mlflow`, and `exec -f scripts/ml3_smoke_vm.py`.
4. `download` the summary, logs and `mlruns` into `data/ml3_smoke/` (git-ignored).
5. **`colab stop -s ml3smoke` in every case**, then `colab usage` again.
6. Stop early if the balance falls by 10 units from the start.

## Not in this gate
The full fine-tune, Drive-based data transfer for the full 7–8 GB chip set, and publishing.

## Blast radius
New files only, plus `requirements-dev.txt` if a CPU dependency is needed (none planned). No lake twin files change.

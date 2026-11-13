# Gate index

What each gate ID stands for. Status lives in each gate record. The plan is in the lake twin repo (`docs/plans/plan_open_lake_models_colab.md`).

| Gate ID | Stands for | Branch | Record |
| --- | --- | --- | --- |
| ML-R | Repository scaffold | `main` (first commit) | [record](gates/gate_ml_r_scaffold.md) |
| ML1 | Wildfire data staging (NBAC + Sentinel-2 L2A chips) | `feat/ml1-fire-data-staging` | [record](gates/gate_ml1_fire_data_staging.md) |
| ML1a | Pilot staging run and its fixes (split fires, covering scenes, offset) | `feat/ml1-fire-data-staging` | [record](gates/gate_ml1a_pilot_fixes.md) |
| ML3 | Canadian burn-scar fine-tune (part 1: build and Colab smoke test) | `feat/ml3-burn-scar-finetune` | [record](gates/gate_ml3_smoke.md) |
| ML3b | Live MLflow tracking on the Centauri Research server | `feat/ml3b-mlflow-tracking` | [record](gates/gate_ml3b_mlflow_tracking.md) |
| ML3c | Full Canadian burn-scar fine-tune and held-out test | `feat/ml3c-full-finetune` | [record](gates/gate_ml3c_full_finetune.md) |

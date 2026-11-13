# Colab compute-unit ledger

One row per Colab session. Units are copied from Colab's own display before and after the session. No estimates.

| Date | Gate | Runtime (GPU) | Start (local) | End (local) | Units before | Units after | Units used | MLflow run / note |
|---|---|---|---|---|---|---|---|---|
| 2026-11-12 | ML3 smoke | L4 (NVIDIA L4, 23034 MiB) | 22:35 | 22:57 | 599.79 | 599.16 | 0.63 | MLflow runs `finetune-smoke` x2 (fit, test); zero-shot metrics in `smoke_summary.json`. Rate shown 1.94/hr. Balance read at 22:57; Colab may update it later |
| 2026-11-12 | ML3b live MLflow check | L4 | 23:42 | 23:51 | 599.16 | 598.94 | 0.22 | Runs on mlflow.centauri-research.com, experiment cr-geo-models-fire: baseline-us-burnscars-zero-shot (run 45888c6b…), ml3b-check-finetune-1epoch. Credentials removed from the VM before stop |

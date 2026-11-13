# Colab compute-unit ledger

One row per Colab session. Units are copied from Colab's own display before and after the session. No estimates.

| Date | Gate | Runtime (GPU) | Start (local) | End (local) | Units before | Units after | Units used | MLflow run / note |
|---|---|---|---|---|---|---|---|---|
| 2026-11-12 | ML3 smoke | L4 (NVIDIA L4, 23034 MiB) | 22:35 | 22:57 | 599.79 | 599.16 | 0.63 | MLflow runs `finetune-smoke` x2 (fit, test); zero-shot metrics in `smoke_summary.json`. Rate shown 1.94/hr. Balance read at 22:57; Colab may update it later |
| 2026-11-12 | ML3b live MLflow check | L4 | 23:42 | 23:51 | 599.16 | 598.94 | 0.22 | Runs on mlflow.centauri-research.com, experiment cr-geo-models-fire: baseline-us-burnscars-zero-shot (run 45888c6b…), ml3b-check-finetune-1epoch. Credentials removed from the VM before stop |
| 2026-11-12/13 | ML3c ml3cfull | L4 | 23:54 | ~01:17 (runtime ended by Colab) | 598.94 | 596.36 | 2.58 | VM staging too slow; chips streamed from the Mac; baseline 8043f176…; fine-tune reached epoch 14, lost |
| 2026-11-13 | ML3c ml3cfull2 | L4 | 01:27 | ~02:25 (runtime ended by Colab) | 596.36 | 594.50 | 1.86 | r2 745f8057… reached epoch 13; epoch-10 fp16 weights pulled to the Mac |
| 2026-11-13 | ML3c ml3ceval | L4 | 02:29 | 02:44 | 594.50 | 594.19 | 0.31 | Held-out test of epoch-10 weights: burn-scar IoU 0.873, mIoU 0.922 (run 25cc3f6a…) |

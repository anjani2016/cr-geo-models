# Gate ML3b: live MLflow tracking on the Centauri Research server

- **Repo / branch:** `cr-geo-models`, `feat/ml3b-mlflow-tracking` (from `main` `3fabe25`)
- **Commit SHA:** pending
- **Status:** built, self-reviewed, live check passed 2026-11-12 23:51. Owner sign-off pending.
- **Owner direction (2026-11-12), verbatim:**
  - "i would like to plug mlflow for model training"
  - Monitoring: "Near-live on your Mac (Recommended)"
  - Logged items: "Reproducibility record (Recommended),Baseline as its own run (Recommended),Checkpoints as artifacts,Sample prediction images"
  - "https://mlflow.centauri-research.com/"
  - Credentials: "Training user, file I upload (Recommended)"
  - Checkpoints: "Best one to MLflow server"
- **Roles:** Implementer and reviewer: Claude Code (self-review per `AGENTS.md`). Owner sign-off: pending.

## Server facts (curl, 2026-11-12, no credentials sent)
- `/health` → 200 `OK`.
- `/`, `/version` and `/api/2.0/mlflow/experiments/search` → 401 "You are not authenticated", with `www-authenticate: Basic realm="mlflow"` and `server: Google Frontend`.
  So it fails closed without credentials.
- **Changed plan:** with a hosted server, training logs to it live. The SQLite snapshot-and-pull sync first planned for "near-live on your Mac" is not needed and was not built.

## Files
- `ml/tracking.py`:
  - `run_tags`: the reproducibility record as `cr.*` tags.
  - `check_credentials`: env vars or `~/.mlflow/credentials`; the file must be mode 600 or stricter and complete; fails closed and never echoes contents.
  - `export_credentials_to_env`: values are never returned.
  - `require_https`: no clear-text remote URIs.
  - `preflight`: the server must accept the credentials before any GPU work.
  - `artifact_upload_probe`.
- `ml/fire/callbacks.py`: `render_comparison` (false colour | NBAC outline | predicted outline) and the `PredictionImages` Lightning callback. It logs PNGs per validation epoch, and never stops training on a rendering or upload error.
- `ml/fire/train_prep.py` `terratorch_config`: new options:
  - `tags`
  - `log_model` (strict true/false)
  - `save_top_k` / `save_last` (best checkpoint only)
  - `prediction_images`
- `scripts/ml3b_tracking_check_vm.py`: the live check, in this order:
  1. preflight
  2. 100 MB upload probe
  3. baseline run with tags (logged through a Trainer that carries the MLflow logger, which fixes the ML3 smoke gap)
  4. one-epoch fine-tune with images and the best checkpoint as an artifact
  5. read-back of the server view
- `docs/colab_mlflow.md`: the owner's one-time setup and what the launcher does each session.
- Tests: `tests/test_tracking.py`, `tests/fire/test_callbacks.py`, and additions to `tests/fire/test_train_prep.py`.

## Checks
- `.venv/bin/pytest -q` (clean venv from `requirements-dev.txt`): **222 passed**.

## Self-review probes
| Probe | Result | Action |
|---|---|---|
| `log_model: 1` | **Accepted** at first (`1 in (True, False)` is true in Python, the same trap as ML1a) | Identity check; test added |
| Credentials file with mode 644, 640 or 604 | Refused ("chmod 600") | OK (tests) |
| Malformed credentials file | Refused, and the error text does not contain the password | OK (test) |
| `http://` or bare-host tracking URI | Refused | OK (tests) |
| A failing upload inside the image callback | Caught and printed; training continues | By design. Not testable on CPU, because there is no lightning here |
| Does MLflow 3.18 read `~/.mlflow/credentials` itself? | `[verify]` | Removed the dependency: `preflight` exports the file into env vars for this process and its children |

## Live check (Colab L4, 2026-11-12 23:42–23:51 EST, 0.22 units by `colab usage`)
- **Login:** owner's existing MLflow login in `~/.mlflow/credentials` (mode 600). Checked from the Mac first by printing only line types and status codes:
  `/version` → 200 `2.15.0`; experiment `cr-geo-models-fire` → id 1, active, `artifact_location gs://centauri-mlflow-data/artifacts/1`.
- **Artifacts:** the experiment writes artifacts straight to a GCS bucket that Colab can't write to.
  Owner's decision, verbatim: "Metrics live; artifacts later (Recommended)". Images and checkpoints stay on the VM; the 100 MB probe was dropped.
- **First run** (code `d5cbd34`):
  - record, preflight (`credentials_source: credentials-file`, client mlflow 3.18.0), configs and **baseline** all OK.
    The baseline run went to the server with 12 metrics and 9 `cr.*` tags: burn-scar IoU 0.584, mIoU 0.708, matching the smoke test.
  - **Fine-tune failed:** `TypeError: bad argument type` in `RunTag.to_proto`. Tag values like "42" passed through the TerraTorch config parser came back as numbers.
- **Fix** (`082b01f`): tags now travel as one JSON string through a `RunTags` callback. A test checks that numeric-looking values survive a YAML round trip as text.
- **Rerun:** `ml3b-check-finetune-1epoch` FINISHED on the server with 24 metrics and 9 tags.
  - 4 prediction images were written to `out_ft/predictions/epoch_000/` on the VM.
  - The best checkpoint was `epoch=0-step=6.ckpt`.
  - Sample image: `ml3b_prediction_image_epoch000_val00.png` (false colour | NBAC yellow | prediction cyan; grey = ignore, here an earlier-year scar).
- **Cleanup:** credentials removed from the VM ("credentials removed"), session stopped, `colab sessions`: none.
- **Client/server versions:** client 3.18 with server 2.15 worked for runs, metrics and tags. Artifacts and the model registry were not exercised.

## Known limits
- **Artifacts and Model Registry:** blocked until the server proxies artifacts (`--serve-artifacts`) or Colab gets bucket access. A follow-up infra gate is logged in `~/dev/TASKS.md`. Cloud Run's request-size limit `[verify]` matters for proxied 1.3 GB checkpoints.
- MLflow 3.13 in another project's environment on the Mac can't open the 3.18 smoke database (schema `f4a9b2c7d1e6` vs `da6fb0208061`). With the hosted server, the browser UI replaces any local viewer.

## Reviewer Verdict & Owner Sign-Off Date
- **Reviewer Verdict:** no review text on file (self-review per the owner's instruction)
- **Owner Sign-Off Date:** pending

# Gate ML3b: live MLflow tracking on the Centauri Research server

- **Repo / branch:** `cr-geo-models`, `feat/ml3b-mlflow-tracking` (from `main` `3fabe25`)
- **Commit SHA:** pending
- **Status:** built and self-reviewed. The live check waits for the owner's training user.
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

## Known limits
- **The live check hasn't run yet.** It needs the owner's training user and `~/.mlflow/credentials` on the Mac. That file did not exist at 2026-11-12 23:2x (`ls`).
- **Large uploads:** the server may run on Cloud Run, which caps request size `[verify]`, so 1.3 GB checkpoint uploads may fail. The 100 MB probe answers this first.
- The probe run is soft-deleted; its artifact stays until `mlflow gc` on the server.
- MLflow 3.13 in another project's environment on the Mac can't open the 3.18 smoke database (schema `f4a9b2c7d1e6` vs `da6fb0208061`). With the hosted server, the browser UI replaces any local viewer.

## Reviewer Verdict & Owner Sign-Off Date
- **Reviewer Verdict:** no review text on file (self-review per the owner's instruction)
- **Owner Sign-Off Date:** pending

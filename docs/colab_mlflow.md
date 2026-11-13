# Training runs on the Centauri Research MLflow server

Training on Colab logs live to `https://mlflow.centauri-research.com` (gate ML3b). The server uses MLflow's built-in sign-in
(HTTP Basic, realm `mlflow`). Without credentials it answers 401 to everything except `/health` (checked 2026-11-12 with curl).

## One-time setup (owner)
1. **Create a separate training user** on the server (not your admin login), e.g. `colab-trainer`, with a strong password from
   your password manager. In MLflow's auth, a user who creates an experiment gets MANAGE on it. The training runs create
   the experiment `burn-scar-ca` themselves, so no extra grant is needed `[verify against the server's auth config, e.g. default_permission]`.
2. **Save it on your Mac** in MLflow's standard credentials file, readable only by you:

   ```bash
   mkdir -p ~/.mlflow && touch ~/.mlflow/credentials && chmod 600 ~/.mlflow/credentials && open -e ~/.mlflow/credentials
   ```

   Contents:

   ```
   [mlflow]
   mlflow_tracking_username = colab-trainer
   mlflow_tracking_password = <the password>
   ```

The repo never contains the password. Claude Code never reads or prints the file. The launcher only copies it.

## What the launcher does each Colab session
1. `colab upload -s <session> ~/.mlflow/credentials /content/.mlflow_credentials`
2. `colab ssh -s <session> -- 'mkdir -p /root/.mlflow && mv /content/.mlflow_credentials /root/.mlflow/credentials && chmod 600 /root/.mlflow/credentials'`
3. Training runs `ml.tracking.preflight` first. If the file is missing, readable by others, incomplete, or refused by the server,
   the run stops **before** any GPU work.
4. Before `colab stop`: `colab ssh -s <session> -- 'rm -f /root/.mlflow/credentials'`. The VM is deleted on stop anyway.

## What every run records
- **Reproducibility tags** (`cr.*`): commit SHA, base model and revision, data description or manifest checksum, seed, GPU type, normalisation, role.
- **Metrics** per epoch: loss, IoU per class, mIoU, accuracy.
- **Baseline:** the untouched US BurnScars checkpoint gets its own run (`baseline-us-burnscars-zero-shot`), so every fine-tune sits next to it.
- **Prediction images:** 4 validation chips per epoch, as false colour | NBAC outline (yellow) | predicted outline (cyan), under `predictions/epoch_NNN/`.
- **Best checkpoint** as an artifact (about 1.3 GB). The ML3b check first uploads a 100 MB probe to confirm the server accepts large artifacts.
  If it doesn't, checkpoints stay on Colab and the best one is downloaded instead.

## Note on deleted runs
MLflow deletes runs softly. The 100 MB probe run's artifact stays in the server's artifact store until you run `mlflow gc` on the server.

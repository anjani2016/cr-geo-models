"""Gate ML3c: the full Canadian burn-scar fine-tune. Runs ON the Colab GPU VM (launched detached over `colab ssh`).

Layout: /content/work/repo (git archive with COMMIT), MLflow credentials at /root/.mlflow/credentials (mode 600).
Steps, each recorded in /content/work/full_summary.json (credentials never printed):
  record -> preflight (MLflow) -> nbac (download + SHA-256 check) -> stage (ml.fire.stage on this VM) -> stats ->
  configs -> baseline (US BurnScars, base normalisation, full test split, own MLflow run) ->
  finetune (from the BurnScars weights, our normalisation, early stopping, time cap from the units cap) ->
  test of the best checkpoint (same MLflow run).
"""

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

T0 = time.time()
WORK = Path("/content/work")
REPO = WORK / "repo"
sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = str(REPO)
summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": {}}
state: dict = {}


def save():
    (WORK / "full_summary.json").write_text(json.dumps(summary, indent=1, default=str))


def step(name, fn):
    t0 = time.time()
    try:
        summary["steps"][name] = {"ok": True, "result": fn()}
    except Exception as exc:
        summary["steps"][name] = {"ok": False, "error": repr(exc)[:800], "trace": traceback.format_exc()[-3000:]}
    summary["steps"][name]["seconds"] = round(time.time() - t0, 1)
    save()
    print(time.strftime("%H:%M:%S"), name, "ok" if summary["steps"][name]["ok"] else "FAILED", flush=True)
    return summary["steps"][name]["ok"]


def record():
    import yaml

    from ml.fire.full_run import check_full_config

    run = yaml.safe_load((REPO / "configs/ml3_burn_scar.yaml").read_text())
    state["run"], state["full"] = run, check_full_config(run["full"])
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    state["record"] = {"commit_sha": (REPO / "COMMIT").read_text().strip(), "config_name": run["name"], "seed": state["full"]["seed"],
                       "base_model_id": run["base_model"]["id"], "base_model_revision": run["base_model"]["revision"],
                       "gpu_type": gpu or "cpu", "start_from": state["full"]["start_from"]}
    return state["record"]


def preflight():
    from ml.tracking import preflight as pf

    return pf(state["full"]["tracking_uri"], state["full"]["experiment_name"])


def nbac():
    from ml.fire.full_run import read_checksums, verify_files

    expected = read_checksums(REPO / state["full"]["nbac_checksums"])
    folder = WORK / "nbac"
    folder.mkdir(exist_ok=True)
    for name in sorted(expected):
        if not (folder / name).exists():
            urllib.request.urlretrieve(state["full"]["nbac_base_url"] + name, folder / name)
    state["nbac_files"] = [str(folder / n) for n in verify_files(folder, expected)]
    return {"files": len(state["nbac_files"])}


def stage():
    staged = WORK / "staged"
    if (staged / "manifest.json").exists():
        # Chips staged elsewhere (the Mac) with the same code and config, copied here. Use them only if every file matches
        # the manifest's size and SHA-256. Staging on a VM in asia-southeast1 read Sentinel-2 from AWS us-west-2 at about
        # 2.6 chips a minute (ML3c, 2026-11-13), several times slower than the Mac.
        from ml.manifest import load_manifest, verify

        manifest = load_manifest(staged / "manifest.json")
        verify(manifest, staged)
        counts = {s: len(list((staged / s).glob("*_img.tif"))) for s in ("train", "val", "test")}
        if min(counts.values()) == 0:
            raise RuntimeError(f"a split is empty: {counts}")
        return {"chips": counts, "source": "pre-staged, manifest verified", "files": len(manifest["files"])}
    cmd = [sys.executable, "-m", "ml.fire.stage", "--config", str(REPO / state["full"]["staging_config"]), "--out", str(WORK / "staged")]
    for f in state["nbac_files"]:
        cmd += ["--nbac", f]
    env = {**os.environ, "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), env=env, timeout=6 * 3600)
    (WORK / "stage.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"staging exited {r.returncode}: {r.stderr[-1500:]}")
    counts = {s: len(list((WORK / "staged" / s).glob("*_img.tif"))) for s in ("train", "val", "test")}
    if min(counts.values()) == 0:
        raise RuntimeError(f"a split is empty: {counts}")
    return {"chips": counts, "stdout_tail": r.stdout[-600:]}


def stats():
    from ml.fire.train_prep import band_stats

    state["stats"] = band_stats(WORK / "staged/train")
    return state["stats"]


def configs():
    import yaml
    from huggingface_hub import hf_hub_download

    from ml.fire.train_prep import terratorch_config, write_yaml
    from ml.tracking import run_tags

    run, full = state["run"], state["full"]
    base = yaml.safe_load((REPO / run["base_model"]["vendored_config"]).read_text())
    bm = run["base_model"]
    state["ckpt"] = hf_hub_download(bm["id"], bm["checkpoint_file"], revision=bm["revision"], local_dir=str(WORK / "ckpt"))
    common = {"max_epochs": full["max_epochs"], "batch_size": full["batch_size"], "num_workers": full.get("num_workers", 4),
              "precision": full["precision"], "experiment_name": full["experiment_name"], "tracking_uri": full["tracking_uri"],
              "seed": full["seed"]}
    data = str(WORK / "staged")
    chips = summary["steps"]["stage"]["result"]["chips"]
    data_tag = f"NBAC 2017-2025 on Sentinel-2 L2A, staged on the VM: train {chips['train']}, val {chips['val']}, test {chips['test']}"
    bi = base["data"]["init_args"]
    write_yaml(terratorch_config(base, data, bi["means"], bi["stds"], dict(common, run_name="baseline-us-burnscars-full-test",
                                                                            output_dir=str(WORK / "out_baseline")),
                                 tags=run_tags(state["record"], {"role": "baseline", "normalisation": "base model", "data": data_tag})),
               WORK / "baseline.yaml")
    means, stds = (state["stats"]["means"], state["stats"]["stds"]) if full["normalisation"] == "training_chips" else (bi["means"], bi["stds"])
    write_yaml(terratorch_config(base, data, means, stds, dict(common, run_name="finetune-burnscars-ca-v0", output_dir=str(WORK / "out_ft")),
                                 tags=run_tags(state["record"], {"role": "finetune", "normalisation": full["normalisation"], "data": data_tag,
                                                                 "max_units": full["max_units"], "lr": full["lr"]})),
               WORK / "finetune.yaml")
    return {"checkpoint": state["ckpt"]}


def _tags(cfg):
    cb = next((c for c in cfg["trainer"]["callbacks"] if c["class_path"] == "ml.fire.callbacks.RunTags"), None)
    return json.loads(cb["init_args"]["tags_json"]) if cb else {}


def baseline():
    import yaml
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import MLFlowLogger
    from terratorch.cli_tools import LightningInferenceModel

    cfg = yaml.safe_load((WORK / "baseline.yaml").read_text())
    lim = LightningInferenceModel.from_config(str(WORK / "baseline.yaml"), state["ckpt"])
    la = cfg["trainer"]["logger"]["init_args"]
    logger = MLFlowLogger(experiment_name=la["experiment_name"], tracking_uri=la["tracking_uri"], run_name=la["run_name"], tags=_tags(cfg))
    trainer = Trainer(logger=logger, precision=state["full"]["precision"], enable_progress_bar=False)
    return {"run_id": logger.run_id, "metrics": trainer.test(model=lim.model, datamodule=lim.datamodule)}


def finetune():
    import torch
    import yaml
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from terratorch.cli_tools import LightningInferenceModel

    from ml.fire.callbacks import PredictionImages, RunTags
    from ml.fire.full_run import lightning_max_time, max_train_hours

    full = state["full"]
    seed_everything(full["seed"], workers=True)
    cfg = yaml.safe_load((WORK / "finetune.yaml").read_text())
    # Builds the task and datamodule from our config, then loads the BurnScars weights (same path as the baseline).
    lim = LightningInferenceModel.from_config(str(WORK / "finetune.yaml"), state["ckpt"] if full["start_from"] == "burnscars_checkpoint" else None)
    model, dm = lim.model, lim.datamodule
    frozen_before = sum(1 for p in model.parameters() if not p.requires_grad)
    for p in model.parameters():  # config says freeze_backbone: false; the inference loader may leave weights frozen
        p.requires_grad_(True)
    state["frozen_before"] = frozen_before
    lr = float(full["lr"])

    def configure_optimizers():  # base config: AdamW lr 1e-4, ReduceLROnPlateau(val/loss, factor 0.5, patience 4)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}

    model.configure_optimizers = configure_optimizers
    elapsed_units = (time.time() - T0) / 3600 * float(full["rate_units_per_hour"])
    hours = max_train_hours(full, reserve_units=elapsed_units + 2.0)  # 2 units kept for setup before this script and the final test
    la = cfg["trainer"]["logger"]["init_args"]
    logger = MLFlowLogger(experiment_name=la["experiment_name"], tracking_uri=la["tracking_uri"], run_name=la["run_name"])
    ckpt_cb = ModelCheckpoint(dirpath=str(WORK / "out_ft/checkpoints"), monitor="val/loss", mode="min", save_top_k=1, save_last=True)
    trainer = Trainer(
        logger=logger, max_epochs=full["max_epochs"], max_time=lightning_max_time(hours), precision=full["precision"],
        log_every_n_steps=10, enable_progress_bar=False, default_root_dir=str(WORK / "out_ft"),
        callbacks=[ckpt_cb, EarlyStopping(monitor="val/loss", patience=full["early_stopping_patience"]),
                   LearningRateMonitor(logging_interval="epoch"), RunTags(json.dumps(_tags(cfg))),
                   PredictionImages(n_images=full.get("prediction_images", 4), out_dir=str(WORK / "out_ft/predictions"))],
    )
    trainer.fit(model, datamodule=dm)
    state["trainer"], state["model"], state["dm"] = trainer, model, dm
    return {"run_id": logger.run_id, "epochs_run": trainer.current_epoch, "params_frozen_before": state["frozen_before"],
            "best_model_path": ckpt_cb.best_model_path,
            "best_val_loss": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
            "max_time": lightning_max_time(hours), "stopped_early": trainer.should_stop}


def test_best():
    trainer = state["trainer"]
    return {"metrics": trainer.test(model=state["model"], datamodule=state["dm"], ckpt_path="best")}


if step("record", record) and step("preflight", preflight) and step("nbac", nbac) and step("stage", stage) \
        and step("stats", stats) and step("configs", configs):
    step("baseline", baseline)
    if step("finetune", finetune):
        step("test_best", test_best)
summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
summary["wall_hours"] = round((time.time() - T0) / 3600, 3)
save()
print("done", flush=True)

"""Gate ML3b live check. Runs ON the Colab VM (same layout as the ML3 smoke test: /content/work/repo, /content/work/subset).

1. Preflight: credentials present (file mode 600), HTTPS, the MLflow server accepts them, experiment available.
2. Baseline run on the server: the untouched US BurnScars checkpoint tested on the subset, with the reproducibility tags.
3. One-epoch fine-tune on the subset: metrics and tags live on the server; prediction images and the best checkpoint
   stay on the VM (the experiment's artifacts go straight to gs://centauri-mlflow-data, which Colab can't write to yet;
   owner 2026-11-12: "Metrics live; artifacts later").
Writes /content/work/tracking_check.json. Credentials are never printed.
"""

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/content/work")
REPO = WORK / "repo"
sys.path.insert(0, str(REPO))
TRACKING_URI = "https://mlflow.centauri-research.com"
EXPERIMENT = "cr-geo-models-fire"  # created by the owner on the server (experiment id 1)
summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": {}}


def step(name, fn):
    t0 = time.time()
    try:
        summary["steps"][name] = {"ok": True, "result": fn()}
    except Exception as exc:
        summary["steps"][name] = {"ok": False, "error": repr(exc)[:500], "trace": traceback.format_exc()[-2500:]}
    summary["steps"][name]["seconds"] = round(time.time() - t0, 1)
    (WORK / "tracking_check.json").write_text(json.dumps(summary, indent=1, default=str))
    print(name, "ok" if summary["steps"][name]["ok"] else "FAILED", flush=True)
    return summary["steps"][name]["ok"]


def record():
    import yaml

    run = yaml.safe_load((REPO / "configs/ml3_burn_scar.yaml").read_text())
    commit = (REPO / "COMMIT").read_text().strip() if (REPO / "COMMIT").exists() else "unknown"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"commit_sha": commit, "config_name": run["name"], "seed": run["seed"], "base_model_id": run["base_model"]["id"],
            "base_model_revision": run["base_model"]["revision"], "gpu_type": gpu or "cpu",
            "data": "ML3 smoke subset 24/8/16 chips (SHA-256 ec1221d6...bdd8)"}


def preflight():
    from ml.tracking import preflight as pf

    return pf(TRACKING_URI, EXPERIMENT)


def configs():
    import yaml
    from huggingface_hub import hf_hub_download

    from ml.fire.train_prep import band_stats, terratorch_config, write_yaml
    from ml.tracking import run_tags

    run = yaml.safe_load((REPO / "configs/ml3_burn_scar.yaml").read_text())
    base = yaml.safe_load((REPO / run["base_model"]["vendored_config"]).read_text())
    bm = run["base_model"]
    ckpt = hf_hub_download(bm["id"], bm["checkpoint_file"], revision=bm["revision"], local_dir=str(WORK / "ckpt"))
    stats = band_stats(WORK / "subset/train")
    rec = summary["steps"]["record"]["result"]
    common = dict(run["training"], tracking_uri=TRACKING_URI, experiment_name=EXPERIMENT)
    base_init = base["data"]["init_args"]
    zs = dict(common, run_name="baseline-us-burnscars-zero-shot", output_dir=str(WORK / "out_zs"))
    write_yaml(terratorch_config(base, str(WORK / "subset"), base_init["means"], base_init["stds"], zs,
                                 tags=run_tags(rec, {"role": "baseline", "normalisation": "base model"})), WORK / "baseline.yaml")
    ft = dict(common, run_name="ml3b-check-finetune-1epoch", output_dir=str(WORK / "out_ft"), max_epochs=1,
              limit_train_batches=6, limit_val_batches=2, log_model=False, save_last=False, save_top_k=1, prediction_images=4)
    write_yaml(terratorch_config(base, str(WORK / "subset"), stats["means"], stats["stds"], ft,
                                 tags=run_tags(rec, {"role": "tracking-check", "normalisation": "training chips"})), WORK / "finetune.yaml")
    return {"checkpoint": ckpt}


def baseline():
    """Zero-shot test of the US checkpoint, logged to MLflow through a Trainer that carries the logger."""
    import yaml
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import MLFlowLogger
    from terratorch.cli_tools import LightningInferenceModel

    cfg = yaml.safe_load((WORK / "baseline.yaml").read_text())
    lim = LightningInferenceModel.from_config(str(WORK / "baseline.yaml"), summary["steps"]["configs"]["result"]["checkpoint"])
    largs = cfg["trainer"]["logger"]["init_args"]
    tag_cb = next((c for c in cfg["trainer"]["callbacks"] if c["class_path"] == "ml.fire.callbacks.RunTags"), None)
    tags = json.loads(tag_cb["init_args"]["tags_json"]) if tag_cb else None
    logger = MLFlowLogger(experiment_name=largs["experiment_name"], tracking_uri=largs["tracking_uri"],
                          run_name=largs["run_name"], tags=tags)
    trainer = Trainer(logger=logger, precision=cfg["trainer"].get("precision", "32"), enable_progress_bar=False)
    metrics = trainer.test(model=lim.model, datamodule=lim.datamodule)
    return {"run_id": logger.run_id, "metrics": metrics}


def finetune():
    r = subprocess.run(["terratorch", "fit", "-c", str(WORK / "finetune.yaml")], capture_output=True, text=True, timeout=3600,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    (WORK / "finetune.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"terratorch fit exited {r.returncode}; tail: {r.stderr[-1500:]}")
    return {"ok": True}


def server_view():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], max_results=10)
    out = []
    for r in runs:
        out.append({"run_name": r.info.run_name, "status": r.info.status, "n_metrics": len(r.data.metrics),
                    "tags": {k: v for k, v in r.data.tags.items() if k.startswith("cr.")}})
    images = sorted(str(p.relative_to(WORK)) for p in (WORK / "out_ft" / "predictions").rglob("*.png"))
    ckpts = sorted(p.name for p in (WORK / "out_ft" / "checkpoints").glob("*.ckpt"))
    return {"runs": out, "local_images": images, "local_checkpoints": ckpts}


if step("record", record) and step("preflight", preflight):
    if step("configs", configs):
        if not (WORK / "baseline_done").exists() and step("baseline", baseline):
            (WORK / "baseline_done").write_text("1")
        step("finetune", finetune)
    step("server_view", server_view)
summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(WORK / "tracking_check.json").write_text(json.dumps(summary, indent=1, default=str))
print("done", flush=True)

"""Gate ML3c: test rescued best weights on the held-out split. Runs ON a Colab GPU VM.

Expects /content/work/repo (with COMMIT), /content/work/staged/{test/, manifest.json}, /content/work/weights/best_weights_fp16.pt
(+ .json), /content/work/norm_stats.json (training-chip means/stds of the run the weights came from), and MLflow credentials
at /root/.mlflow/credentials. Env: EVAL_RUN_NAME, SOURCE_RUN_ID. Writes /content/work/eval_summary.json.
Only the test files are checked against the manifest (train/val are not on this VM).
"""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/content/work")
REPO = WORK / "repo"
sys.path.insert(0, str(REPO))
summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": {}}
state: dict = {}


def step(name, fn):
    t0 = time.time()
    try:
        summary["steps"][name] = {"ok": True, "result": fn()}
    except Exception as exc:
        summary["steps"][name] = {"ok": False, "error": repr(exc)[:800], "trace": traceback.format_exc()[-3000:]}
    summary["steps"][name]["seconds"] = round(time.time() - t0, 1)
    (WORK / "eval_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(time.strftime("%H:%M:%S"), name, "ok" if summary["steps"][name]["ok"] else "FAILED", flush=True)
    return summary["steps"][name]["ok"]


def setup():
    import yaml

    from ml.fire.full_run import check_full_config
    from ml.manifest import load_manifest, verify
    from ml.tracking import preflight

    run = yaml.safe_load((REPO / "configs/ml3_burn_scar.yaml").read_text())
    full = check_full_config(run["full"])
    manifest = load_manifest(WORK / "staged/manifest.json")
    test_only = dict(manifest, files=[f for f in manifest["files"] if f["path"].startswith("test/")])
    verify(test_only, WORK / "staged")
    for split in ("train", "val"):  # the datamodule config names these folders; they stay empty for a test-only run
        (WORK / "staged" / split).mkdir(exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    state.update(run=run, full=full, gpu=gpu)
    return {"test_files_verified": len(test_only["files"]), "preflight": preflight(full["tracking_uri"], full["experiment_name"]),
            "weights_note": json.loads((WORK / "weights/best_weights_fp16.json").read_text()), "gpu": gpu}


def evaluate():
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import MLFlowLogger
    from terratorch.cli_tools import LightningInferenceModel

    from ml.fire.train_prep import terratorch_config, write_yaml
    from ml.tracking import run_tags

    run, full = state["run"], state["full"]
    base = yaml.safe_load((REPO / run["base_model"]["vendored_config"]).read_text())
    bm = run["base_model"]
    ckpt = hf_hub_download(bm["id"], bm["checkpoint_file"], revision=bm["revision"], local_dir=str(WORK / "ckpt"))
    stats = json.loads((WORK / "norm_stats.json").read_text())
    note = json.loads((WORK / "weights/best_weights_fp16.json").read_text())
    cfg = terratorch_config(base, str(WORK / "staged"), stats["means"], stats["stds"],
                            {"max_epochs": 1, "batch_size": full["batch_size"], "num_workers": 4, "precision": full["precision"],
                             "experiment_name": full["experiment_name"], "tracking_uri": full["tracking_uri"],
                             "output_dir": str(WORK / "out_eval"), "run_name": os.environ["EVAL_RUN_NAME"]})
    write_yaml(cfg, WORK / "eval.yaml")
    lim = LightningInferenceModel.from_config(str(WORK / "eval.yaml"), ckpt)  # builds task + datamodule (same path as training)
    weights = torch.load(WORK / "weights/best_weights_fp16.pt", map_location="cpu")
    weights = {k: v.float() if v.is_floating_point() else v for k, v in weights.items()}
    missing, unexpected = lim.model.load_state_dict(weights, strict=True), None
    tags = run_tags({"commit_sha": (REPO / "COMMIT").read_text().strip(), "base_model_id": bm["id"], "base_model_revision": bm["revision"],
                     "gpu_type": state["gpu"]},
                    {"role": "finetune-test", "source_run_id": os.environ.get("SOURCE_RUN_ID"), "weights_epoch": note.get("epoch"),
                     "weights_val_loss": note.get("val/loss"), "weights_dtype": "float16 (cast to float32 for test)",
                     "normalisation": "training chips", "data": "held-out test split, NBAC 2017-2025 (quebec_boreal region)"})
    logger = MLFlowLogger(experiment_name=full["experiment_name"], tracking_uri=full["tracking_uri"],
                          run_name=os.environ["EVAL_RUN_NAME"], tags=tags)
    trainer = Trainer(logger=logger, precision=full["precision"], enable_progress_bar=False)
    return {"run_id": logger.run_id, "metrics": trainer.test(model=lim.model, datamodule=lim.datamodule)}


if step("setup", setup):
    step("evaluate", evaluate)
summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(WORK / "eval_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print("done", flush=True)

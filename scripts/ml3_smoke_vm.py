"""Gate ML3 smoke test. Runs ON the Colab VM (sent with `colab exec -f`), never locally.

Expects /content/work/repo (git archive of a commit) and /content/work/subset/{train,val,test}.
Writes /content/work/smoke_summary.json and an MLflow file store at /content/work/mlruns.
Every step is time-boxed and its error, if any, is recorded rather than hidden.
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
summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": {}}


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        summary["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": out}
    except Exception as exc:  # recorded, not swallowed: the summary shows it
        summary["steps"][name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": repr(exc),
                                  "trace": traceback.format_exc()[-3000:]}
    (WORK / "smoke_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(name, "ok" if summary["steps"][name]["ok"] else "FAILED", flush=True)
    return summary["steps"][name]


def gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    import importlib.metadata as md
    return {"gpu": r.stdout.strip(), "terratorch": md.version("terratorch"), "torch": md.version("torch"),
            "python": sys.version.split()[0]}


def setup():
    import yaml
    from ml.config import load_config
    from ml.fire.train_prep import band_stats, terratorch_config, write_yaml

    run = yaml.safe_load((REPO / "configs/ml3_burn_scar.yaml").read_text())
    load_config(REPO / "configs/ml3_burn_scar.yaml")  # fails closed on an unpinned base model
    base = yaml.safe_load((REPO / run["base_model"]["vendored_config"]).read_text())
    stats = band_stats(WORK / "subset/train")
    base_init = base["data"]["init_args"]
    training = dict(run["training"])
    if "T4" in summary["steps"].get("gpu", {}).get("result", {}).get("gpu", ""):
        training["precision"] = "16-mixed"  # T4 has no bf16 support
    zs = dict(training, run_name="zero-shot-us-checkpoint", output_dir=str(WORK / "out_zs"))
    write_yaml(terratorch_config(base, str(WORK / "subset"), base_init["means"], base_init["stds"], zs), WORK / "zero_shot.yaml")
    write_yaml(terratorch_config(base, str(WORK / "subset"), stats["means"], stats["stds"], training), WORK / "finetune.yaml")
    return {"train_stats": stats, "base_means": base_init["means"], "precision": training.get("precision")}


def checkpoint():
    import yaml
    from huggingface_hub import hf_hub_download

    run = yaml.safe_load((REPO / "configs/ml3_burn_scar.yaml").read_text())
    bm = run["base_model"]
    path = hf_hub_download(bm["id"], bm["checkpoint_file"], revision=bm["revision"], local_dir=str(WORK / "ckpt"))
    return {"path": path, "bytes": Path(path).stat().st_size}


def zero_shot():
    from terratorch.cli_tools import LightningInferenceModel

    ckpt = summary["steps"]["checkpoint"]["result"]["path"]
    lim = LightningInferenceModel.from_config(str(WORK / "zero_shot.yaml"), ckpt)
    return lim.trainer.test(model=lim.model, datamodule=lim.datamodule)


def finetune():
    r = subprocess.run(["terratorch", "fit", "-c", str(WORK / "finetune.yaml")], capture_output=True, text=True, timeout=2400)
    (WORK / "finetune.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"terratorch fit exited {r.returncode}; tail: {r.stderr[-1500:]}")
    ckpts = sorted((WORK / "out/checkpoints").glob("*.ckpt"))
    return {"checkpoints": [c.name for c in ckpts]}


def finetune_test():
    ckpt = WORK / "out/checkpoints/last.ckpt"
    r = subprocess.run(["terratorch", "test", "-c", str(WORK / "finetune.yaml"), "--ckpt_path", str(ckpt)],
                       capture_output=True, text=True, timeout=1200)
    (WORK / "finetune_test.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"terratorch test exited {r.returncode}; tail: {r.stderr[-1500:]}")
    return {"stdout_tail": r.stdout[-2500:]}


def mlflow_metrics():
    import mlflow

    mlflow.set_tracking_uri(f"file:{WORK / 'mlruns'}")
    runs = mlflow.search_runs(search_all_experiments=True)
    cols = [c for c in runs.columns if c.startswith("metrics.") and ("test" in c or "val" in c)]
    return runs[["run_id", "tags.mlflow.runName"] + cols].to_dict(orient="records") if len(runs) else []


step("gpu", gpu)
if step("setup", setup)["ok"] and step("checkpoint", checkpoint)["ok"]:
    step("zero_shot", zero_shot)
    if step("finetune", finetune)["ok"]:
        step("finetune_test", finetune_test)
step("mlflow_metrics", mlflow_metrics)
summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
(WORK / "smoke_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print("summary written", flush=True)

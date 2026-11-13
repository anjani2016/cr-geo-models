# cr-geo-models

Open-weight geospatial models from Centauri Research, trained on public data and published on Hugging Face with their code, configs and evaluation.

**Status:** setup only. No model has been trained or published yet.

## Planned models

| Model | Task | Base model | Status |
|---|---|---|---|
| Canadian burn severity | Segment burned area and severity classes from satellite imagery | Prithvi-EO-2.0-300M BurnScars (Apache-2.0) | Planned |
| Lake chlorophyll-a | Estimate chlorophyll-a from Sentinel-2 reflectance | Gradient-boosting baseline and a mixture density network | Planned |

All outputs are model estimates (Modeled), not measurements. Each model card states its training data, held-out test regions, metrics against a
baseline, limits and licences.

## How it runs
- The `ml/` package holds all the code. It is tested on CPU in CI.
- Training runs on Google Colab from a thin notebook (`notebooks/colab_launcher.ipynb`) that clones a pinned commit and runs one command.
- Data and checkpoints live in cloud storage. Only manifests and configs are in this repo.
- Experiment tracking uses MLflow.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```

## Licence

Code: Apache-2.0 (see `LICENSE`). Each published model and dataset states its own licence and data attributions.

## Citation

See `CITATION.cff`.

© Alpha Centauri Ventures Inc. Centauri Research is a brand of Alpha Centauri Ventures Inc.

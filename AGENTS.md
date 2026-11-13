# cr-geo-models: agent directives

Public repository for open-weight geospatial models (wildfire burn severity, lake chlorophyll-a, and later others) trained on Google Colab
and published on Hugging Face and Zenodo. Owner: Alpha Centauri Ventures Inc. (brand: Centauri Research).
The plan lives in the lake twin repo: `cr_lake_digital_twin/docs/plans/plan_open_lake_models_colab.md` (gates ML-R, ML1, ML3, ML5–ML7 are built here).

This repo is **public**. Everything committed here can be read by anyone.

## Gated work
- The owner signs off every gate.
- **Review (owner, 2026-11-12, verbatim):** "from now on we continue withotu codex. I dont see a need for intense review of this project as it is meant to be an experiment. wE will get a over all review after we finish the work if neeed. from now on act as your own review agent".
  So from gate ML-R on, the implementing agent reviews its own gate. It must still run its own adversarial negative probes and record them in the gate record.
  An overall review may follow when the track is finished.
- Flow: plan with blast radius → owner confirms → code left **uncommitted** → review (active adversarial probes, not only passing tests) →
  change commit → gate-record commit → sign-off commit only on the owner's word.
- Every gate has a record in `docs/audit/gates/` and a row in `docs/audit/gate_index.md`. The record is written from git inspection
  (`git show`, `git diff`); the active gate's commit SHA is `pending` until committed.
- Quote the owner's and reviewer's words verbatim. If no review text was supplied, write "no review text on file". Never invent a sign-off.
- Fast check before stopping for review: `pytest -q` passes with 0 failures in a clean virtualenv built from `requirements-dev.txt`.
- Never push, tag, merge or create releases. The owner does that. A tagged release is archived by Zenodo and is public and permanent.

## Code layout
- All logic lives in the `ml/` package and is tested in `tests/` **without a GPU and without network access**.
- Notebooks in `notebooks/` are thin launchers: check the GPU, clone at a pinned commit, install, read secrets, mount storage, run one
  `python -m ml...` command. No modelling logic in notebook cells.
- Heavy dependencies (PyTorch, TerraTorch, MLflow) go in `requirements-ml.txt`, used on Colab only. Pin them from a resolved install (`pip freeze` on the
  Colab runtime), never guessed. `requirements-dev.txt` holds only what the CPU tests need.
- This repo never imports lake twin code. It reads staged data exports with a checksum manifest and writes model files and metric tables.

## Data and reproducibility
- Data, chips, checkpoints and `mlruns/` live on Google Drive or GCS, never in git. Only manifests (paths, sizes, SHA-256) and configs are committed.
- Every dataset gets an entry in `data_manifest/` with source URL, licence, attribution text, retrieval date and checksums. A dataset whose licence
  is unconfirmed can be used for exploration but not published from.
- Every training or evaluation run records: repo commit SHA, base-model ID and revision, data-manifest checksum, config checksum, random seed,
  GPU type. `ml.run_metadata` builds this record. A run that can't name these is not publishable.
- Splits are by fire, region, lake or year, never by random chip, so neighbouring pixels never leak between train and test.
- Colab compute units are recorded in `docs/colab_ledger.md` as shown by Colab (before and after each session). No estimates in the ledger.

## Claims
- Quote only numbers produced by a command or run, and name the command or MLflow run. Unverified facts get a visible `[verify: …]`.
- Four data tiers, as in the lake twin (AGENTS.md there, Gate SAT-F0a): `Sensor (Live)`, `Observed (Field Sample)`, `Satellite (Uncalibrated proxy)` for spectral indices computed from imagery, and `Modeled (Simulation)` for anything a model or regression derives from them.
- Model outputs are **`Modeled (Simulation)`** (e.g. "satellite-derived burn severity, model estimate"). Never "verified", "validated", "proven", "real-time",
  "live" or "ground truth" without code or the owner backing it.
- Report the baseline next to every result. If a fine-tune doesn't beat its baseline, say so.
- Say "assumed" wherever a distribution or fit is assumed.
- No implied partnerships. Name no person or organisation as a collaborator, client or data provider unless the owner confirms in writing.
  Data publishers are credited only as their licences require. Placeholders use `Example Org`.
- Run a claims check on every model card, dataset card, README and report before publishing.

## Publishing (outward-facing; the owner confirms each one)
- Hugging Face organisation: `centauri-research2019`. The owner creates repos and holds the write token.
- Zenodo: archives tagged GitHub releases; DOIs are linked to the owner's ORCID.
- Code licence: Apache-2.0. Model weights take the most restrictive licence among the base model and the training data. Check before publishing.

## Secrets
- Tokens come from Colab Secrets (`google.colab.userdata`) or environment variables, never from cells, configs or commits.
- Fail closed: a required token that is missing or empty stops the run with an error. Training needs no write token; only publishing does.

## Never commit
`.env*`, tokens or keys, `.claude/settings.local.json`, `.claude/launch.json`, `.DS_Store`, data files, checkpoints, `mlruns/`, notebook outputs
(clear outputs before committing).

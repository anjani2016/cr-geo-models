# Gate ML7: publish `burn-scar-ca` v1

- **Repo / branch:** `cr-geo-models`, `feat/ml7-publish` (from `docs/roadmap` at `3e2f00d`).
- **Commits:** change commit `de269e7`; this record: the next commit on the branch.
- **Status:** files built, self-reviewed and committed (owner, verbatim: "Yes, commit it"). Nothing has been uploaded, tagged, pushed or posted.
- **Owner direction (2026-11-13), verbatim:**
  - Weights licence: "Apache-2.0 plus"
  - Claims sweep: "ok"
  - Safetensors conversion with a tensor-equality check, both hashes recorded: "ok"
  - Release `v0.1.0`, model repo `centauri-research2019/burn-scar-ca`, public once the card has no `[verify]`: "ok"
  - LinkedIn draft: "keep it out of the public repo"
  - "go ahead with my confirmed decisions"
  - ORCID: "https://orcid.org/0009-0003-8789-4851"
  - Account: "anjanid account". `hf auth whoami` showed the CLI logged in as `anjanid` (organisations: `AIFoundry-hackathon` only); the public page of `centauri-research2019` is a user account titled with the owner's name, not an organisation. Model repo is now `anjanid/burn-scar-ca`; card citation URL, AGENTS.md, plan, roadmap and report updated
- **Roles:** Implementer and reviewer: Claude Code (self-review per `AGENTS.md`). Route: Claude Code coded it, because the change is small (documents, one staging module, two scripts, one test file).
- **Plan:** `docs/plans/ml7_publish.md`.

## Changes
| File | Change |
|---|---|
| `hf/burn-scar-ca/README.md` | Model card (Hugging Face metadata, held-out results beside the US baseline, limits, tier, data, training, files, usage, licence, attribution, citation) |
| `hf/burn-scar-ca/NOTICE` | Apache-2.0 notice, the base-model modification statement, NBAC/OGL-Canada and Copernicus attributions |
| `hf/burn-scar-ca/inference_example.py` | Loads the safetensors weights into the TerraTorch task (strict) and writes a burned mask for one chip |
| `hf/burn-scar-ca/norm_stats.json` | Copy of the registered `norm_stats.json` (SHA-256 `e3bcbda7…` on both, `shasum -a 256`) |
| `ml/publish.py`, `tests/test_publish.py` | `stage_upload`: allow-listed copy into an empty folder with `SHA256SUMS`; fails closed |
| `scripts/ml7_stage_hf.py` | The ML7 allow-list (card files, LICENSE, weights with pinned hash, three configs) |
| `scripts/ml7_convert_safetensors.py` | `.pt` → `.safetensors` with `weights_only=True` load and a tensor-by-tensor check (run locally, outside CI) |
| `CITATION.cff` | Version 0.1.0, repository URL, ORCID, keywords; date and DOI added at release |
| `docs/reports/2026-11-13_burn_scar_ca_v1_training_report.md` | Copernicus `[verify]` resolved (below) |
| `docs/roadmap.md`, `docs/audit/gate_index.md`, `docs/plans/ml7_publish.md` | Status, index row, plan |
| Outside the repo | LinkedIn draft at `~/dev/docs/outreach/2026-11_burn-scar-ca-linkedin/linkedin_draft.md` (git-ignored by `~/dev/.gitignore`, checked with `git check-ignore -v`) |

## Licence and attribution checks (sources read in this gate)
| Item | Source | Result |
|---|---|---|
| Base model licence | Hugging Face API for `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` at revision `a3f2c410…` (`curl`): `license: apache-2.0`; the repo has no `LICENSE` or `NOTICE` file (HTTP 404 for both at that revision) | Apache-2.0. Our `NOTICE` states the modification; the citation BibTeX is copied from the upstream card |
| NBAC labels | OGL-Canada 2.0, https://open.canada.ca/en/open-government-licence-canada (fetched): attribution must include the provider's statement and, where possible, a link to the licence; otherwise "Contains information licensed under the Open Government Licence – Canada." | Card and `NOTICE` carry the NBAC citation (read from the NBAC metadata PDF in gate ML1), the OGL statement and the licence link |
| Sentinel-2 imagery | Registry of Open Data on AWS page for `sentinel-2-l2a-cogs` points to the Sentinel Data Legal Notice. The notice PDF (https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice, `pdftotext`) gives "Contains modified Copernicus Sentinel data [Year]" | Year range from the scene IDs of the 1,024 chips in `data/staged/full/manifest.json`, matched to `staging_log.json`: 2017 (54), 2018 (115), 2019 (56), 2020 (20), 2021 (200), 2022 (33), 2023 (282), 2024 (123), 2025 (137), 2026 (4). Wording used: **"Contains modified Copernicus Sentinel data 2017–2026"**. The ML3c record and the registered MLflow README still say `[verify]` / 2017–2025; they are left unchanged (signed off / outside git) and this record supersedes them |
| Earth Search catalogue | Only used to find scenes | Described as "found through the Earth Search STAC API, hosted on AWS"; no provider credit implied |
| Weights licence | Owner decision | Apache-2.0 with `NOTICE` |

## Weights conversion
`scripts/ml7_convert_safetensors.py data/ml3c_r2/model_v1/best_weights_fp16.pt data/ml7/burn_scar_ca_v1_fp16.safetensors --expect-sha256 b471b653…`
(scratch venv: torch 2.14.1, safetensors 0.8.0), output:

| | |
|---|---|
| Tensors / parameters | 355 / 324,411,275; dtypes float16 and int64 (`num_batches_tracked`) |
| Source `.pt` | 648,942,995 bytes, SHA-256 `b471b653b076bac8ec92c80e25d31fe0659996d06176f1c8d9a982ee9c924381` (equals the registered v1) |
| `.safetensors` | 648,861,492 bytes, SHA-256 `ffd8e9c692b20202674431df8805e2e104a4d632be582b193fea054a5120425a` |
| Equality | Keys, dtype, shape and values (`torch.equal`) identical for all 355 tensors |

## Staged upload folder
`scripts/ml7_stage_hf.py <scratch>/hf_upload` staged exactly: `LICENSE`, `NOTICE`, `README.md`, `SHA256SUMS`, `burn_scar_ca_v1_fp16.safetensors`,
`config/` (3 YAML files), `inference_example.py`, `norm_stats.json`. The folder is in the session scratchpad, outside the repo, and must be restaged
from the committed files before upload.

## Checks
- `pytest -q -p no:cacheprovider` in a clean venv built from `requirements-dev.txt`: **263 passed** (17 new in `tests/test_publish.py`).
- **Inference example** (scratch venv, TerraTorch 1.2.14, CPU) on 6 held-out Quebec chips (every 45th test chip): strict load OK; pooled burn-scar IoU **0.828**
  (per chip 0.866, 0.790, 0.841, 0.851, 0.827, 0.789). A usage check only; the reported metrics remain the 274-chip `trainer.test` results from ML3c.
- **Adversarial probes** (all fail closed as intended):
  1. Conversion with a wrong expected hash → refused.
  2. Conversion onto an existing output → refused.
  3. A `.pt` holding a non-tensor (`epoch`) → refused; a `.pt` with a pickle payload calling `os.system` → refused by `weights_only=True`, payload not executed.
  4. Staging into a non-empty folder → refused, exit 1.
  5. `.env.probe` and `.DS_Store` placed beside the card → not staged (allow-list).
  6. One byte flipped in the safetensors file → staging refused, no folder created; the file restored and re-hashed to `ffd8e9c6…`.
  7. Inference example with one weight key removed → strict load refused (`Missing key(s)`).
  8. Grep of card, `NOTICE` and draft for claim words and em dashes → only "not an endorsement" (card) and the draft's own check notes.
- **Claims sweep** (`claims-auditor` agent): no high findings. Applied: cloud-cover rule added to imagery timing; "Registry of Open Data on AWS" changed to
  "found through the Earth Search STAC API, hosted on AWS"; "1 − 0.849" derivation shown; conversion output recorded here. Not changed: the Files table
  (lists the staged upload, confirmed above).

## Hugging Face upload (2026-11-13)
- **Repo:** owner ran `hf repos create burn-scar-ca --type model --private` → `anjanid/burn-scar-ca`, private (`hf models info`: `"private": true`).
- **Upload:** owner ran `hf upload anjanid/burn-scar-ca <scratch>/hf_upload . --repo-type model --commit-message "burn-scar-ca v1 (cr-geo-models <commit>)"`
  → Hub commit `1508dc788a4c06c48bf565dfe2c8242fe6cc0cf4`. The commit message kept the literal placeholder `<commit>`; Hub commit messages can't be edited.
- **Source:** the folder was staged from the working tree with the account-switch edits, before they were committed. Those edits are committed in the commit after `b2852b7`.
- **Check:** files downloaded at that Hub revision (`hf download --exclude "*.safetensors"`) pass `shasum -a 256 -c` for all 8 small files, and `SHA256SUMS` is identical to the staged copy.
  The weights' LFS SHA-256 on the Hub (`HfApi.model_info(files_metadata=True)`) is `ffd8e9c692b20202674431df8805e2e104a4d632be582b193fea054a5120425a`, 648,861,492 bytes, equal to the conversion output.
  The Hub repo holds 11 files: the 10 staged plus its own `.gitattributes`.

## Not done (each needs the owner's word)
Make the Hugging Face repo public (after the owner checks the card on the Hub); merge and push (owner); enable Zenodo for the repo (owner); tag `v0.1.0` and release (owner);
add the DOI to the card, `CITATION.cff` and README; ORCID link (owner); LinkedIn post (owner).

## Reviewer Verdict & Owner Sign-Off Date
- **Reviewer Verdict:** no review text on file (self-review per the owner's instruction)
- **Owner Sign-Off Date:** pending

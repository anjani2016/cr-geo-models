# Gate ML7 plan: publish `burn-scar-ca` v1

Publish the registered `burn-scar-ca` v1 weights on Hugging Face with a model card. Archive a tagged code release on Zenodo for a DOI,
which the owner links to their ORCID. Then draft a LinkedIn post. Source: `docs/reports/2026-11-13_burn_scar_ca_v1_training_report.md`.

## Owner decisions, 2026-11-13 (verbatim)
- Weights licence: "Apache-2.0 plus"
- Claims sweep in this gate: "ok"
- Weights format, convert to `.safetensors` with a tensor-equality check and record both hashes: "ok"
- Release `v0.1.0` and model repo `centauri-research2019/burn-scar-ca`, made public once the card has no `[verify]` left: "ok"
- LinkedIn draft: "keep it out of the public repo"
- Go-ahead: "go ahead with my confirmed decisions"

## Route
Claude Code codes this gate. The changes are small: documents, one staging script, one conversion script and their tests. Per AGENTS.md (owner, 2026-11-12),
the implementing agent reviews its own gate and runs adversarial probes against it.

## Blast radius
| Area | Change |
|---|---|
| `hf/burn-scar-ca/` | New: model card `README.md`, `NOTICE`, `inference_example.py` (the files uploaded to Hugging Face, plus the weights and config files staged at upload time) |
| `scripts/ml7_convert_safetensors.py` | New. Runs locally in a scratch venv with CPU PyTorch; converts the fp16 state_dict and checks every tensor is equal |
| `scripts/ml7_stage_hf.py`, `tests/test_ml7_stage_hf.py` | New. Copies an allow-list of files into an upload folder; fails closed on a hash mismatch, a missing file, a symlink or anything not on the list |
| `CITATION.cff` | Version, release date, repository URL; ORCID left as a placeholder for the owner |
| `docs/audit/gates/gate_ml7_publish.md`, `docs/audit/gate_index.md`, `docs/roadmap.md` | Gate record and status |
| Outside the repo | LinkedIn draft in `~/dev/docs/outreach/2026-11_burn-scar-ca-linkedin/` (git-ignored there; owner: "keep it out of the public repo") |

**Not touched:** training code in `ml/`, training configs, `data_manifest/`, the MLflow server, the lake twin repo, ML3d/ML5/ML6.

## Licence and attribution checks
The sources and wording are recorded in the gate record. Anything not confirmed from the publisher's own page stays `[verify]`, and the model repo stays private until none are left.

## Publishing steps (each needs the owner's word)
1. Change commit, then gate-record commit.
2. Owner creates `centauri-research2019/burn-scar-ca` and logs in with `hf auth login` in their own terminal.
3. Upload from the staged folder only.
4. Owner merges to `main` and pushes.
5. Owner enables this repo in Zenodo's GitHub settings (before the release).
6. Owner tags `v0.1.0` and publishes the GitHub release; Zenodo mints the DOI.
7. DOI added to the card, `CITATION.cff` and the README (second card upload, with the owner's word).
8. Owner links the DOI to ORCID and posts on LinkedIn.

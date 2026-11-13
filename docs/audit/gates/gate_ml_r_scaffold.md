# Gate ML-R: repository scaffold

- **Repo:** `cr-geo-models` (local `~/dev/esg/cr-geo-models`, new, no remote yet)
- **Branch:** `main` (first commit)
- **Commit SHA:** b148205e8df1ab757ed68122538796f260efd56c (change commit, 2026-11-12)
- **Status:** Codex review CHANGES REQUESTED (6 items), all addressed; self-reviewed after that per the owner; signed off
- **Plan:** lake twin `docs/plans/plan_open_lake_models_colab.md` revision 3, §3 ML-R
- **Owner approval (2026-11-12), verbatim:** "Approve, start ML-R"
- **Roles:** Implementer: Claude Code. Reviewer: Codex for the first round (owner, 2026-11-12, verbatim: "Codex"); Claude Code self-review of the fixes, on the owner's instruction (below). Owner sign-off: 2026-11-12.
- **Route and reason:** Claude Code codes it because it is small (scaffold, four helper modules, tests) and the Colab units expire about 2026-11-21.
  The reviewer must not be Claude Code.

## Scope
Set up the public repo so later gates (ML1 data staging, ML3 burn-severity fine-tune, ML5 chlorophyll-a) have a reviewed base. No model, no data, no
training code, no Colab units used.

## Files (all new, untracked)
- `AGENTS.md`, `CLAUDE.md`: gate rules, claims, publishing, secrets, never-commit list
- `README.md`, `LICENSE` (Apache-2.0 text from apache.org, SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`), `CITATION.cff`
- `.gitignore`, `pyproject.toml`, `requirements-dev.txt` (pinned from a clean 3.12 venv)
- `.github/workflows/tests.yml`: CPU `pytest` on push and pull request, `contents: read`
- `ml/config.py`: YAML config loader; fails closed on missing keys, unpinned base-model revision (`main`, `latest`, empty), bad seed
- `ml/manifest.py`: data manifests with SHA-256; rejects absolute paths, `..`, backslashes and symlinks resolving outside the data root; verifies size and hash
- `ml/tokens.py`: reads secrets from Colab Secrets, then the environment; refuses missing or blank values
- `ml/run_metadata.py`: run record (commit SHA, config and manifest checksums, seed, base model and revision, GPU type); refuses uncommitted tracked changes unless `allow_dirty`
- `notebooks/colab_launcher.ipynb`: thin launcher, 9 cells, no outputs
- `configs/README.md`, `data_manifest/README.md`, `docs/colab_ledger.md` (empty table), `docs/audit/gate_index.md`, this record
- `tests/`: `test_config.py`, `test_manifest.py`, `test_tokens.py`, `test_run_metadata.py`, `test_repo_hygiene.py`, `conftest.py`

## Checks run (2026-11-12, clean venv from `requirements-dev.txt`, Python 3.12)
- First build: `.venv/bin/pytest -q`: **71 passed**.
- After the review fixes: `.venv/bin/pytest -q`: **109 passed**.
- Negative probes run by hand, all detected (the tests failed as intended, then the probe was reverted):
  - a token-shaped string planted in `ml/_probe.py`: `test_no_token_shaped_strings_in_committable_files` failed
  - an output added to a launcher cell: `test_notebooks_have_no_outputs` failed
- After the reverts: `pytest -q` 71 passed again.

## Known limits
- The launcher's training cell is a placeholder until ML3. `requirements-ml.txt` is created in ML3 from a resolved Colab install (plan changed; review item 5).
- The CI workflow has not run: there is no GitHub remote yet.
- `CITATION.cff` lacks ORCID, repository URL, version and release date (owner adds them before the first release).

## Reviewer notes

**Codex review (relayed by the owner, 2026-11-12), verbatim:**

> CHANGES REQUESTED
>
> 1. [ml/config.py (line 57)](/Users/anjanid/dev/esg/cr-geo-models/ml/config.py:57) accepts mutable branch-qualified revisions such as `refs/heads/main`, `origin/main`, and `refs/remotes/origin/main`. These violate the promised pinned revision behavior. Reject branch/ref forms and add negative tests.
> 2. [ml/config.py (line 42)](/Users/anjanid/dev/esg/cr-geo-models/ml/config.py:42) accepts absolute and escaping manifest paths, while [configs/README.md (line 14)](/Users/anjanid/dev/esg/cr-geo-models/configs/README.md:14) requires a repo-relative path under `data_manifest/`. Probes accepted `/tmp/x.json` and `../x.json`; [ml/run_metadata.py (line 50)](/Users/anjanid/dev/esg/cr-geo-models/ml/run_metadata.py:50) expressly follows absolute paths. Close this known item now.
> 3. [ml/manifest.py (line 42)](/Users/anjanid/dev/esg/cr-geo-models/ml/manifest.py:42) allows `build_entries()` to follow a symlink outside the data root and checksum the external file. This contradicts the gate record’s symlink-escape claim. Apply the resolved-root containment check during manifest creation too.
> 4. [ml/manifest.py (line 75)](/Users/anjanid/dev/esg/cr-geo-models/ml/manifest.py:75) does not validate `bytes` or `sha256` types/formats. A floating-point size of `3.0` and Boolean size `true` both passed full verification for matching files. Require a non-Boolean, non-negative integer and a 64-character hexadecimal SHA-256.
> 5. [plan_open_lake_models_colab.md (line 97)](/Users/anjanid/dev/esg/cr_lake_digital_twin/docs/plans/plan_open_lake_models_colab.md:97) includes pinned `requirements-ml.txt` in ML-R, but [gate_ml_r_scaffold.md (line 39)](/Users/anjanid/dev/esg/cr-geo-models/docs/audit/gates/gate_ml_r_scaffold.md:39) defers it. Because dependencies must not be guessed, either supply a resolved Colab lock or record explicit owner approval for the scope change.
> 6. [plan_open_lake_models_colab.md (line 106)](/Users/anjanid/dev/esg/cr_lake_digital_twin/docs/plans/plan_open_lake_models_colab.md:106) assigns repository registration in `CATALOG.md` to ML-R, but no `cr-geo-models` entry exists and the gate inventory omits this work. Add it or record an owner-approved deferral.
>
> Verified:
>
> * `71 passed` independently on Python 3.12.
> * Non-string and blank secrets failed closed.
> * Staged modifications and staged renames were rejected by run metadata.
> * License checksum matches `cfc7749b…3d30`.
> * Notebook has 9 cells and no outputs.
> * No model, data, or training implementation is present.
> * `AGENTS.md` contains the required Modeled wording, quoted-number rule, partnership restriction, and owner confirmation for every publication.
> * No apparent token, key, dataset, checkpoint, or notebook output is among the 27 public candidates.
> * Working tree remained unchanged.

**Owner's instruction with the review, verbatim:** "review and incorporate. from now on we continue withotu codex. I dont see a need for intense review of this project as it is meant to be an experiment. wE will get a over all review after we finish the work if neeed. from now on act as your own review agent"

**How each item was handled (Claude Code):**
1. `ml/config.py`: `base_model.revision` must be a full 40-character lowercase commit hash. Branches, refs, tags, short or padded hashes and `HEAD~1` are refused. New negative tests.
2. `ml/config.py` `check_manifest_path`: `data_manifest` must be exactly `data_manifest/<name>.json` in its plain spelling (no absolute path, `..`, subfolder, backslash, null byte, `./`, `//` or trailing `/`).
   `ml/run_metadata.py` no longer follows absolute paths, and refuses a manifest that resolves (e.g. through a symlink) outside `data_manifest/`. New tests.
3. `ml/manifest.py`: one `_contained` check, used by both `build_entries` and `verify`, refuses files resolving outside the data root. New test.
4. `ml/manifest.py` `_check_entry`, used by both `load_manifest` and `verify`: `bytes` must be a non-negative int (not bool or float), and `sha256` must be 64 lowercase hex characters. Null bytes in paths are refused. New tests.
5. `requirements-ml.txt` moved to ML3 in the plan, because it can only be pinned from a resolved Colab install. The basis recorded in the plan is the owner's "review and incorporate".
6. `~/dev/CATALOG.md`: `cr-geo-models` row added, "Last verified" set to 2026-11-12. The plan now marks registration as done in ML-R.

**Self-review probes after the fixes (Claude Code, run by hand with the venv's Python):**
- Revisions `refs/tags/v1`, the hash plus a newline, `HEAD~1` and the hash plus `^` were all refused.
- Manifest-path spellings `./data_manifest/x.json`, `data_manifest//x.json` and `data_manifest/x.json/` were at first **accepted**. They now fail, with tests.
- A null byte in a manifest path raised `ValueError` instead of `ManifestError`. It now raises `ManifestError`, with a test.
- A sha256 with a trailing space and a `bytes` of 2**70 were both refused.

## Sign-off
Owner committed on 2026-11-12. Owner's words, verbatim: "Sign off, start ML0".

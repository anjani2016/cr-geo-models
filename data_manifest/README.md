# Data manifests

One JSON file per staged dataset. The data itself lives on Google Drive or GCS. `ml.manifest.verify` checks the staged files before every run.

Required fields: `dataset`, `source_url`, `licence`, `attribution`, `retrieved_on` (YYYY-MM-DD), and `files`. `files` is a list of
`{path, bytes, sha256}`, where each path is relative to the data root, with no `..` and not absolute.

A dataset whose licence is not confirmed may be used for exploration but nothing trained on it is published.

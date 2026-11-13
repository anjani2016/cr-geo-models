# Configs

One YAML file per training or evaluation run. `ml.config.load_config` checks every file before a run starts.

Required keys:

| Key | Meaning |
|---|---|
| `name` | Short run name, e.g. `burn-severity-ca-v0` |
| `task` | e.g. `segmentation`, `regression` |
| `seed` | Non-negative integer |
| `base_model.id` | Hugging Face model ID |
| `base_model.revision` | The full 40-character lowercase commit hash of the Hugging Face model repo. Branches, refs (`refs/heads/main`, `origin/main`), tags and short hashes are refused because they can move or be ambiguous |
| `data_manifest` | `data_manifest/<name>.json`: repo-relative, directly in that folder. Absolute paths, `..` and subfolders are refused |

Gate-specific keys (bands, splits, class edges, training settings) are added by the gate that needs them, with their source noted beside the value.

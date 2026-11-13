"""Gate ML7: stage the `burn-scar-ca` v1 upload folder for Hugging Face.

    python scripts/ml7_stage_hf.py OUT_DIR

Reads the converted weights from `data/ml7/` (git-ignored) and the card, notice, example, configs and norm stats from the repo.
Uploads send OUT_DIR only. Nothing here touches the network.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.publish import StageItem, StagingError, stage_upload  # noqa: E402

WEIGHTS_SHA256 = "ffd8e9c692b20202674431df8805e2e104a4d632be582b193fea054a5120425a"  # from scripts/ml7_convert_safetensors.py
CARD = REPO / "hf" / "burn-scar-ca"

ITEMS = [
    StageItem("README.md", CARD / "README.md"),
    StageItem("NOTICE", CARD / "NOTICE"),
    StageItem("LICENSE", REPO / "LICENSE"),
    StageItem("inference_example.py", CARD / "inference_example.py"),
    StageItem("norm_stats.json", CARD / "norm_stats.json"),
    StageItem("burn_scar_ca_v1_fp16.safetensors", REPO / "data" / "ml7" / "burn_scar_ca_v1_fp16.safetensors", WEIGHTS_SHA256),
    StageItem("config/ml3_burn_scar.yaml", REPO / "configs" / "ml3_burn_scar.yaml"),
    StageItem("config/ml1_fire_staging.yaml", REPO / "configs" / "ml1_fire_staging.yaml"),
    StageItem("config/prithvi_eo_v2_300m_burnscars_config.yaml", REPO / "configs" / "vendor" / "prithvi_eo_v2_300m_burnscars_config.yaml"),
]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        sums = stage_upload(ITEMS, Path(argv[0]))
    except StagingError as err:
        print(f"staging refused: {err}", file=sys.stderr)
        return 1
    print(json.dumps(sums, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

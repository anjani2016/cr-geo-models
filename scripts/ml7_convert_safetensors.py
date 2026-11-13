"""Gate ML7: convert the registered fp16 state_dict (`.pt`, pickle) to `.safetensors` and check every tensor is unchanged.

Runs locally, outside CI, in a venv with CPU PyTorch and safetensors (not in requirements-dev.txt):

    python scripts/ml7_convert_safetensors.py SRC.pt DST.safetensors --expect-sha256 <64 hex>

Fails closed: the source hash must match, the file is loaded with `weights_only=True` (no arbitrary pickle code), every value must be a
tensor, and the written file is read back and compared tensor by tensor (keys, dtype, shape, exact values). Prints both SHA-256 values as JSON.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    import torch
    from safetensors.torch import load_file, save_file

    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--expect-sha256", required=True)
    a = p.parse_args(argv)

    if a.dst.exists():
        sys.exit(f"refusing to overwrite {a.dst}")
    src_hash = sha256(a.src)
    if src_hash != a.expect_sha256.lower():
        sys.exit(f"source SHA-256 {src_hash} does not match the expected {a.expect_sha256}")

    state = torch.load(a.src, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        sys.exit("source is not a non-empty state_dict")
    bad = [k for k, v in state.items() if not isinstance(k, str) or not isinstance(v, torch.Tensor)]
    if bad:
        sys.exit(f"non-tensor entries: {bad[:5]}")

    # clone() breaks shared storage, which safetensors refuses.
    save_file({k: v.detach().clone().contiguous() for k, v in state.items()}, str(a.dst),
              metadata={"format": "pt", "source_sha256": src_hash})

    back = load_file(str(a.dst))
    if set(back) != set(state):
        sys.exit("key sets differ after conversion")
    for k, v in state.items():
        w = back[k]
        if w.dtype != v.dtype or w.shape != v.shape or not torch.equal(w, v):
            sys.exit(f"tensor {k} differs after conversion")

    dtypes = sorted({str(v.dtype) for v in state.values()})
    print(json.dumps({"tensors": len(state), "parameters": int(sum(v.numel() for v in state.values())), "dtypes": dtypes,
                      "src": str(a.src), "src_sha256": src_hash, "src_bytes": a.src.stat().st_size,
                      "dst": str(a.dst), "dst_sha256": sha256(a.dst), "dst_bytes": a.dst.stat().st_size,
                      "all_tensors_equal": True}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

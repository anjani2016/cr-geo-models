"""Stage the files for a Hugging Face model upload (gate ML7).

`stage_upload` copies an explicit allow-list of files into an empty folder and writes `SHA256SUMS`. It fails closed: a missing source,
a symlink, a hash mismatch, a destination that escapes the folder, a non-empty destination or a token-shaped string in a text file
stops staging with an error. The upload then sends that folder and nothing else.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from ml.manifest import SHA256_HEX, file_sha256

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".cff", ""}
TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


class StagingError(ValueError):
    """The upload folder could not be staged safely."""


@dataclass(frozen=True)
class StageItem:
    dest: str                      # path inside the upload folder, e.g. "config/ml3_burn_scar.yaml"
    source: Path
    sha256: Optional[str] = None   # required for binary files; checked before and after copying


def _check_dest(dest: str) -> PurePosixPath:
    if not isinstance(dest, str) or not dest or "\\" in dest or "\x00" in dest:
        raise StagingError(f"bad destination: {dest!r}")
    pure = PurePosixPath(dest)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) > 2 or any(p.startswith(".") for p in pure.parts):
        raise StagingError(f"destination not allowed: {dest!r}")
    return pure


def _scan_text(path: Path, dest: str) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            raise StagingError(f"token-shaped string in {dest}")


def stage_upload(items: Iterable[StageItem], out_dir: Path) -> dict:
    """Copy `items` into `out_dir` (which must be missing or empty). Returns {dest: sha256}."""
    items = list(items)
    if not items:
        raise StagingError("nothing to stage")
    dests = [str(_check_dest(i.dest)) for i in items]
    if len(set(dests)) != len(dests):
        raise StagingError("duplicate destination")
    if "SHA256SUMS" in dests:
        raise StagingError("SHA256SUMS is written by staging")

    for item in items:
        src = Path(item.source)
        if src.is_symlink() or not src.is_file():
            raise StagingError(f"source is missing, not a regular file, or a symlink: {src}")
        suffix = PurePosixPath(item.dest).suffix.lower()
        if suffix not in TEXT_SUFFIXES and item.sha256 is None:
            raise StagingError(f"binary file needs an expected SHA-256: {item.dest}")
        if item.sha256 is not None:
            if not SHA256_HEX.fullmatch(item.sha256):
                raise StagingError(f"expected SHA-256 is not 64 lowercase hex characters: {item.dest}")
            if file_sha256(src) != item.sha256:
                raise StagingError(f"SHA-256 mismatch for {item.dest} ({src})")
        if suffix in TEXT_SUFFIXES:
            _scan_text(src, item.dest)

    out_dir = Path(out_dir)
    if out_dir.is_symlink():
        raise StagingError(f"upload folder is a symlink: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise StagingError(f"upload folder is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sums = {}
    for item in items:
        dst = out_dir / item.dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.source, dst)
        digest = file_sha256(dst)
        if item.sha256 is not None and digest != item.sha256:
            raise StagingError(f"SHA-256 changed while copying {item.dest}")
        sums[item.dest] = digest
    (out_dir / "SHA256SUMS").write_text("".join(f"{h}  {d}\n" for d, h in sorted(sums.items())), encoding="utf-8")
    return sums

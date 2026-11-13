"""Checksum manifests for staged data.

Data lives on Drive or GCS, not in git. The manifest (committed) lists every file with its size and SHA-256, plus
the source, licence and attribution. Training verifies the staged files against it before using them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

REQUIRED_FIELDS = ("dataset", "source_url", "licence", "attribution", "retrieved_on", "files")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1024 * 1024


class ManifestError(ValueError):
    """The manifest is malformed or the staged files don't match it."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(rel: str) -> PurePosixPath:
    """Reject absolute paths and any '..' so a manifest can't point outside its data root."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        raise ManifestError(f"bad file path in manifest: {rel!r}")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        raise ManifestError(f"file path escapes the data root: {rel!r}")
    return pure


def _contained(root: Path, pure: PurePosixPath) -> Path:
    """The file's path under root, refusing anything (e.g. a symlink) that resolves outside root."""
    target = root.joinpath(*pure.parts)
    resolved = target.resolve()
    if root not in resolved.parents:
        raise ManifestError(f"{pure}: resolves outside the data root")
    return target


def _check_entry(entry: Any) -> str:
    """Exactly path, bytes (non-negative int, not bool) and sha256 (64 lowercase hex). Returns the path."""
    if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
        raise ManifestError("each file needs exactly path, bytes, sha256")
    rel = str(_safe_relative(entry["path"]))
    size = entry["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ManifestError(f"{rel}: 'bytes' must be a non-negative integer, not {size!r}")
    if not isinstance(entry["sha256"], str) or not SHA256_HEX.fullmatch(entry["sha256"]):
        raise ManifestError(f"{rel}: 'sha256' must be 64 lowercase hex characters")
    return rel


def build_entries(root: str | Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    """Size and SHA-256 for each file under root, sorted by path."""
    root = Path(root).resolve()
    entries = []
    for rel in sorted(set(relative_paths)):
        pure = _safe_relative(rel)
        target = _contained(root, pure)
        if not target.is_file():
            raise ManifestError(f"not a file: {rel}")
        entries.append({"path": str(pure), "bytes": target.stat().st_size, "sha256": file_sha256(target)})
    return entries


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """Checksum of the manifest content (canonical JSON), as logged with each run."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"{path}: top level must be an object")
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ManifestError(f"{path}: missing fields {missing}")
    for field in REQUIRED_FIELDS[:-1]:
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ManifestError(f"{path}: '{field}' must be a non-empty string")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ManifestError(f"{path}: 'files' must be a non-empty list")
    seen = set()
    for entry in files:
        try:
            rel = _check_entry(entry)
        except ManifestError as exc:
            raise ManifestError(f"{path}: {exc}") from exc
        if rel in seen:
            raise ManifestError(f"{path}: duplicate file {rel}")
        seen.add(rel)
    return manifest


def verify(manifest: dict[str, Any], root: str | Path) -> None:
    """Fail closed unless every listed file exists under root with the recorded size and SHA-256."""
    root = Path(root).resolve()
    problems = []
    for entry in manifest["files"]:
        pure = PurePosixPath(_check_entry(entry))
        try:
            target = _contained(root, pure)
        except ManifestError as exc:
            problems.append(str(exc))
            continue
        if not target.is_file():
            problems.append(f"{entry['path']}: missing")
            continue
        if target.stat().st_size != entry["bytes"]:
            problems.append(f"{entry['path']}: size {target.stat().st_size} != {entry['bytes']}")
            continue
        if file_sha256(target) != entry["sha256"]:
            problems.append(f"{entry['path']}: SHA-256 mismatch")
    if problems:
        raise ManifestError("staged data does not match the manifest:\n" + "\n".join(problems))

"""MLflow tracking for training runs on the hosted server (gate ML3b).

Credentials never live in code, configs or commits. MLflow reads them from `~/.mlflow/credentials`
(`[mlflow]` section, `mlflow_tracking_username` / `mlflow_tracking_password`) or from the
MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD environment variables. `check_credentials` fails closed
before any GPU work if neither is present. Nothing here reads or prints the password itself.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any, Mapping, Optional

MAX_TAG_CHARS = 5000  # MLflow tag values are limited in length; longer values are cut and marked


class TrackingError(RuntimeError):
    """Tracking is not set up safely; the run must not start."""


def run_tags(record: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None, prefix: str = "cr.") -> dict[str, str]:
    """Flatten a run record (ml.run_metadata) into MLflow tags: string keys with a prefix, string values, None dropped."""
    out: dict[str, str] = {}
    for source in (record, extra or {}):
        for key, value in source.items():
            if value is None:
                continue
            text = str(value)
            if len(text) > MAX_TAG_CHARS:
                text = text[: MAX_TAG_CHARS - 12] + "…[truncated]"
            out[f"{prefix}{key}"] = text
    return out


def check_credentials(home: Optional[Path] = None, environ: Optional[Mapping[str, str]] = None) -> str:
    """Say where MLflow credentials will come from, without reading the password out. Fails closed if none.

    Returns "env" or "credentials-file". A file must be readable only by its owner (mode 600 or stricter).
    """
    environ = os.environ if environ is None else environ
    if environ.get("MLFLOW_TRACKING_USERNAME", "").strip() and environ.get("MLFLOW_TRACKING_PASSWORD", "").strip():
        return "env"
    path = (home or Path.home()) / ".mlflow" / "credentials"
    if not path.is_file():
        raise TrackingError(f"no MLflow credentials: set MLFLOW_TRACKING_USERNAME/PASSWORD or create {path}")
    if path.stat().st_mode & 0o077:
        raise TrackingError(f"{path} is readable by others; run chmod 600 on it")
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except configparser.Error as exc:
        raise TrackingError(f"{path} is not a valid credentials file ({type(exc).__name__})") from None
    section = parser["mlflow"] if parser.has_section("mlflow") else {}
    missing = [k for k in ("mlflow_tracking_username", "mlflow_tracking_password") if not str(section.get(k, "")).strip()]
    if missing:
        raise TrackingError(f"{path} [mlflow] section lacks {missing}")
    return "credentials-file"


def export_credentials_to_env(home: Optional[Path] = None, environ: Optional[dict] = None) -> str:
    """Copy the credentials file into MLFLOW_TRACKING_USERNAME/PASSWORD of this process (inherited by child processes such
    as `terratorch fit`), so tracking works whichever way the installed MLflow looks for credentials.
    Returns only the source name; the values are never returned or printed."""
    environ = os.environ if environ is None else environ
    source = check_credentials(home=home, environ=environ)
    if source == "credentials-file":
        parser = configparser.ConfigParser()
        parser.read((home or Path.home()) / ".mlflow" / "credentials")
        environ["MLFLOW_TRACKING_USERNAME"] = parser["mlflow"]["mlflow_tracking_username"].strip()
        environ["MLFLOW_TRACKING_PASSWORD"] = parser["mlflow"]["mlflow_tracking_password"].strip()
    return source


def require_https(tracking_uri: str) -> str:
    """Remote tracking must use HTTPS, so credentials are never sent in clear text. Local sqlite/file stores are allowed."""
    if tracking_uri.startswith("https://") or tracking_uri.startswith(("sqlite:", "file:")):
        return tracking_uri
    raise TrackingError(f"tracking URI must be https:// (or a local sqlite/file store), not {tracking_uri!r}")


def preflight(tracking_uri: str, experiment_name: str) -> dict[str, Any]:
    """On the training machine, before any GPU work: credentials present, HTTPS, server reachable and accepting them,
    experiment available. Returns facts for the run record. Imports mlflow only here."""
    require_https(tracking_uri)
    if tracking_uri.startswith("https://"):
        source = export_credentials_to_env()
    else:
        source = "none (local store)"
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    try:
        client.search_experiments(max_results=1)
    except Exception as exc:  # 401/403/connection errors: stop before spending GPU time
        raise TrackingError(f"MLflow server refused or unreachable: {type(exc).__name__}: {str(exc)[:300]}") from None
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
    return {"tracking_uri": tracking_uri, "credentials_source": source, "experiment_id": experiment_id,
            "mlflow_client": mlflow.__version__}


def artifact_upload_probe(tracking_uri: str, experiment_name: str, size_mb: int, workdir: Path) -> dict[str, Any]:
    """Upload a throwaway artifact of size_mb to a run tagged as a probe, then delete the run.
    Tells us before training whether the server takes large artifacts (e.g. a 1.3 GB checkpoint)."""
    import mlflow

    mlflow.set_tracking_uri(require_https(tracking_uri))
    mlflow.set_experiment(experiment_name)
    path = Path(workdir) / f"upload_probe_{size_mb}mb.bin"
    with open(path, "wb") as handle:
        handle.write(os.urandom(1024 * 1024) * size_mb)
    with mlflow.start_run(run_name=f"artifact-upload-probe-{size_mb}mb", tags={"cr.probe": "true"}) as run:
        try:
            mlflow.log_artifact(str(path), artifact_path="probe")
            ok, error = True, None
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {str(exc)[:300]}"
    mlflow.tracking.MlflowClient().delete_run(run.info.run_id)
    path.unlink(missing_ok=True)
    return {"size_mb": size_mb, "ok": ok, "error": error}

import os

import pytest

from ml.tracking import TrackingError, check_credentials, require_https, run_tags

FAKE = "[mlflow]\nmlflow_tracking_username = trainer\nmlflow_tracking_password = not-a-real-secret\n"


def _creds(home, text=FAKE, mode=0o600):
    d = home / ".mlflow"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "credentials"
    p.write_text(text)
    os.chmod(p, mode)
    return p


def test_run_tags_flatten_prefix_and_drop_none():
    tags = run_tags({"commit_sha": "abc", "seed": 42, "dirty": False, "gpu_type": None}, {"units_start": 599.16})
    assert tags == {"cr.commit_sha": "abc", "cr.seed": "42", "cr.dirty": "False", "cr.units_start": "599.16"}


def test_run_tags_truncate_long_values():
    tags = run_tags({"x": "a" * 9000})
    assert len(tags["cr.x"]) == 5000 and tags["cr.x"].endswith("[truncated]")


def test_credentials_from_env_take_priority(tmp_path):
    env = {"MLFLOW_TRACKING_USERNAME": "u", "MLFLOW_TRACKING_PASSWORD": "p"}
    assert check_credentials(home=tmp_path, environ=env) == "env"


@pytest.mark.parametrize("env", [{}, {"MLFLOW_TRACKING_USERNAME": "u"}, {"MLFLOW_TRACKING_USERNAME": "u", "MLFLOW_TRACKING_PASSWORD": "  "}])
def test_missing_credentials_fail_closed(tmp_path, env):
    with pytest.raises(TrackingError, match="no MLflow credentials"):
        check_credentials(home=tmp_path, environ=env)


def test_credentials_file_ok_with_mode_600(tmp_path):
    _creds(tmp_path)
    assert check_credentials(home=tmp_path, environ={}) == "credentials-file"


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_credentials_file_readable_by_others_is_refused(tmp_path, mode):
    _creds(tmp_path, mode=mode)
    with pytest.raises(TrackingError, match="chmod 600"):
        check_credentials(home=tmp_path, environ={})


@pytest.mark.parametrize(
    "text",
    ["[mlflow]\nmlflow_tracking_username = trainer\n", "[other]\nmlflow_tracking_username = a\nmlflow_tracking_password = b\n",
     "[mlflow]\nmlflow_tracking_username = trainer\nmlflow_tracking_password =   \n"],
)
def test_incomplete_credentials_file_fails(tmp_path, text):
    _creds(tmp_path, text=text)
    with pytest.raises(TrackingError, match="lacks"):
        check_credentials(home=tmp_path, environ={})


def test_malformed_credentials_file_fails_without_echoing_contents(tmp_path):
    _creds(tmp_path, text="mlflow_tracking_password = not-a-real-secret\n[[[")
    with pytest.raises(TrackingError) as err:
        check_credentials(home=tmp_path, environ={})
    assert "not-a-real-secret" not in str(err.value)


@pytest.mark.parametrize("uri", ["https://mlflow.example.org", "sqlite:////content/work/mlflow.db", "file:/tmp/mlruns"])
def test_https_or_local_store_allowed(uri):
    assert require_https(uri) == uri


@pytest.mark.parametrize("uri", ["http://mlflow.example.org", "mlflow.example.org", "", "databricks"])
def test_cleartext_or_unknown_uri_refused(uri):
    with pytest.raises(TrackingError):
        require_https(uri)


def test_export_credentials_sets_env_without_returning_values(tmp_path):
    from ml.tracking import export_credentials_to_env

    _creds(tmp_path)
    env = {}
    assert export_credentials_to_env(home=tmp_path, environ=env) == "credentials-file"
    assert env == {"MLFLOW_TRACKING_USERNAME": "trainer", "MLFLOW_TRACKING_PASSWORD": "not-a-real-secret"}


def test_export_credentials_fails_closed_on_open_permissions(tmp_path):
    from ml.tracking import export_credentials_to_env

    _creds(tmp_path, mode=0o644)
    env = {}
    with pytest.raises(TrackingError):
        export_credentials_to_env(home=tmp_path, environ=env)
    assert env == {}

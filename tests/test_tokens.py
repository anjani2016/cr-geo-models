import pytest

from ml.tokens import MissingSecretError, require_secret


def test_returns_value_from_getter():
    assert require_secret("HF_TOKEN", getter={"HF_TOKEN": "abc"}.get) == "abc"


@pytest.mark.parametrize("value", [None, "", "   ", "\n", 0, b"bytes"])
def test_missing_empty_or_non_string_fails_closed(value):
    with pytest.raises(MissingSecretError):
        require_secret("HF_TOKEN", getter=lambda _name: value)


@pytest.mark.parametrize("name", ["", "  "])
def test_blank_name_fails(name):
    with pytest.raises(MissingSecretError):
        require_secret(name, getter=lambda _n: "abc")


def test_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("CR_GEO_TEST_TOKEN", "from-env")
    assert require_secret("CR_GEO_TEST_TOKEN") == "from-env"


def test_absent_everywhere_fails(monkeypatch):
    monkeypatch.delenv("CR_GEO_TEST_TOKEN", raising=False)
    with pytest.raises(MissingSecretError, match="refusing"):
        require_secret("CR_GEO_TEST_TOKEN")


def test_error_message_never_contains_a_value():
    with pytest.raises(MissingSecretError) as err:
        require_secret("HF_TOKEN", getter=lambda _n: "   ")
    assert "HF_TOKEN" in str(err.value)

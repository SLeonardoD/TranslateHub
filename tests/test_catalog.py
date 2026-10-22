import copy
import json

import pytest

from app.config import Settings, load_catalog, parse_catalog
from app.core.exceptions import CatalogError
from tests.conftest import RAW_CATALOG


def raw_with(model_id, **changes):
    raw = copy.deepcopy(RAW_CATALOG)
    entry = raw["models"].setdefault(model_id, copy.deepcopy(raw["models"]["opus-mt-en-es"]))
    for key, value in changes.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return raw


def test_real_catalog_is_valid():
    catalog = parse_catalog(copy.deepcopy(RAW_CATALOG))
    assert set(catalog) == {"opus-mt-en-es", "opus-mt-es-en", "nllb-200-distilled-600M"}
    assert catalog["nllb-200-distilled-600M"]["supported_pairs"][0] == ["en", "es"]
    assert "supported_pairs" not in catalog["opus-mt-en-es"]  # sin claves None


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"hf_name": None}, "hf_name"),
        ({"type": None}, "type"),
        ({"description": None}, "description"),
        ({"type": "otra-cosa"}, "type"),
        ({"tier": "ultra"}, "tier"),
        ({"source_lang": None}, "source_lang"),
        ({"defualt_max_length": 10}, "defualt_max_length"),  # errata
        ({"default_max_length": 0}, "default_max_length"),
    ],
)
def test_invalid_entries_are_rejected_with_a_clear_message(changes, fragment):
    with pytest.raises(CatalogError, match=fragment):
        parse_catalog(raw_with("opus-mt-bad", **changes))


def test_multilingual_requires_code_for_every_language():
    raw = copy.deepcopy(RAW_CATALOG)
    del raw["models"]["nllb-200-distilled-600M"]["language_code_map"]["de"]
    with pytest.raises(CatalogError, match="no incluye los idiomas: de"):
        parse_catalog(raw)


def test_all_errors_are_reported_together():
    raw = raw_with("a-bad", tier="x")
    raw["models"]["b-bad"] = {"hf_name": "no-slash"}
    with pytest.raises(CatalogError) as exc:
        parse_catalog(raw)
    assert "a-bad" in str(exc.value)
    assert "b-bad" in str(exc.value)


@pytest.mark.parametrize("raw", [[], {}, {"models": {}}, {"models": []}])
def test_wrong_shape(raw):
    with pytest.raises(CatalogError, match="models"):
        parse_catalog(raw)


def test_load_catalog_file_errors(tmp_path):
    with pytest.raises(CatalogError, match="No se encontró"):
        load_catalog(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(CatalogError, match="JSON válido"):
        load_catalog(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(RAW_CATALOG), encoding="utf-8")
    assert "opus-mt-en-es" in load_catalog(good)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_PORT", "9001")
    monkeypatch.setenv("TRANSLATOR_CORS_ORIGINS", "http://a.com, http://b.com")
    monkeypatch.setenv("TRANSLATOR_DEVICE", "CPU")
    s = Settings.from_env()
    assert s.port == 9001
    assert s.cors_origins == ("http://a.com", "http://b.com")
    assert s.device == "cpu"


@pytest.mark.parametrize(
    ("name", "value"),
    [("PORT", "abc"), ("PORT", "0"), ("DEVICE", "tpu"), ("CHUNK_TARGET_TOKENS", "4")],
)
def test_settings_reject_bad_values(monkeypatch, name, value):
    monkeypatch.setenv(f"TRANSLATOR_{name}", value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


# ---------- .env y lotes ilimitados ----------


def test_parse_dotenv_formats():
    from app.config import parse_dotenv

    text = (
        "\ufeff# comentario\nTRANSLATOR_PORT=9966  # puerto\n"
        "export TRANSLATOR_HOST=\"127.0.0.1\"\nTRANSLATOR_API_KEY='a#b'\nmalo\n1X=2\n\n"
    )
    assert parse_dotenv(text) == {
        "TRANSLATOR_PORT": "9966",
        "TRANSLATOR_HOST": "127.0.0.1",
        "TRANSLATOR_API_KEY": "a#b",
    }


def test_dotenv_does_not_override_real_environment(tmp_path, monkeypatch):
    from app.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text("TRANSLATOR_PORT=9966\nTRANSLATOR_MAX_BATCH_ITEMS=0\n", encoding="utf-8")
    monkeypatch.setenv("TRANSLATOR_PORT", "7000")
    monkeypatch.delenv("TRANSLATOR_MAX_BATCH_ITEMS", raising=False)
    assert load_dotenv(env) == env
    s = Settings.from_env()
    assert s.port == 7000  # el entorno real manda
    assert s.max_batch_items == 0  # viene del .env
    monkeypatch.delenv("TRANSLATOR_MAX_BATCH_ITEMS")


def test_load_dotenv_missing_file(tmp_path):
    from app.config import load_dotenv

    assert load_dotenv(tmp_path / "nope.env") is None


def test_max_batch_items_zero_is_valid_negative_is_not(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_MAX_BATCH_ITEMS", "0")
    assert Settings.from_env().max_batch_items == 0
    monkeypatch.setenv("TRANSLATOR_MAX_BATCH_ITEMS", "-1")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_known_transformers_warning_is_filtered():
    import logging

    from app.logging_setup import _KnownWarningFilter

    f = _KnownWarningFilter()

    def rec(msg):
        return logging.LogRecord("x", logging.WARNING, "f", 1, msg, None, None)

    assert not f.filter(
        rec(
            "Both `max_new_tokens` (=512) and `max_length`(=512) ... `max_new_tokens` will take precedence."
        )
    )
    assert f.filter(rec("otra alerta cualquiera"))

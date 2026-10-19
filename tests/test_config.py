"""Tests de app.config: carga y validación del catálogo."""
import json

import pytest


def test_load_catalog_returns_models_dict(catalog_file, sample_catalog):
    from app.config import load_catalog
    catalog = load_catalog()
    assert isinstance(catalog, dict)
    assert set(catalog.keys()) == set(sample_catalog.keys())
    assert catalog["opus-mt-en-es"]["source_lang"] == "en"


def test_load_catalog_missing_file_raises(tmp_path, monkeypatch):
    missing = tmp_path / "no-existe.json"
    monkeypatch.setattr("app.config.MODEL_INFO_PATH", missing)
    from app.config import load_catalog
    with pytest.raises(FileNotFoundError):
        load_catalog()


def test_load_catalog_invalid_json_raises(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    monkeypatch.setattr("app.config.MODEL_INFO_PATH", bad)
    from app.config import load_catalog
    with pytest.raises(json.JSONDecodeError):
        load_catalog()
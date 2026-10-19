"""Fixtures compartidas. Todo lo pesado se mockea; todo el I/O va a tmp_path."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Fuerza CPU en todos los tests para no depender del hardware real."""
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ImportError:
        pass


@pytest.fixture
def sample_catalog() -> dict:
    """Catálogo mínimo con los 3 casos representativos:
    par (fast) x2 y multilingüe (quality) x1.
    """
    return {
        "opus-mt-en-es": {
            "hf_name": "Helsinki-NLP/opus-mt-en-es",
            "family": "opus-mt",
            "type": "pair",
            "source_lang": "en",
            "target_lang": "es",
            "tier": "fast",
            "description": "OPUS-MT inglés→español",
            "default_max_length": 512,
        },
        "opus-mt-es-en": {
            "hf_name": "Helsinki-NLP/opus-mt-es-en",
            "family": "opus-mt",
            "type": "pair",
            "source_lang": "es",
            "target_lang": "en",
            "tier": "fast",
            "description": "OPUS-MT español→inglés",
            "default_max_length": 512,
        },
        "nllb-200-distilled-600M": {
            "hf_name": "facebook/nllb-200-distilled-600M",
            "family": "nllb",
            "type": "multilingual",
            "tier": "quality",
            "supported_pairs": [["en", "es"], ["es", "en"], ["en", "fr"]],
            "language_code_map": {
                "en": "eng_Latn",
                "es": "spa_Latn",
                "fr": "fra_Latn",
            },
            "description": "NLLB",
            "default_max_length": 512,
        },
    }


@pytest.fixture
def catalog_file(tmp_path, monkeypatch, sample_catalog):
    """Escribe el catálogo a un fichero temporal y redirige config.MODEL_INFO_PATH."""
    path = tmp_path / "model_info.json"
    path.write_text(json.dumps({"models": sample_catalog}), encoding="utf-8")
    monkeypatch.setattr("app.config.MODEL_INFO_PATH", path)
    return path


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Carpeta temporal para modelos y redirige model_manager.MODELS_BASE_DIR."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr("app.core.model_manager.MODELS_BASE_DIR", d)
    return d


@pytest.fixture
def manager(catalog_file, models_dir):
    """Instancia limpia de ModelManager apuntando a paths temporales."""
    from app.core.model_manager import ModelManager
    return ModelManager()


@pytest.fixture
def mock_transformers(monkeypatch):
    """Mockea AutoTokenizer y AutoModelForSeq2SeqLM.

    - from_pretrained devuelve una instancia MagicMock.
    - save_pretrained escribe un config.json dummy para que is_model_downloaded()
      vea el modelo como descargado (no escribimos pesos reales).
    """
    def _fake_save_pretrained(path_str):
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}", encoding="utf-8")

    mock_tok = MagicMock()
    mock_tok.save_pretrained.side_effect = _fake_save_pretrained

    mock_mdl = MagicMock()
    mock_mdl.save_pretrained.side_effect = _fake_save_pretrained

    mock_tok_class = MagicMock()
    mock_tok_class.from_pretrained.return_value = mock_tok

    mock_mdl_class = MagicMock()
    mock_mdl_class.from_pretrained.return_value = mock_mdl

    monkeypatch.setattr("app.core.model_manager.AutoTokenizer", mock_tok_class)
    monkeypatch.setattr("app.core.model_manager.AutoModelForSeq2SeqLM", mock_mdl_class)

    return {
        "tok_class": mock_tok_class,
        "tok": mock_tok,
        "mdl_class": mock_mdl_class,
        "mdl": mock_mdl,
    }
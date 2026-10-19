"""Tests de app.services.translation. Mockea ModelManager.translate para no tocar transformers."""
import pytest

from app.core.exceptions import ModelNotAvailableError, ModelLoadError


@pytest.fixture
def svc(monkeypatch, catalog_file, models_dir):
    """Devuelve el módulo service con una instancia limpia de ModelManager."""
    import app.services.translation as tm
    from app.core.model_manager import ModelManager
    mgr = ModelManager()
    monkeypatch.setattr(tm, "model_manager", mgr)
    return tm, mgr


# ------------------------------------------------------------------
# translate_text
# ------------------------------------------------------------------

def test_translate_text_con_model_id(svc, monkeypatch):
    tm, mgr = svc
    monkeypatch.setattr(mgr, "translate", lambda **kw: "Hola")

    result = tm.translate_text("Hello", model_id="opus-mt-en-es")
    assert result == {
        "translated_text": "Hola",
        "model_id": "opus-mt-en-es",
        "source_lang": "en",
        "target_lang": "es",
    }


def test_translate_text_resuelve_por_par(svc, monkeypatch):
    tm, mgr = svc
    captured = {}

    def fake_translate(**kwargs):
        captured.update(kwargs)
        return "x"

    monkeypatch.setattr(mgr, "translate", fake_translate)
    tm.translate_text("Hi", source_lang="en", target_lang="es")
    # Debe haber resuelto a OPUS-MT (fast)
    assert captured["model_id"] == "opus-mt-en-es"


def test_translate_text_resuelve_por_tier_quality(svc, monkeypatch):
    tm, mgr = svc
    captured = {}

    def fake_translate(**kwargs):
        captured.update(kwargs)
        return "x"

    monkeypatch.setattr(mgr, "translate", fake_translate)
    tm.translate_text("Hi", source_lang="en", target_lang="es", tier="quality")
    assert captured["model_id"] == "nllb-200-distilled-600M"


def test_translate_text_multilingue_requiere_langs(svc, monkeypatch):
    tm, _ = svc
    with pytest.raises(ModelLoadError, match="multilingüe"):
        tm.translate_text("Hi", model_id="nllb-200-distilled-600M")


def test_translate_text_multilingue_par_no_soportado(svc, monkeypatch):
    tm, _ = svc
    with pytest.raises(ModelNotAvailableError, match="no soporta el par"):
        tm.translate_text(
            "Hi", model_id="nllb-200-distilled-600M",
            source_lang="en", target_lang="ja",
        )


def test_translate_text_pair_mismatch(svc, monkeypatch):
    tm, _ = svc
    with pytest.raises(ModelNotAvailableError, match="es en→es"):
        tm.translate_text(
            "Hi", model_id="opus-mt-en-es",
            source_lang="es", target_lang="en",
        )


def test_translate_text_sin_seleccion(svc, monkeypatch):
    tm, _ = svc
    # Sin model_id ni par: resolve_model_id recibe None,None → falla
    with pytest.raises(ModelNotAvailableError):
        tm.translate_text("Hi")


# ------------------------------------------------------------------
# get_models_status / get_pairs
# ------------------------------------------------------------------

def test_get_models_status_vacio(svc):
    tm, _ = svc
    result = tm.get_models_status()
    assert len(result) == 3
    assert all(m["downloaded"] is False for m in result)
    assert all(m["loaded"] is False for m in result)


def test_get_models_status_con_descarga(svc, models_dir, monkeypatch):
    tm, mgr = svc
    # Descarga simulada
    path = mgr._get_local_path("opus-mt-en-es")
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")

    result = {m["model_id"]: m for m in tm.get_models_status()}
    assert result["opus-mt-en-es"]["downloaded"] is True
    assert result["opus-mt-es-en"]["downloaded"] is False


def test_get_pairs(svc):
    tm, _ = svc
    pairs = {(p["source_lang"], p["target_lang"]) for p in tm.get_pairs()}
    assert ("en", "es") in pairs
    assert ("en", "fr") in pairs
    assert ("es", "en") in pairs
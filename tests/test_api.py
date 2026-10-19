"""Tests de la capa HTTP. Mockea las funciones de servicio para no tocar transformers."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(catalog_file, models_dir):
    """TestClient con el catálogo temporal ya aplicado antes de importar app.main."""
    from app.main import app
    return TestClient(app)


# ------------------------------------------------------------------
# Endpoints informativos
# ------------------------------------------------------------------

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_endpoint(client, monkeypatch):
    monkeypatch.setattr("app.main.get_models_status", lambda: [])
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json() == []


def test_models_endpoint_con_datos(client, monkeypatch):
    fake = [{
        "model_id": "opus-mt-en-es",
        "hf_name": "Helsinki-NLP/opus-mt-en-es",
        "family": "opus-mt",
        "type": "pair",
        "tier": "fast",
        "description": "d",
        "source_lang": "en",
        "target_lang": "es",
        "supported_pairs": None,
        "downloaded": True,
        "loaded": False,
    }]
    monkeypatch.setattr("app.main.get_models_status", lambda: fake)
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json()[0]["model_id"] == "opus-mt-en-es"


def test_pairs_endpoint(client, monkeypatch):
    fake = [{"source_lang": "en", "target_lang": "es", "models": ["opus-mt-en-es"]}]
    monkeypatch.setattr("app.main.get_pairs", lambda: fake)
    r = client.get("/pairs")
    assert r.status_code == 200
    assert r.json()[0]["source_lang"] == "en"


# ------------------------------------------------------------------
# POST /translate
# ------------------------------------------------------------------

def test_translate_exito(client, monkeypatch):
    def fake(**kw):
        return {
            "translated_text": "Hola",
            "model_id": "opus-mt-en-es",
            "source_lang": "en",
            "target_lang": "es",
        }
    monkeypatch.setattr("app.main.translate_text", fake)

    r = client.post("/translate", json={
        "text": "Hello", "source_lang": "en", "target_lang": "es",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["translated_text"] == "Hola"
    assert body["model_id"] == "opus-mt-en-es"


def test_translate_422_sin_seleccion(client):
    # Ni model_id ni par → ValidationError → 422
    r = client.post("/translate", json={"text": "Hello"})
    assert r.status_code == 422


def test_translate_422_sin_texto(client):
    r = client.post("/translate", json={"source_lang": "en", "target_lang": "es"})
    assert r.status_code == 422


def test_translate_404_modelo_no_disponible(client, monkeypatch):
    from app.core.exceptions import ModelNotAvailableError

    def fake(**kw):
        raise ModelNotAvailableError("no existe")

    monkeypatch.setattr("app.main.translate_text", fake)
    r = client.post("/translate", json={"text": "Hi", "model_id": "nope"})
    assert r.status_code == 404
    assert "no existe" in r.json()["detail"]


def test_translate_500_error_de_carga(client, monkeypatch):
    from app.core.exceptions import ModelLoadError

    def fake(**kw):
        raise ModelLoadError("fallo de carga")

    monkeypatch.setattr("app.main.translate_text", fake)
    r = client.post("/translate", json={"text": "Hi", "model_id": "opus-mt-en-es"})
    assert r.status_code == 500


def test_translate_pasa_parametros_al_servicio(client, monkeypatch):
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return {
            "translated_text": "x",
            "model_id": "nllb-200-distilled-600M",
            "source_lang": "en",
            "target_lang": "fr",
        }

    monkeypatch.setattr("app.main.translate_text", fake)
    r = client.post("/translate", json={
        "text": "Hi", "model_id": "nllb-200-distilled-600M",
        "source_lang": "en", "target_lang": "fr", "max_length": 64,
    })
    assert r.status_code == 200
    assert captured["model_id"] == "nllb-200-distilled-600M"
    assert captured["source_lang"] == "en"
    assert captured["target_lang"] == "fr"
    assert captured["max_length"] == 64
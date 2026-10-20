import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.core.exceptions import InferenceError, ModelLoadError
from app.services import translation


@pytest.fixture
def client(manager, monkeypatch):
    monkeypatch.setattr(translation, "model_manager", manager)
    with TestClient(main.app) as test_client:
        yield test_client


def post_translate(client, **body):
    return client.post("/translate", json=body)


# ---------- Endpoints básicos ----------


def test_root(client):
    body = client.get("/").json()
    assert body["status"] == "ok"
    assert body["queue"] == {"pending": 0, "max": main.settings.max_pending_requests}


def test_models_and_pairs(client, manager):
    manager.translate_batch("opus-mt-en-es", ["x"])
    models = {m["model_id"]: m for m in client.get("/models").json()}
    assert models["opus-mt-en-es"]["loaded"] is True
    assert models["opus-mt-es-en"]["downloaded"] is True
    assert {
        "source_lang": "en",
        "target_lang": "de",
        "models": ["nllb-200-distilled-600M"],
    } in client.get("/pairs").json()


def test_translate_by_pair(client):
    r = post_translate(client, text="Hello world", source_lang="en", target_lang="es")
    assert r.status_code == 200
    assert r.json() == {
        "translated_text": "HELLO WORLD",
        "chunks": 1,
        "truncated": False,
        "warnings": [],
        "model_id": "opus-mt-en-es",
        "source_lang": "en",
        "target_lang": "es",
    }


def test_translate_batch(client):
    r = client.post(
        "/translate/batch", json={"texts": ["uno", "dos"], "source_lang": "es", "target_lang": "en"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == "opus-mt-es-en"
    assert [x["translated_text"] for x in body["results"]] == ["UNO", "DOS"]


def test_truncation_is_visible_in_the_response(client):
    r = post_translate(client, text="this will TRUNC", model_id="opus-mt-en-es", max_length=5)
    assert r.status_code == 200
    assert r.json()["truncated"] is True
    assert "max_length=5" in r.json()["warnings"][0]


# ---------- Códigos de error ----------


def test_404_unknown_model(client):
    r = post_translate(client, text="x", model_id="nope")
    assert r.status_code == 404


def test_404_not_downloaded_gives_the_command(client, manager):
    (manager._get_local_path("opus-mt-es-en") / ".download_complete").unlink()
    r = post_translate(client, text="x", source_lang="es", target_lang="en")
    assert r.status_code == 404
    assert "model_manager download opus-mt-es-en" in r.json()["detail"]


def test_404_target_mismatch_on_pair_model(client):
    r = post_translate(
        client, text="x", model_id="opus-mt-en-es", source_lang="en", target_lang="fr"
    )
    assert r.status_code == 404


def test_422_multilingual_without_languages(client):
    """Antes: 500. Es un error del cliente."""
    r = post_translate(client, text="x", model_id="nllb-200-distilled-600M")
    assert r.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"text": "", "source_lang": "en", "target_lang": "es"},
        {"text": "   ", "source_lang": "en", "target_lang": "es"},
        {"text": "x"},
        {"text": "x", "source_lang": "en"},
        {"text": "x", "source_lang": "en", "target_lang": "es", "tier": "ultra"},
        {"text": "x", "source_lang": "en", "target_lang": "es", "max_length": 0},
        {"text": "x", "source_lang": "en", "target_lang": "es", "max_length": 99999},
        {"text": "x", "source_lang": "e1!", "target_lang": "es"},
        {
            "text": "a" * (main.settings.max_text_chars + 1),
            "source_lang": "en",
            "target_lang": "es",
        },
    ],
)
def test_422_invalid_bodies(client, body):
    assert post_translate(client, **body).status_code == 422


def test_422_batch_limits(client):
    base = {"source_lang": "en", "target_lang": "es"}
    assert client.post("/translate/batch", json={**base, "texts": []}).status_code == 422
    assert client.post("/translate/batch", json={**base, "texts": ["ok", " "]}).status_code == 422
    too_many = ["x"] * (main.settings.max_batch_items + 1)
    assert client.post("/translate/batch", json={**base, "texts": too_many}).status_code == 422


def test_language_codes_are_normalized_in_requests(client):
    r = post_translate(client, text="hola", source_lang=" ES-es ", target_lang="EN")
    assert r.status_code == 200
    assert r.json()["model_id"] == "opus-mt-es-en"


def test_500_inference_error(client, manager, monkeypatch):
    def boom(*args, **kwargs):
        raise InferenceError("sin memoria")

    monkeypatch.setattr(manager, "translate_batch", boom)
    r = post_translate(client, text="x", source_lang="en", target_lang="es")
    assert r.status_code == 500
    assert "sin memoria" in r.json()["detail"]


def test_500_load_error(client, manager, monkeypatch):
    def boom(*args, **kwargs):
        raise ModelLoadError("pesos corruptos")

    monkeypatch.setattr(manager, "_load_pretrained", boom)
    r = post_translate(client, text="x", source_lang="en", target_lang="es")
    assert r.status_code == 500


def test_unexpected_errors_are_500_with_json(manager, monkeypatch):
    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(manager, "translate_batch", lambda *a, **k: 1 / 0)
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        r = post_translate(test_client, text="x", source_lang="en", target_lang="es")
    assert r.status_code == 500
    assert "Error interno" in r.json()["detail"]


# ---------- Cola: el servidor sigue respondiendo ----------


def test_other_endpoints_answer_while_a_translation_is_running(client, manager):
    manager.delay = 0.8
    thread = threading.Thread(
        target=post_translate,
        args=(client,),
        kwargs={"text": "slow", "source_lang": "en", "target_lang": "es"},
    )
    thread.start()
    time.sleep(0.2)  # la traducción ya está en curso

    started = time.perf_counter()
    assert client.get("/models").status_code == 200
    assert client.get("/pairs").status_code == 200
    assert client.get("/").json()["queue"]["pending"] == 1
    elapsed = time.perf_counter() - started

    thread.join()
    assert elapsed < 0.5, f"los endpoints esperaron a la traducción ({elapsed:.2f}s)"


def test_translations_are_served_one_by_one_in_order(client, manager):
    manager.delay = 0.15
    finished: list[str] = []

    def call(name):
        post_translate(client, text=name, source_lang="en", target_lang="es")
        finished.append(name)

    threads = []
    for name in ("first", "second", "third"):
        t = threading.Thread(target=call, args=(name,))
        t.start()
        threads.append(t)
        time.sleep(0.05)
    for t in threads:
        t.join()

    assert finished == ["first", "second", "third"]
    assert [b[0] for b in manager.batches] == ["first", "second", "third"]


def test_503_when_the_queue_is_full(manager, monkeypatch):
    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(main, "settings", replace(main.settings, max_pending_requests=1))
    manager.delay = 0.6
    with TestClient(main.app) as test_client:
        thread = threading.Thread(
            target=post_translate,
            args=(test_client,),
            kwargs={"text": "slow", "source_lang": "en", "target_lang": "es"},
        )
        thread.start()
        time.sleep(0.2)
        r = post_translate(test_client, text="extra", source_lang="en", target_lang="es")
        thread.join()
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"


# ---------- API key y CORS ----------


def test_api_key_required_when_configured(manager, monkeypatch):
    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(main, "settings", replace(main.settings, api_key="s3cret"))
    with TestClient(main.app) as c:
        assert c.get("/").status_code == 200  # el health-check queda abierto
        assert c.get("/models").status_code == 401
        assert c.get("/pairs", headers={"X-API-Key": "mal"}).status_code == 401
        assert post_translate(c, text="x", source_lang="en", target_lang="es").status_code == 401
        ok = {"X-API-Key": "s3cret"}
        assert c.get("/models", headers=ok).status_code == 200
        r = c.post(
            "/translate", json={"text": "x", "source_lang": "en", "target_lang": "es"}, headers=ok
        )
        assert r.status_code == 200


def test_no_api_key_means_open(client):
    assert client.get("/models").status_code == 200

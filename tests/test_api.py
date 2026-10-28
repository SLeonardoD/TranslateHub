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
    assert body["queue"]["pending"] == 0
    assert body["queue"]["max"] == main.settings.max_pending_requests
    assert body["queue"]["pending_chars"] == 0
    assert "estimated_wait_s" in body["queue"]


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
    r = post_translate(client, text="x", model_id="opus-mt-es-en")
    assert r.status_code == 404
    assert "model_manager download opus-mt-es-en" in r.json()["detail"]


def test_models_and_pairs_only_show_complete_local_models(client, manager):
    (manager._get_local_path("opus-mt-es-en") / ".download_complete").unlink()  # incompleto
    manager._get_local_path("nllb-200-distilled-600M").joinpath(
        "model.safetensors"
    ).unlink()  # sin pesos
    assert [m["model_id"] for m in client.get("/models").json()] == ["opus-mt-en-es"]
    pairs = client.get("/pairs").json()
    assert pairs == [{"source_lang": "en", "target_lang": "es", "models": ["opus-mt-en-es"]}]


def test_pair_resolution_only_considers_local_models(client, manager):
    """es→en: el modelo rápido no está completo, pero NLLB sí => se usa NLLB, no hay error."""
    (manager._get_local_path("opus-mt-es-en") / ".download_complete").unlink()
    r = post_translate(client, text="hola", source_lang="es", target_lang="en")
    assert r.status_code == 200
    assert r.json()["model_id"] == "nllb-200-distilled-600M"


def test_404_when_no_local_model_covers_the_pair(client, manager):
    for mid in ("opus-mt-es-en", "nllb-200-distilled-600M"):
        (manager._get_local_path(mid) / ".download_complete").unlink()
    r = post_translate(client, text="hola", source_lang="es", target_lang="en")
    assert r.status_code == 404
    assert "Ningún modelo local" in r.json()["detail"]
    assert "download opus-mt-es-en" in r.json()["detail"]


def test_api_never_downloads_anything(client, manager, monkeypatch):
    """En runtime la API no puede descargar: ni siquiera importa/llama a huggingface_hub."""
    import sys
    import types

    def forbidden(*args, **kwargs):
        raise AssertionError("¡la API intentó descargar un modelo!")

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = forbidden
    hub.snapshot_download = forbidden
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    (manager._get_local_path("opus-mt-en-es") / ".download_complete").unlink()
    assert post_translate(client, text="x", model_id="opus-mt-en-es").status_code == 404
    assert post_translate(client, text="x", source_lang="en", target_lang="fr").status_code == 200
    client.get("/models")
    client.get("/pairs")


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
    body = r.json()
    assert "sin memoria" not in r.text  # el detalle interno no sale del servidor
    assert body["request_id"] == r.headers["x-request-id"]


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
    assert "division" not in r.text and "ZeroDivisionError" not in r.text
    assert r.json()["request_id"] == r.headers["x-request-id"]


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


def test_batch_without_limits_when_max_batch_items_is_zero(monkeypatch):
    """max_batch_items=0 → ni límite de textos ni de caracteres totales (el de cada texto sigue)."""
    import importlib

    import app.config as config
    import app.models.schemas as schemas

    monkeypatch.setattr(
        config, "settings", replace(config.settings, max_batch_items=0, max_batch_chars=10)
    )
    importlib.reload(schemas)  # el límite de la lista se evalúa al importar
    try:
        req = schemas.BatchTranslationRequest(
            texts=["palabra larga"] * 500, source_lang="es", target_lang="en"
        )
        assert len(req.texts) == 500
        with pytest.raises(ValueError):  # el límite por texto sigue activo
            schemas.BatchTranslationRequest(
                texts=["x" * (config.settings.max_text_chars + 1)],
                source_lang="es",
                target_lang="en",
            )
    finally:
        monkeypatch.undo()
        importlib.reload(schemas)


# ---------- Trabajos abandonados por el cliente ----------


class _FakeRequest:
    """Request mínimo: ``is_disconnected`` pasa a True a los ``after`` segundos."""

    def __init__(self, queue, after: float | None):
        import types

        self.app = types.SimpleNamespace(state=types.SimpleNamespace(queue=queue))
        self._t0 = time.monotonic()
        self._after = after

    async def is_disconnected(self) -> bool:
        return self._after is not None and time.monotonic() - self._t0 >= self._after


def test_abandoned_request_is_skipped_when_its_turn_comes():
    import asyncio

    from app.core.concurrency import RequestQueue, check_cancelled
    from app.core.exceptions import ClientDisconnectedError

    ran: list[str] = []

    def work(name, cancel):
        check_cancelled(cancel)
        ran.append(name)
        time.sleep(0.3)
        return name

    async def scenario():
        queue = RequestQueue(max_pending=5)
        alive = asyncio.create_task(
            main._run_job(_FakeRequest(queue, None), {}, work, "vivo", chars=10)
        )
        await asyncio.sleep(0.02)
        gone = asyncio.create_task(
            main._run_job(_FakeRequest(queue, 0.0), {}, work, "ausente", chars=10)
        )
        results = await asyncio.gather(alive, gone, return_exceptions=True)
        return results, queue.pending

    (alive_result, gone_result), pending = asyncio.run(scenario())
    assert alive_result == "vivo"
    assert isinstance(gone_result, ClientDisconnectedError)
    assert ran == ["vivo"]  # el ausente no llegó a ejecutarse
    assert pending == 0


def test_disconnect_mid_translation_stops_early():
    import asyncio

    from app.core.concurrency import RequestQueue, check_cancelled
    from app.core.exceptions import ClientDisconnectedError

    def work(cancel):
        for _ in range(100):  # 5 s si nadie la corta
            check_cancelled(cancel)
            time.sleep(0.05)

    async def scenario():
        queue = RequestQueue(max_pending=5)
        started = time.monotonic()
        with pytest.raises(ClientDisconnectedError):
            await main._run_job(_FakeRequest(queue, 0.3), {}, work, chars=10)
        return time.monotonic() - started

    assert asyncio.run(scenario()) < 2


def test_503_reports_estimate_and_retry_after(manager, monkeypatch):
    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(main, "settings", replace(main.settings, max_pending_requests=1))
    manager.delay = 0.4
    body = {"text": "hola mundo", "source_lang": "en", "target_lang": "es"}
    with TestClient(main.app) as c:
        assert post_translate(c, **body).status_code == 200  # primera medición
        thread = threading.Thread(target=post_translate, args=(c,), kwargs=body)
        thread.start()
        time.sleep(0.15)
        r = post_translate(c, **body)
        queue_info = c.get("/").json()["queue"]
        thread.join()
    assert r.status_code == 503
    assert r.headers["retry-after"].isdigit() and int(r.headers["retry-after"]) >= 5
    assert r.json()["estimated_wait_s"] >= 0
    assert queue_info["pending"] == 1 and queue_info["pending_chars"] == len(body["text"])

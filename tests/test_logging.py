"""El log de la API: archivo en vivo, quién/para qué/cuándo, excepciones y seguridad."""

import logging
import logging.config
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.logging_setup import build_log_config, clean
from app.services import translation


@pytest.fixture
def log_file(tmp_path):
    """Configura el logging como lo hace run.py y devuelve el archivo; limpia al terminar."""
    path = tmp_path / "translator.log"
    logging.config.dictConfig(build_log_config("INFO", path, max_mb=1, backups=1))
    yield path
    for name in ("translator", "uvicorn"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.propagate = True


@pytest.fixture
def client(manager, monkeypatch, log_file):
    monkeypatch.setattr(translation, "model_manager", manager)
    with TestClient(main.app) as test_client:
        yield test_client


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_log_file_is_created_and_written_live(client, log_file):
    client.get("/")
    assert log_file.exists()
    assert any("op=health" in line for line in lines(log_file))  # sin cerrar ni vaciar nada


def test_translate_logs_when_who_and_what(client, log_file):
    r = client.post(
        "/translate",
        json={"text": "Hello world", "source_lang": "en", "target_lang": "es"},
        headers={"X-Client-Name": "app-facturacion", "User-Agent": "pruebas/1.0"},
    )
    line = next(line for line in lines(log_file) if "POST /translate" in line)
    assert line[:4].isdigit() and line[4] == "-"  # fecha y hora al inicio
    assert " INFO " in line
    assert f"[{r.headers['x-request-id']}]" in line  # el id de la respuesta está en el log
    for expected in (
        "ip=testclient",  # quién
        'client="app-facturacion"',
        'ua="pruebas/1.0"',
        "status=200",  # resultado
        "op=translate",  # para qué
        "model=opus-mt-en-es",
        "pair=en>es",
        "chars=11",
        "chunks=1",
        "truncated=False",
        "work_ms=",
    ):
        assert expected in line, expected
    assert "Hello world" not in "\n".join(lines(log_file))  # el texto no se registra por defecto


def test_text_preview_is_opt_in_and_limited(client, log_file, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, log_text_preview=5))
    client.post(
        "/translate", json={"text": "Hello world", "source_lang": "en", "target_lang": "es"}
    )
    content = "\n".join(lines(log_file))
    assert 'text="Hello"' in content
    assert "Hello world" not in content


def test_batch_is_described(client, log_file):
    client.post(
        "/translate/batch",
        json={"texts": ["uno", "dos"], "source_lang": "es", "target_lang": "en"},
    )
    line = next(line for line in lines(log_file) if "POST /translate/batch" in line)
    assert "op=translate_batch" in line and "texts=2" in line and "chars=6" in line


def test_client_errors_are_warnings_with_the_reason(client, log_file):
    client.post("/translate", json={"text": "x", "model_id": "nope"})
    client.post("/translate", json={"text": "", "source_lang": "en", "target_lang": "es"})
    log = lines(log_file)
    not_found = next(line for line in log if "status=404" in line)
    assert " WARNING " in not_found and 'error="Modelo' in not_found and "model=nope" in not_found
    invalid = next(line for line in log if "status=422" in line)
    assert " WARNING " in invalid and "error=" in invalid and "text" in invalid


def test_unexpected_exception_is_logged_with_traceback(manager, monkeypatch, log_file):
    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(manager, "translate_batch", lambda *a, **k: 1 / 0)
    with TestClient(main.app, raise_server_exceptions=False) as c:
        r = c.post("/translate", json={"text": "x", "source_lang": "en", "target_lang": "es"})
    log = "\n".join(lines(log_file))
    assert r.status_code == 500
    assert "Traceback (most recent call last)" in log
    assert "ZeroDivisionError" in log
    request_line = next(line for line in lines(log_file) if "status=500" in line)
    assert " ERROR " in request_line and f"[{r.headers['x-request-id']}]" in request_line


def test_model_load_failure_keeps_the_traceback(client, manager, log_file, monkeypatch):
    def boom(path):
        raise OSError("pesos corruptos")

    monkeypatch.setattr(manager, "_load_pretrained", boom)
    client.post("/translate", json={"text": "x", "source_lang": "en", "target_lang": "es"})
    log = "\n".join(lines(log_file))
    assert "Fallo al cargar" in log and "pesos corruptos" in log and "Traceback" in log


def test_manager_lines_carry_the_request_id(client, log_file):
    r = client.post("/translate", json={"text": "x", "source_lang": "en", "target_lang": "es"})
    rid = r.headers["x-request-id"]
    load_line = next(
        line for line in lines(log_file) if "translator.manager" in line and "Cargando" in line
    )
    assert f"[{rid}]" in load_line  # las líneas del gestor (hilo de trabajo) también llevan el id


def test_auth_failures_are_logged(manager, monkeypatch, log_file):
    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(main, "settings", replace(main.settings, api_key="s3cret"))
    with TestClient(main.app) as c:
        c.get("/models")
        c.get("/models", headers={"X-API-Key": "mala"})
    log = lines(log_file)
    assert any("status=401" in ln and "API key ausente" in ln for ln in log)
    assert any("status=401" in ln and "API key incorrecta" in ln for ln in log)
    assert "s3cret" not in "\n".join(log) and "mala" not in "\n".join(log)


def test_queue_full_is_logged_as_warning(manager, monkeypatch, log_file):
    import threading
    import time

    monkeypatch.setattr(translation, "model_manager", manager)
    monkeypatch.setattr(main, "settings", replace(main.settings, max_pending_requests=1))
    manager.delay = 0.5
    body = {"text": "x", "source_lang": "en", "target_lang": "es"}
    with TestClient(main.app) as c:
        t = threading.Thread(target=lambda: c.post("/translate", json=body))
        t.start()
        time.sleep(0.2)
        c.post("/translate", json=body)
        t.join()
    full = next(ln for ln in lines(log_file) if "status=503" in ln)
    assert " WARNING " in full and "Cola llena" in full


def test_slow_translation_reports_queue_time(manager, monkeypatch, log_file):
    import threading
    import time

    monkeypatch.setattr(translation, "model_manager", manager)
    manager.delay = 0.3
    body = {"text": "x", "source_lang": "en", "target_lang": "es"}
    with TestClient(main.app) as c:
        t = threading.Thread(target=lambda: c.post("/translate", json=body))
        t.start()
        time.sleep(0.1)
        c.post("/translate", json=body)  # espera a la primera
        t.join()
    waits = [
        int(ln.split("queued_ms=")[1].split()[0]) for ln in lines(log_file) if "queued_ms=" in ln
    ]
    assert max(waits) >= 100


def test_client_data_cannot_forge_log_lines(client, log_file):
    evil = "x\x00\x1b[31m fake" + '" status=200 op=forged'
    assert "\n" not in clean("a\nb\rc")
    assert '"' not in clean(evil)
    client.get("/", headers={"X-Client-Name": evil})
    assert not any(ln.startswith("forged") for ln in lines(log_file))
    assert all(ln[:2] == "20" or ln.startswith(("Traceback", " ")) for ln in lines(log_file))


def test_log_can_be_disabled():
    from app.config import Settings

    assert Settings(log_file="off").resolve_log_file() is None
    assert Settings(log_file="logs/x.log").resolve_log_file().name == "x.log"
    assert build_log_config("INFO", None)["loggers"]["translator"]["handlers"] == ["app_console"]

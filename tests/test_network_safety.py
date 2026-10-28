"""Modo local/LAN, guarda de Host, tope de cuerpo (413) y errores sin detalles internos."""

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

import app.main as main
import run
from app.config import Settings, resolve_host
from app.http_guards import BodySizeLimitMiddleware, LocalHostGuard
from app.logging_setup import ConsoleFilter, ConsoleFormatter, build_log_config

# ---------- Ajustes de red ----------


def test_default_is_local_only(monkeypatch):
    assert Settings().host == "127.0.0.1" and Settings().local_only
    monkeypatch.delenv("TRANSLATOR_HOST", raising=False)
    assert Settings.from_env().local_only


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("local", "127.0.0.1"),
        ("LOCALHOST", "127.0.0.1"),
        ("lan", "0.0.0.0"),
        ("Red", "0.0.0.0"),
        ("192.168.1.50", "192.168.1.50"),
        ("0.0.0.0", "0.0.0.0"),
    ],
)
def test_host_aliases(raw, expected):
    assert resolve_host(raw) == expected


def test_lan_modes_are_not_local_only(monkeypatch):
    for value in ("lan", "192.168.1.50", "0.0.0.0"):
        monkeypatch.setenv("TRANSLATOR_HOST", value)
        assert not Settings.from_env().local_only


def test_new_settings_validation(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_MAX_BODY_MB", "0")
    with pytest.raises(ValueError, match="MAX_BODY_MB"):
        Settings.from_env()
    monkeypatch.setenv("TRANSLATOR_MAX_BODY_MB", "20")
    monkeypatch.setenv("TRANSLATOR_CONSOLE_LOG", "ruidosa")
    with pytest.raises(ValueError, match="CONSOLE_LOG"):
        Settings.from_env()
    monkeypatch.delenv("TRANSLATOR_CONSOLE_LOG")
    assert Settings().max_body_bytes == 20 * 1024 * 1024


# ---------- Flags de run.py ----------


def test_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        run.parse_args(["--local", "--lan"])
    assert run.parse_args([]).lan is False
    assert run.parse_args(["--lan"]).lan is True


def test_flag_overrides_env_and_defaults(monkeypatch):
    import app.config as config

    original = config.settings
    monkeypatch.setenv("TRANSLATOR_HOST", "local")
    try:
        run.apply_mode_flag(run.parse_args(["--lan"]))
        assert config.settings.host == "0.0.0.0"  # el flag gana al entorno / .env
        run.apply_mode_flag(run.parse_args(["--local"]))
        assert config.settings.local_only
    finally:
        monkeypatch.setattr(config, "settings", original)


def test_no_flag_keeps_env_value(monkeypatch):
    import app.config as config

    before = config.settings
    run.apply_mode_flag(run.parse_args([]))
    assert config.settings is before


def test_banner_matches_the_binding():
    local = "\n".join(run.access_lines("127.0.0.1", 9966))
    assert "127.0.0.1:9966" in local and "--lan" in local
    wildcard = "\n".join(run.access_lines("0.0.0.0", 9966, [("192.168.1.5", "Wi-Fi")]))
    assert "192.168.1.5:9966" in wildcard and "Modo LAN" in wildcard
    specific = "\n".join(run.access_lines("192.168.1.50", 9966))
    assert "192.168.1.50:9966" in specific
    assert "no accesible" not in specific and "127.0.0.1" not in specific


# ---------- Guarda de Host ----------


def _guarded_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(LocalHostGuard)

    @app.get("/")
    def root():
        return {"ok": True}

    return app


@pytest.mark.parametrize(
    "base", ["http://localhost", "http://127.0.0.1:9966", "http://LOCALHOST:8000"]
)
def test_local_hosts_are_accepted(base):
    assert TestClient(_guarded_app(), base_url=base).get("/").status_code == 200


@pytest.mark.parametrize(
    "base", ["http://evil.example.com", "http://192.168.1.5:9966", "http://localhost.evil.com"]
)
def test_foreign_hosts_are_rejected_in_local_mode(base):
    r = TestClient(_guarded_app(), base_url=base).get("/")
    assert r.status_code == 400 and "modo local" in r.json()["detail"]


# ---------- Tope de cuerpo ----------


def _upload_app(limit: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=limit)

    @app.post("/echo")
    async def echo(request: Request):
        return PlainTextResponse(str(len(await request.body())))

    return app


def test_small_bodies_pass():
    assert TestClient(_upload_app(1024)).post("/echo", content=b"x" * 1024).text == "1024"


def test_declared_size_over_the_limit_is_413():
    r = TestClient(_upload_app(1024)).post("/echo", content=b"x" * 1025)
    assert r.status_code == 413 and "MB" in r.json()["detail"]


def test_chunked_body_over_the_limit_is_413():
    def chunks():
        for _ in range(10):
            yield b"x" * 200  # sin Content-Length: se cuenta lo que llega

    r = TestClient(_upload_app(1024)).post("/echo", content=chunks())
    assert r.status_code == 413


def test_real_app_rejects_huge_requests_and_logs_it(manager, monkeypatch, caplog):
    from app.services import translation

    monkeypatch.setattr(translation, "model_manager", manager)
    big = b'{"text": "' + b"a" * (main.settings.max_body_bytes + 1) + b'"}'
    with caplog.at_level(logging.INFO, logger="translator"), TestClient(main.app) as c:
        r = c.post("/translate", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert any("status=413" in rec.getMessage() for rec in caplog.records)
    assert manager.batches == []  # nunca llegó a traducir


# ---------- Consola simple ----------


def _record(msg="hola", level=logging.INFO, **extra):
    rec = logging.LogRecord("translator.x", level, "f.py", 1, msg, None, None)
    rec.request_id = extra.pop("request_id", "-")
    rec.__dict__.update(extra)
    return rec


def test_console_formatter_is_short_and_uses_summary():
    fmt = ConsoleFormatter()
    line = fmt.format(_record("línea larguísima completa", console_msg="resumen"))
    assert line.endswith("INFO    resumen") and "larguísima" not in line
    assert fmt.format(_record("tal cual")).endswith("tal cual")


def test_console_shows_request_id_only_for_problems():
    fmt = ConsoleFormatter()
    assert "[ab12cd34]" not in fmt.format(_record(request_id="ab12cd34"))
    assert "[ab12cd34]" in fmt.format(_record(level=logging.WARNING, request_id="ab12cd34"))


def test_console_filter_hides_only_marked_records():
    f = ConsoleFilter()
    assert f.filter(_record())
    assert not f.filter(_record(console=False))


def test_log_config_console_modes(tmp_path):
    simple = build_log_config("INFO", tmp_path / "x.log", console="simple")
    assert simple["handlers"]["app_console"]["formatter"] == "console"
    assert "console" in simple["handlers"]["app_console"]["filters"]
    full = build_log_config("INFO", tmp_path / "x.log", console="full")
    assert full["handlers"]["app_console"]["formatter"] == "app"
    # el archivo siempre lleva el formato completo
    assert simple["handlers"]["app_file"]["formatter"] == "app"


def test_console_summary_has_no_client_details():
    line = main._console_summary(
        "192.168.1.20",
        {
            "op": "translate",
            "pair": "en>es",
            "model": "opus-mt-en-es",
            "chars": 117,
            "work_ms": 1400,
            "truncated": False,
        },
        200,
        1500,
        None,
    )
    assert line == "192.168.1.20  translate · 200 · en>es · opus-mt-en-es · 117 car · 1.5 s"
    queued = main._console_summary(
        "ip", {"op": "translate", "chars": 5, "work_ms": 500, "truncated": True}, 200, 4000, "x"
    )
    assert "(cola 3.5 s)" in queued and "[TRUNCADO]" in queued and "— x" in queued

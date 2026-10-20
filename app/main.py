"""Punto de entrada FastAPI. Solo capa HTTP; toda la lógica está en services."""

import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.concurrency import RequestQueue
from app.core.exceptions import (
    InvalidRequestError,
    ModelNotAvailableError,
    ServerBusyError,
    TranslationError,
)
from app.logging_setup import clean, request_id_var, setup_logging
from app.models.schemas import (
    BatchTranslationRequest,
    BatchTranslationResponse,
    ModelInfo,
    PairInfo,
    TranslationRequest,
    TranslationResponse,
)
from app.services.translation import (
    get_loaded_model_ids,
    get_models_status,
    get_pairs,
    preload_model,
    translate_text,
    translate_texts,
)

setup_logging(settings.log_level)
logger = logging.getLogger("translator.api")
access_logger = logging.getLogger("translator.access")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.queue = RequestQueue(settings.max_pending_requests)
    if settings.preload_model:
        try:
            await run_in_threadpool(preload_model, settings.preload_model)
        except TranslationError as exc:
            logger.error("No se pudo precargar '%s': %s", settings.preload_model, exc)
    local = [m["model_id"] for m in get_models_status()]
    logger.info(
        "Servidor listo | modelos_locales=%s | cola_max=%d | api_key=%s | cors=%s",
        ",".join(local) or "ninguno",
        settings.max_pending_requests,
        "si" if settings.api_key else "no",
        ",".join(settings.cors_origins) or "no",
    )
    yield
    logger.info("Servidor detenido")


app = FastAPI(
    title="API de Traducción Local",
    description="Servidor con carga dinámica de modelos y cola FIFO de traducciones",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ---------------- Registro de operaciones (access log) ----------------

_OPERATIONS = {
    "/": "health",
    "/models": "list_models",
    "/pairs": "list_pairs",
    "/translate": "translate",
    "/translate/batch": "translate_batch",
}
# Campos de la operación, en el orden en que se escriben en el log.
_OP_FIELDS = (
    "model",
    "pair",
    "tier",
    "chars",
    "texts",
    "chunks",
    "truncated",
    "warnings",
    "text",
)


def _describe_operation(op: dict[str, Any]) -> str:
    parts = [f"op={op.get('op', '?')}"]
    for key in _OP_FIELDS:
        if key in op:
            value = op[key]
            parts.append(f'{key}="{clean(value, 200)}"' if key == "text" else f"{key}={value}")
    return " ".join(parts)


def _log_request(request: Request, request_id: str, status: int, elapsed_ms: float) -> None:
    op = request.state.op
    error = request.state.error
    client = request.client.host if request.client else "-"
    # Quién: IP + User-Agent + nombre que la propia aplicación cliente declare (X-Client-Name).
    parts = [
        f"ip={client}",
        f'client="{clean(request.headers.get("x-client-name", "-"), 60)}"',
        f'ua="{clean(request.headers.get("user-agent", "-"), 80)}"',
        f"{request.method} {clean(request.url.path, 200)}",
        f"status={status}",
        f"ms={elapsed_ms:.0f}",
    ]
    # Para qué: operación, modelo, par de idiomas, tamaño y resultado.
    parts.append("|")
    parts.append(_describe_operation(op))
    work_ms = op.get("work_ms")
    if work_ms is not None:
        parts.append(f"work_ms={work_ms:.0f} queued_ms={max(elapsed_ms - work_ms, 0):.0f}")
    if error:
        parts.append(f'error="{clean(error, 300)}"')
    level = logging.INFO
    if status >= 500 and status != 503:
        level = logging.ERROR
    elif status >= 400:
        level = logging.WARNING
    access_logger.log(level, " ".join(parts))


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Registra cada petición (cuándo, quién, para qué, resultado) y le asigna un id."""
    request_id = uuid.uuid4().hex[:8]
    request.state.op = {"op": _OPERATIONS.get(request.url.path, "unknown")}
    request.state.error = None
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500  # si la excepción no se controla, ese es el resultado
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            # Error no previsto: se registra con traceback (aún con el id de la petición activo)
            # y se responde con JSON, incluyendo el id para que el cliente pueda citarlo.
            request.state.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Error inesperado en %s %s", request.method, request.url.path)
            response = JSONResponse({"detail": f"Error interno: {exc}"}, status_code=500)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        try:
            _log_request(request, request_id, status, (time.perf_counter() - started) * 1000)
        finally:
            request_id_var.reset(token)


# ---------------- Seguridad opcional ----------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request, key: Annotated[str | None, Security(_api_key_header)]
) -> None:
    """Si TRANSLATOR_API_KEY está definida, exige la cabecera X-API-Key."""
    expected = settings.api_key
    if expected is None:
        return
    if key is None or not secrets.compare_digest(key.encode(), expected.encode()):
        request.state.error = "API key ausente" if key is None else "API key incorrecta"
        raise HTTPException(
            status_code=401,
            detail="API key inválida o ausente (cabecera X-API-Key).",
            headers={"WWW-Authenticate": "ApiKey"},
        )


protected = [Depends(require_api_key)]


# ---------------- Errores ----------------

_STATUS_BY_ERROR: tuple[tuple[type[TranslationError], int], ...] = (
    (ModelNotAvailableError, 404),
    (InvalidRequestError, 422),
    (ServerBusyError, 503),
)


@app.exception_handler(TranslationError)
async def handle_translation_error(request: Request, exc: TranslationError) -> JSONResponse:
    status = next((code for cls, code in _STATUS_BY_ERROR if isinstance(exc, cls)), 500)
    request.state.error = str(exc)
    if status == 500:  # fallo real (carga, inferencia): queda el traceback en el log
        logger.error("Error en %s %s: %s", request.method, request.url.path, exc, exc_info=exc)
    return JSONResponse(
        {"detail": str(exc)},
        status_code=status,
        headers={"Retry-After": "5"} if status == 503 else None,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    request.state.error = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    return await request_validation_exception_handler(request, exc)


def _queue(app_: FastAPI) -> RequestQueue:
    """La cola se crea en el lifespan; si no corrió (p. ej. tests sin contexto), se crea aquí."""
    if not hasattr(app_.state, "queue"):
        app_.state.queue = RequestQueue(settings.max_pending_requests)
    return app_.state.queue


def _timed(op: dict[str, Any], func, *args, **kwargs):
    """Ejecuta ``func`` y anota en ``op`` cuánto tardó el trabajo (sin contar la espera en cola)."""
    started = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        op["work_ms"] = (time.perf_counter() - started) * 1000


def _describe_request(op: dict[str, Any], payload, **sizes: Any) -> None:
    op["model"] = payload.model_id or "auto"
    op["pair"] = (
        f"{payload.source_lang}>{payload.target_lang}"
        if payload.source_lang and payload.target_lang
        else "-"
    )
    op["tier"] = payload.tier or "-"
    op.update(sizes)


# ---------------- Endpoints ----------------


@app.get("/")
def read_root(request: Request):
    queue = _queue(request.app)
    return {
        "status": "ok",
        "loaded_models": get_loaded_model_ids(),
        "queue": {"pending": queue.pending, "max": queue.max_pending},
    }


@app.get("/models", response_model=list[ModelInfo], dependencies=protected)
def list_models():
    """Modelos disponibles en local (descargados por completo) y cuáles están cargados."""
    return get_models_status()


@app.get("/pairs", response_model=list[PairInfo], dependencies=protected)
def list_pairs():
    """Pares de idiomas que pueden traducirse con los modelos locales."""
    return get_pairs()


@app.post("/translate", response_model=TranslationResponse, dependencies=protected)
async def translate(payload: TranslationRequest, request: Request):
    """Traduce un texto. Las traducciones se atienden de una en una, en orden de llegada."""
    op = request.state.op
    _describe_request(op, payload, chars=len(payload.text))
    if settings.log_text_preview:
        op["text"] = payload.text[: settings.log_text_preview]
    result = await _queue(request.app).run(
        _timed,
        op,
        translate_text,
        payload.text,
        model_id=payload.model_id,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        tier=payload.tier,
        max_length=payload.max_length,
    )
    op.update(
        model=result["model_id"],
        pair=f"{result['source_lang']}>{result['target_lang']}",
        chunks=result["chunks"],
        truncated=result["truncated"],
        warnings=len(result["warnings"]),
    )
    return TranslationResponse(**result)


@app.post("/translate/batch", response_model=BatchTranslationResponse, dependencies=protected)
async def translate_batch(payload: BatchTranslationRequest, request: Request):
    """Traduce varios textos con el mismo modelo en una sola petición."""
    op = request.state.op
    _describe_request(
        op, payload, texts=len(payload.texts), chars=sum(len(t) for t in payload.texts)
    )
    if settings.log_text_preview:
        op["text"] = payload.texts[0][: settings.log_text_preview]
    result = await _queue(request.app).run(
        _timed,
        op,
        translate_texts,
        payload.texts,
        model_id=payload.model_id,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        tier=payload.tier,
        max_length=payload.max_length,
    )
    items = result["results"]
    op.update(
        model=result["model_id"],
        pair=f"{result['source_lang']}>{result['target_lang']}",
        chunks=sum(i["chunks"] for i in items),
        truncated=any(i["truncated"] for i in items),
        warnings=sum(len(i["warnings"]) for i in items),
    )
    return BatchTranslationResponse(**result)

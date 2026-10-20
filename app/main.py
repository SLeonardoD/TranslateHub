"""Punto de entrada FastAPI. Solo capa HTTP; toda la lógica está en services."""

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Security
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
from app.logging_setup import setup_logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.queue = RequestQueue(settings.max_pending_requests)
    if settings.preload_model:
        try:
            await run_in_threadpool(preload_model, settings.preload_model)
        except TranslationError as exc:
            logger.error("No se pudo precargar '%s': %s", settings.preload_model, exc)
    logger.info("Servidor listo. Cola máxima: %d peticiones.", settings.max_pending_requests)
    yield
    logger.info("Servidor detenido.")


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


# ---------------- Seguridad opcional ----------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Annotated[str | None, Security(_api_key_header)]) -> None:
    """Si TRANSLATOR_API_KEY está definida, exige la cabecera X-API-Key."""
    expected = settings.api_key
    if expected is None:
        return
    if key is None or not secrets.compare_digest(key.encode(), expected.encode()):
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
    if status == 500:
        logger.error("Error en %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        {"detail": str(exc)},
        status_code=status,
        headers={"Retry-After": "5"} if status == 503 else None,
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Error inesperado en %s %s", request.method, request.url.path)
    return JSONResponse({"detail": f"Error interno: {exc}"}, status_code=500)


def _queue(app_: FastAPI) -> RequestQueue:
    """La cola se crea en el lifespan; si no corrió (p. ej. tests sin contexto), se crea aquí."""
    if not hasattr(app_.state, "queue"):
        app_.state.queue = RequestQueue(settings.max_pending_requests)
    return app_.state.queue


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
    """Todos los modelos del catálogo con su estado (descargado/cargado)."""
    return get_models_status()


@app.get("/pairs", response_model=list[PairInfo], dependencies=protected)
def list_pairs():
    """Pares de idiomas cubiertos y qué modelos los sirven."""
    return get_pairs()


@app.post("/translate", response_model=TranslationResponse, dependencies=protected)
async def translate(payload: TranslationRequest, request: Request):
    """Traduce un texto. Las traducciones se atienden de una en una, en orden de llegada."""
    result = await _queue(request.app).run(
        translate_text,
        payload.text,
        model_id=payload.model_id,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        tier=payload.tier,
        max_length=payload.max_length,
    )
    return TranslationResponse(**result)


@app.post("/translate/batch", response_model=BatchTranslationResponse, dependencies=protected)
async def translate_batch(payload: BatchTranslationRequest, request: Request):
    """Traduce varios textos con el mismo modelo en una sola petición."""
    result = await _queue(request.app).run(
        translate_texts,
        payload.texts,
        model_id=payload.model_id,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        tier=payload.tier,
        max_length=payload.max_length,
    )
    return BatchTranslationResponse(**result)

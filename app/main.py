"""Punto de entrada FastAPI. Solo capa HTTP; toda la lógica está en services."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.models.schemas import (
    TranslationRequest,
    TranslationResponse,
    ModelInfo,
    PairInfo,
)
from app.services.translation import translate_text, get_models_status, get_pairs
from app.core.exceptions import ModelNotAvailableError, ModelLoadError


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Servidor iniciado. Carga perezosa: ningún modelo en RAM hasta la 1ª petición.")
    yield
    print("👋 Servidor detenido.")


app = FastAPI(
    title="API de Traducción Local",
    description="Servidor con carga dinámica de modelos (uno en RAM a la vez)",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    """Todos los modelos del catálogo con su estado (descargado/cargado)."""
    return get_models_status()


@app.get("/pairs", response_model=list[PairInfo])
def list_pairs():
    """Pares de idiomas cubiertos y qué modelos los sirven."""
    return get_pairs()


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        result = translate_text(
            text=request.text,
            model_id=request.model_id,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            tier=request.tier,
            max_length=request.max_length,
        )
        return TranslationResponse(**result)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
"""Modelos de datos expuestos por la API HTTP."""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Texto a traducir")

    # Opción A: modelo explícito
    model_id: Optional[str] = Field(None, description="ID del modelo del catálogo")

    # Opción B: resolver por par + tier
    source_lang: Optional[str] = Field(None, description="Código ISO de origen (ej. 'en')")
    target_lang: Optional[str] = Field(None, description="Código ISO de destino (ej. 'es')")
    tier: Optional[str] = Field(None, description="Preferencia: fast | balanced | quality")

    max_length: Optional[int] = None

    @model_validator(mode="after")
    def _check_selection(self):
        if self.model_id is None and (not self.source_lang or not self.target_lang):
            raise ValueError(
                "Debes indicar 'model_id' o bien 'source_lang' y 'target_lang'."
            )
        return self


class TranslationResponse(BaseModel):
    translated_text: str
    model_id: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class ModelInfo(BaseModel):
    model_id: str
    hf_name: str
    family: str
    type: str
    tier: Optional[str] = None
    description: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    supported_pairs: Optional[list[list[str]]] = None
    downloaded: bool
    loaded: bool


class PairInfo(BaseModel):
    source_lang: str
    target_lang: str
    models: list[str]
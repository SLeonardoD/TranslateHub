"""Modelos de datos expuestos por la API HTTP."""

import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.config import settings
from app.core.languages import normalize_lang

Tier = Literal["fast", "balanced", "quality"]


def _normalize_lang_field(value: object) -> object:
    """'EN', ' es-ES ' → 'en', 'es'. Vacío → None. Lo que no sea str lo rechaza Pydantic."""
    if not isinstance(value, str):
        return value
    code = normalize_lang(value)
    if code is not None and not re.fullmatch(r"[a-z]{2,3}", code):
        raise ValueError(f"código de idioma inválido: {value!r} (usa ISO 639-1, p. ej. 'en')")
    return code


def _strip_or_none(value: object) -> object:
    if isinstance(value, str):
        return value.strip() or None
    return value


LangCode = Annotated[str | None, BeforeValidator(_normalize_lang_field)]
ModelId = Annotated[str | None, BeforeValidator(_strip_or_none), Field(max_length=100)]


def _check_text(value: str) -> str:
    if not value.strip():
        raise ValueError("el texto no puede estar vacío")
    return value


class _ModelSelection(BaseModel):
    """Campos comunes para elegir modelo: explícito (A) o por par + tier (B)."""

    model_config = ConfigDict(protected_namespaces=())

    # Opción A: modelo explícito
    model_id: ModelId = Field(None, description="ID del modelo del catálogo")

    # Opción B: resolver por par + tier
    source_lang: LangCode = Field(None, description="Código ISO de origen (ej. 'en')")
    target_lang: LangCode = Field(None, description="Código ISO de destino (ej. 'es')")
    tier: Tier | None = Field(None, description="Preferencia: fast | balanced | quality")

    max_length: int | None = Field(
        None,
        ge=1,
        le=1024,
        description="Máximo de tokens generados por fragmento. Por defecto, el del modelo.",
    )

    @model_validator(mode="after")
    def _check_selection(self) -> Self:
        if self.model_id is None and (not self.source_lang or not self.target_lang):
            raise ValueError("Debes indicar 'model_id' o bien 'source_lang' y 'target_lang'.")
        return self


class TranslationRequest(_ModelSelection):
    text: str = Field(
        ..., min_length=1, max_length=settings.max_text_chars, description="Texto a traducir"
    )

    _validate_text = field_validator("text")(_check_text)


class BatchTranslationRequest(_ModelSelection):
    texts: list[str] = Field(..., min_length=1, max_length=settings.max_batch_items)

    @field_validator("texts")
    @classmethod
    def _check_texts(cls, texts: list[str]) -> list[str]:
        for text in texts:
            _check_text(text)
            if len(text) > settings.max_text_chars:
                raise ValueError(
                    f"cada texto admite como máximo {settings.max_text_chars} caracteres"
                )
        if sum(len(t) for t in texts) > settings.max_batch_chars:
            raise ValueError(
                f"el lote admite como máximo {settings.max_batch_chars} caracteres en total"
            )
        return texts


class TranslationResult(BaseModel):
    translated_text: str
    chunks: int = Field(1, description="En cuántos fragmentos se dividió el texto para traducirlo")
    truncated: bool = Field(
        False, description="True si alguna traducción alcanzó 'max_length' y puede estar incompleta"
    )
    warnings: list[str] = []


class TranslationResponse(TranslationResult):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    source_lang: str | None = None
    target_lang: str | None = None


class BatchTranslationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    source_lang: str | None = None
    target_lang: str | None = None
    results: list[TranslationResult]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    hf_name: str
    family: str
    type: str
    tier: str | None = None
    description: str
    source_lang: str | None = None
    target_lang: str | None = None
    supported_pairs: list[list[str]] | None = None
    downloaded: bool
    loaded: bool


class PairInfo(BaseModel):
    source_lang: str
    target_lang: str
    models: list[str]

"""Servicio de traducción: resuelve modelos, valida la petición y coordina la inferencia."""

import threading

from app.core.concurrency import check_cancelled
from app.core.exceptions import InvalidRequestError, ModelNotAvailableError
from app.core.languages import normalize_lang
from app.core.model_manager import ModelManager

model_manager = ModelManager()


def translate_texts(
    texts: list[str],
    model_id: str | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
    tier: str | None = None,
    max_length: int | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    """Resuelve el modelo (si hace falta), valida la combinación y traduce todos los textos.

    Devuelve los metadatos del modelo usado y una lista ``results`` (uno por texto).
    ``cancel`` se activa si el cliente se desconecta: el trabajo se abandona.
    """
    check_cancelled(cancel)  # un cliente ausente no debe ni cargar el modelo
    if not texts:
        raise InvalidRequestError("No hay textos que traducir.")

    source_lang, target_lang = normalize_lang(source_lang), normalize_lang(target_lang)
    model_id = (model_id or "").strip() or None

    # 1. Resolver model_id si no viene explícito
    if model_id is None:
        if not source_lang or not target_lang:
            raise InvalidRequestError(
                "Debes indicar 'model_id' o bien 'source_lang' y 'target_lang'."
            )
        # La API solo "ve" los modelos descargados por completo.
        model_id = model_manager.resolve_model_id(source_lang, target_lang, tier, local_only=True)

    info = model_manager.get_model_info(model_id)

    # 2. Validar coherencia según tipo de modelo
    if info["type"] == "multilingual":
        if not source_lang or not target_lang:
            raise InvalidRequestError(
                f"'{model_id}' es multilingüe: se requieren 'source_lang' y 'target_lang'."
            )
        if not model_manager.model_supports_pair(info, source_lang, target_lang):
            raise ModelNotAvailableError(
                f"'{model_id}' no soporta el par {source_lang}→{target_lang}."
            )
    else:
        native_src, native_tgt = info["source_lang"], info["target_lang"]
        if (source_lang and source_lang != native_src) or (
            target_lang and target_lang != native_tgt
        ):
            raise ModelNotAvailableError(
                f"'{model_id}' es {native_src}→{native_tgt}, "
                f"no {source_lang or native_src}→{target_lang or native_tgt}."
            )
        source_lang, target_lang = native_src, native_tgt

    items = model_manager.translate_batch(
        model_id=model_id,
        texts=texts,
        source_lang=source_lang,
        target_lang=target_lang,
        max_length=max_length,
        cancel=cancel,
    )
    return {
        "model_id": model_id,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "results": [
            {
                "translated_text": item.text,
                "chunks": item.chunks,
                "truncated": item.truncated,
                "warnings": item.warnings,
            }
            for item in items
        ],
    }


def translate_text(text: str, **kwargs) -> dict:
    """Traduce un solo texto. Devuelve metadatos + resultado en un único dict plano."""
    result = translate_texts([text], **kwargs)
    return {**{k: v for k, v in result.items() if k != "results"}, **result["results"][0]}


def get_models_status() -> list[dict]:
    loaded = set(model_manager.loaded_model_ids)
    return [
        {
            **m,
            "downloaded": model_manager.is_model_downloaded(m["model_id"]),
            "loaded": m["model_id"] in loaded,
        }
        for m in model_manager.list_models(local_only=True)
    ]


def get_pairs() -> list[dict]:
    return model_manager.list_pairs(local_only=True)


def get_loaded_model_ids() -> list[str]:
    return model_manager.loaded_model_ids


def preload_model(model_id: str) -> None:
    model_manager.preload(model_id)

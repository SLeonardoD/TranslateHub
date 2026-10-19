"""Servicio de traducción: resuelve modelos y coordina la inferencia."""
from typing import Optional

from app.core.model_manager import ModelManager
from app.core.exceptions import ModelNotAvailableError, ModelLoadError

model_manager = ModelManager()


def translate_text(
    text: str,
    model_id: Optional[str] = None,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    tier: Optional[str] = None,
    max_length: Optional[int] = None,
) -> dict:
    """
    Resuelve el modelo (si hace falta), valida la combinación y traduce.
    Devuelve texto + metadatos del modelo usado.
    """
    # 1. Resolver model_id si no viene explícito
    if model_id is None:
        model_id = model_manager.resolve_model_id(source_lang, target_lang, tier)

    info = model_manager.get_model_info(model_id)

    # 2. Validar coherencia según tipo de modelo
    if info["type"] == "multilingual":
        if not source_lang or not target_lang:
            raise ModelLoadError(
                f"'{model_id}' es multilingüe: se requieren 'source_lang' y 'target_lang'."
            )
        if not model_manager.model_supports_pair(info, source_lang, target_lang):
            raise ModelNotAvailableError(
                f"'{model_id}' no soporta el par {source_lang}→{target_lang}."
            )
    else:
        if source_lang and source_lang != info["source_lang"]:
            raise ModelNotAvailableError(
                f"'{model_id}' es {info['source_lang']}→{info['target_lang']}, "
                f"no {source_lang}→{target_lang}."
            )
        source_lang = info["source_lang"]
        target_lang = info["target_lang"]

    translated = model_manager.translate(
        model_id=model_id,
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        max_length=max_length,
    )
    return {
        "translated_text": translated,
        "model_id": model_id,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


def get_models_status() -> list[dict]:
    result = []
    for m in model_manager.list_models():
        mid = m["model_id"]
        result.append({
            **m,
            "downloaded": model_manager.is_model_downloaded(mid),
            "loaded": model_manager.current_model_id == mid,
        })
    return result


def get_pairs() -> list[dict]:
    return model_manager.list_pairs()
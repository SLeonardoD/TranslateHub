"""Normalización de códigos de idioma."""


def normalize_lang(code: str | None) -> str | None:
    """Normaliza un código de idioma: ``" ES-es "`` → ``"es"``, ``""`` → ``None``."""
    if code is None:
        return None
    code = code.strip().lower().replace("_", "-").split("-")[0]
    return code or None

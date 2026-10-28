"""Excepciones del dominio de traducción.

Cada excepción se traduce a un código HTTP concreto en ``app/main.py``.
"""


class TranslationError(Exception):
    """Base de todas las excepciones del dominio."""


class CatalogError(TranslationError):
    """El catálogo ``model_info.json`` no existe o es inválido."""


class ModelNotAvailableError(TranslationError):
    """Modelo no definido, no descargado, o que no cubre el par pedido (HTTP 404)."""


class InvalidRequestError(TranslationError):
    """Petición incoherente: faltan idiomas, idioma no soportado... (HTTP 422)."""


class ModelLoadError(TranslationError):
    """Fallo al descargar o cargar un modelo en memoria (HTTP 500)."""


class InferenceError(TranslationError):
    """Fallo durante la inferencia: falta de memoria, tensores inválidos... (HTTP 500)."""


class ServerBusyError(TranslationError):
    """La cola de peticiones está llena (HTTP 503).

    ``retry_after`` (segundos) va en la cabecera ``Retry-After``; ``estimated_wait_s`` es lo que
    tardaría en vaciarse la cola según el rendimiento medido (None si aún no hay datos).
    """

    def __init__(
        self, message: str, *, retry_after: int = 5, estimated_wait_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.estimated_wait_s = estimated_wait_s


class ClientDisconnectedError(TranslationError):
    """El cliente cerró la conexión: se abandona el trabajo (HTTP 499, nadie leerá la respuesta)."""

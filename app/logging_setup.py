"""Logging: consola + archivo rotativo (escrito en vivo), con un id por petición.

- ``run.py`` pasa ``build_log_config(...)`` a uvicorn: así los logs de la aplicación y los
  de uvicorn (arranque, excepciones no controladas) acaban en consola y en ``translator.log``.
- Cada petición HTTP recibe un id corto (``request_id_var``) que aparece en TODAS las líneas
  que genera, incluidas las del gestor de modelos (carga, descarga de memoria, fallos).
"""

import copy
import logging
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class RequestIdFilter(logging.Filter):
    """Añade ``record.request_id`` (el id de la petición en curso, o '-')."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _KnownWarningFilter(logging.Filter):
    """Descarta una alerta concreta y conocida de transformers (el resto pasa intacto)."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("`max_new_tokens`" in message and "will take precedence" in message)


def silence_known_warnings() -> None:
    """Silencia «Both `max_new_tokens` and `max_length` seem to have been set».

    Es inofensiva: el modelo trae ``max_length=512`` en su generation_config y la API pasa
    ``max_new_tokens``, que es el que debe mandar. Solo se filtra ese mensaje.
    """
    for name in ("transformers.generation.utils", "transformers.generation.configuration_utils"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, _KnownWarningFilter) for f in logger.filters):
            logger.addFilter(_KnownWarningFilter())


def clean(value: object, limit: int = 120) -> str:
    """Valor seguro para una línea de log.

    Los datos del cliente (cabeceras, rutas, textos) no pueden inyectar saltos de línea
    ni comillas que falsifiquen otras líneas del log.
    """
    text = _CONTROL_CHARS.sub("?", str(value)).replace('"', "'")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def setup_logging(level: str = "INFO") -> None:
    """Logging por defecto (consola) cuando la app se arranca sin ``run.py``.

    No hace nada si ya hay configuración: ``run.py`` (dictConfig) o pytest.
    """
    logger = logging.getLogger("translator")
    logger.setLevel(level)
    if logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)


def build_log_config(
    level: str, log_file: Path | None, max_mb: int = 10, backups: int = 5
) -> dict[str, Any]:
    """Configuración para ``uvicorn.run(log_config=...)``: consola + archivo rotativo."""
    from uvicorn.config import LOGGING_CONFIG  # solo hace falta al arrancar con run.py

    config = copy.deepcopy(LOGGING_CONFIG)
    config["filters"] = {"request_id": {"()": "app.logging_setup.RequestIdFilter"}}
    config["formatters"]["app"] = {"format": LOG_FORMAT, "datefmt": DATE_FORMAT}
    config["handlers"]["app_console"] = {
        "class": "logging.StreamHandler",
        "formatter": "app",
        "filters": ["request_id"],
        "stream": "ext://sys.stderr",
    }
    handlers = ["app_console"]
    if log_file is not None:
        config["handlers"]["app_file"] = {
            "class": "logging.handlers.RotatingFileHandler",  # vacía el buffer en cada línea
            "formatter": "app",
            "filters": ["request_id"],
            "filename": str(log_file),
            "maxBytes": max_mb * 1024 * 1024,
            "backupCount": backups,
            "encoding": "utf-8",
        }
        handlers.append("app_file")
        # Arranque y excepciones no controladas de uvicorn, también al archivo.
        config["loggers"]["uvicorn"]["handlers"] = [
            *config["loggers"]["uvicorn"]["handlers"],
            "app_file",
        ]
    config["loggers"]["translator"] = {"handlers": handlers, "level": level, "propagate": False}
    return config

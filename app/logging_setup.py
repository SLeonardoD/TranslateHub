"""Configuración mínima de logging para el servidor y el CLI."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Muestra los logs de la aplicación (logger ``translator``) en consola.

    ``basicConfig`` no hace nada si el root logger ya tiene manejadores (p. ej. en pytest).
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("translator").setLevel(level)

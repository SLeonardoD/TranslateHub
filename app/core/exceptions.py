"""Excepciones personalizadas del dominio de traducción."""


class ModelNotAvailableError(Exception):
    """El modelo no está definido en el catálogo, no está descargado o no cubre el par."""


class ModelLoadError(Exception):
    """Error al cargar el modelo en memoria o al ejecutar la inferencia."""
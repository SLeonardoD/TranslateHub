"""Rutas, constantes y carga del catálogo de modelos."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MODELS_BASE_DIR = PROJECT_ROOT / "models"
MODELS_BASE_DIR.mkdir(exist_ok=True)

MODEL_INFO_PATH = PROJECT_ROOT / "model_info.json"

# Orden de preferencia cuando el cliente no especifica tier
TIER_PREFERENCE = {"fast": 0, "balanced": 1, "quality": 2}


def load_catalog() -> dict:
    """Carga y valida el catálogo model_info.json."""
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(f"No se encontró {MODEL_INFO_PATH}")
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["models"]
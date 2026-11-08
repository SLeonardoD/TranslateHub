"""Rutas, ajustes (variables de entorno) y carga validada del catálogo de modelos."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.exceptions import CatalogError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Archivo .env (sin dependencias externas)
# ---------------------------------------------------------------------------


def parse_dotenv(text: str) -> dict[str, str]:
    """Interpreta el contenido de un .env: ``CLAVE=valor``, comentarios con ``#``,
    comillas simples o dobles y prefijo opcional ``export``. Las líneas inválidas se ignoran."""
    values: dict[str, str] = {}
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        raw = raw.strip()
        if raw[:1] in {'"', "'"}:
            end = raw.find(raw[0], 1)
            value = raw[1:end] if end != -1 else raw[1:]
        else:
            value = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        values[key] = value
    return values


def load_dotenv(path: Path | None = None) -> Path | None:
    """Carga el .env en ``os.environ`` sin pisar variables que ya estén definidas.

    Precedencia: variable de entorno real > .env > valor por defecto.
    ``TRANSLATOR_ENV_FILE`` cambia la ruta del archivo (``off`` lo desactiva).
    Devuelve la ruta cargada, o None si no hay archivo.
    """
    if path is None:
        custom = (os.environ.get("TRANSLATOR_ENV_FILE") or "").strip()
        if custom.lower() in {"off", "none", "false", "0"}:
            return None
        path = Path(custom) if custom else PROJECT_ROOT / ".env"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    if not path.is_file():
        return None
    try:
        values = parse_dotenv(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"No se pudo leer {path}: {exc}") from exc
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return path


ENV_FILE_LOADED = load_dotenv()  # antes de leer cualquier TRANSLATOR_*

# La carpeta se crea al descargar un modelo, no al importar el módulo.
MODELS_BASE_DIR = Path(os.environ.get("TRANSLATOR_MODELS_DIR") or PROJECT_ROOT / "models")

MODEL_INFO_PATH = PROJECT_ROOT / "model_info.json"

# Orden de preferencia cuando el cliente no especifica tier
TIER_PREFERENCE = {"fast": 0, "balanced": 1, "quality": 2}


# ---------------------------------------------------------------------------
# Ajustes (variables de entorno TRANSLATOR_*)
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    value = os.environ.get(f"TRANSLATOR_{name}")
    return value.strip() if value and value.strip() else None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"TRANSLATOR_{name} debe ser un entero (valor actual: {raw!r}).") from None
    if value < minimum:
        raise ValueError(f"TRANSLATOR_{name} debe ser >= {minimum} (valor actual: {value}).")
    return value


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def resolve_host(raw: str) -> str:
    """Admite alias: ``local`` → solo este equipo, ``lan`` → toda la red local.

    Cualquier otro valor se usa tal cual (p. ej. la IP concreta de tu tarjeta de red).
    """
    value = raw.strip()
    alias = value.lower()
    if alias in {"local", "localhost"}:
        return "127.0.0.1"
    if alias in {"lan", "red", "all"}:
        return "0.0.0.0"
    return value


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"  # por defecto solo este equipo; "lan" / 0.0.0.0 abre a la red
    port: int = 9500
    device: str = "auto"  # auto | cpu | cuda
    max_loaded_models: int = 1  # modelos simultáneos en RAM (LRU)
    max_pending_requests: int = 16  # peticiones en curso + en cola
    chunk_target_tokens: int = 200  # tamaño objetivo de cada fragmento de texto
    batch_size: int = 8  # fragmentos por llamada a generate()
    max_text_chars: int = 20_000  # caracteres máximos por texto
    max_batch_items: int = 32  # textos máximos por petición batch (0 = sin límite)
    max_batch_chars: int = 100_000  # caracteres máximos por batch (se ignora si items = 0)
    max_body_mb: int = 20  # tamaño máximo del cuerpo de una petición (HTTP 413 si se supera)
    preload_model: str | None = None  # model_id a cargar al arrancar
    api_key: str | None = None  # si se define, se exige la cabecera X-API-Key
    cors_origins: tuple[str, ...] = ()  # orígenes permitidos para navegadores
    log_level: str = "INFO"
    log_file: str = "translator.log"  # relativo a la raíz del proyecto; "off" lo desactiva
    log_max_mb: int = 10  # tamaño máximo antes de rotar el archivo
    log_backups: int = 5  # archivos rotados que se conservan (translator.log.1 ...)
    log_text_preview: int = 0  # caracteres del texto a registrar (0 = nunca se registra el texto)
    console_log: str = "simple"  # simple = resumen por petición; full = igual que el archivo

    @property
    def local_only(self) -> bool:
        """True si el servidor escucha solo en este equipo (loopback)."""
        return self.host.lower() in _LOCAL_HOSTS or self.host.startswith("127.")

    @property
    def max_body_bytes(self) -> int:
        return self.max_body_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Self:
        device = (_env("DEVICE") or "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"TRANSLATOR_DEVICE debe ser auto, cpu o cuda (valor actual: {device!r})."
            )
        console_log = (_env("CONSOLE_LOG") or cls.console_log).lower()
        if console_log not in {"simple", "full"}:
            raise ValueError(
                f"TRANSLATOR_CONSOLE_LOG debe ser simple o full (valor actual: {console_log!r})."
            )
        origins = tuple(o.strip() for o in (_env("CORS_ORIGINS") or "").split(",") if o.strip())
        return cls(
            host=resolve_host(_env("HOST") or cls.host),
            port=_env_int("PORT", cls.port),
            device=device,
            max_loaded_models=_env_int("MAX_LOADED_MODELS", cls.max_loaded_models),
            max_pending_requests=_env_int("MAX_PENDING_REQUESTS", cls.max_pending_requests),
            chunk_target_tokens=_env_int(
                "CHUNK_TARGET_TOKENS", cls.chunk_target_tokens, minimum=16
            ),
            batch_size=_env_int("BATCH_SIZE", cls.batch_size),
            max_text_chars=_env_int("MAX_TEXT_CHARS", cls.max_text_chars),
            max_batch_items=_env_int("MAX_BATCH_ITEMS", cls.max_batch_items, minimum=0),
            max_batch_chars=_env_int("MAX_BATCH_CHARS", cls.max_batch_chars),
            max_body_mb=_env_int("MAX_BODY_MB", cls.max_body_mb),
            preload_model=_env("PRELOAD_MODEL"),
            api_key=_env("API_KEY"),
            cors_origins=origins,
            log_level=(_env("LOG_LEVEL") or cls.log_level).upper(),
            log_file=_env("LOG_FILE") or cls.log_file,
            log_max_mb=_env_int("LOG_MAX_MB", cls.log_max_mb),
            log_backups=_env_int("LOG_BACKUPS", cls.log_backups, minimum=0),
            log_text_preview=_env_int("LOG_TEXT_PREVIEW", cls.log_text_preview, minimum=0),
            console_log=console_log,
        )

    def resolve_log_file(self) -> Path | None:
        """Ruta absoluta del archivo de log, o None si está desactivado."""
        if self.log_file.lower() in {"off", "none", "false", "0"}:
            return None
        path = Path(self.log_file)
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings.from_env()


# ---------------------------------------------------------------------------
# Catálogo de modelos
# ---------------------------------------------------------------------------

Tier = Literal["fast", "balanced", "quality"]
_LANG_PATTERN = r"^[a-z]{2,3}$"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class CatalogEntry(BaseModel):
    """Una entrada de ``model_info.json``. Rechaza campos desconocidos (erratas)."""

    model_config = ConfigDict(extra="forbid")

    hf_name: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    family: str = "opus-mt"
    type: Literal["pair", "multilingual"]
    tier: Tier = "balanced"
    description: str = Field(min_length=1)
    default_max_length: int = Field(512, ge=1, le=4096)
    source_lang: str | None = Field(None, pattern=_LANG_PATTERN)
    target_lang: str | None = Field(None, pattern=_LANG_PATTERN)
    supported_pairs: list[tuple[str, str]] | None = None
    language_code_map: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.type == "pair":
            if not (self.source_lang and self.target_lang):
                raise ValueError("los modelos 'pair' requieren 'source_lang' y 'target_lang'")
            if self.source_lang == self.target_lang:
                raise ValueError("'source_lang' y 'target_lang' no pueden ser iguales")
            if self.supported_pairs or self.language_code_map:
                raise ValueError(
                    "'supported_pairs' y 'language_code_map' solo aplican a 'multilingual'"
                )
            return self

        if self.family != "nllb":
            raise ValueError("los modelos 'multilingual' solo están soportados con family='nllb'")
        if self.source_lang or self.target_lang:
            raise ValueError("'source_lang' y 'target_lang' solo aplican a modelos 'pair'")
        if not self.supported_pairs:
            raise ValueError("los modelos 'multilingual' requieren 'supported_pairs'")
        if not self.language_code_map:
            raise ValueError("los modelos 'multilingual' requieren 'language_code_map'")
        languages = {lang for pair in self.supported_pairs for lang in pair}
        missing = sorted(languages - set(self.language_code_map))
        if missing:
            raise ValueError(f"'language_code_map' no incluye los idiomas: {', '.join(missing)}")
        return self


def parse_catalog(raw: object) -> dict[str, dict]:
    """Valida el contenido de ``model_info.json`` y lo devuelve como dict plano.

    Acumula todos los errores para que se corrijan de una vez.
    """
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict) or not models:
        raise CatalogError(
            'model_info.json debe tener la forma {"models": {...}} con al menos un modelo.'
        )

    errors: list[str] = []
    catalog: dict[str, dict] = {}
    for model_id, entry in models.items():
        if not _MODEL_ID_RE.match(model_id):
            errors.append(
                f"  - '{model_id}': el ID solo puede contener letras, números, '.', '_' y '-'"
            )
            continue
        try:
            parsed = CatalogEntry.model_validate(entry)
        except ValidationError as exc:
            for err in exc.errors():
                where = ".".join(str(p) for p in err["loc"])
                msg = err["msg"].removeprefix("Value error, ")
                errors.append(f"  - {model_id}{'.' + where if where else ''}: {msg}")
        else:
            catalog[model_id] = parsed.model_dump(mode="json", exclude_none=True)

    if errors:
        raise CatalogError("Catálogo model_info.json inválido:\n" + "\n".join(errors))
    return catalog


def load_catalog(path: Path | None = None) -> dict[str, dict]:
    """Carga y valida el catálogo de modelos."""
    path = path or MODEL_INFO_PATH
    if not path.exists():
        raise CatalogError(f"No se encontró {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogError(
            f"{path.name} no es un JSON válido (línea {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc
    return parse_catalog(raw)

"""Gestor centralizado de modelos de traducción.

- Catálogo plano con ID único por modelo (varios modelos por par de idiomas).
- Carga perezosa. Por defecto un solo modelo en RAM (TRANSLATOR_MAX_LOADED_MODELS=1);
  con valores mayores se conservan los últimos usados (LRU).
- Thread-safe: el lock cubre carga + inferencia (operación atómica).
- La API NUNCA dispara descargas; eso es exclusivo del CLI.
- Los textos largos se trocean en fragmentos que caben en el modelo: nada se corta
  en silencio, y si una traducción alcanza ``max_length`` se avisa en la respuesta.
- ``torch`` y ``transformers`` se importan solo al cargar un modelo, así que el CLI
  (list, pairs, download...) arranca al instante.
"""

import gc
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import MODELS_BASE_DIR, TIER_PREFERENCE, load_catalog, settings
from app.core.concurrency import check_cancelled
from app.core.exceptions import (
    CatalogError,
    InferenceError,
    InvalidRequestError,
    ModelLoadError,
    ModelNotAvailableError,
    TranslationError,
)
from app.core.text_utils import assemble, build_chunks
from app.logging_setup import setup_logging

logger = logging.getLogger("translator.manager")

# Se escribe al terminar la descarga: sin él, el modelo se considera NO descargado.
COMPLETE_MARKER = ".download_complete"

_MAX_INPUT_TOKENS = 512
_SPECIAL_TOKENS_MARGIN = 8

# Formatos de pesos que nunca se necesitan para PyTorch.
_ALWAYS_IGNORE = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.onnx",
    "*.tflite",
    "*.mlmodel",
    "*.gguf",
    "flax_model*",
    "tf_model*",
    "onnx/*",
    "coreml/*",
    ".gitattributes",
]
# Si el repo tiene safetensors, los .bin/.pt duplicados sobran.
_IGNORE_WHEN_SAFETENSORS = ["*.bin", "*.pt", "*.pth"]


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str


@dataclass(slots=True)
class ItemResult:
    """Resultado de traducir un texto."""

    text: str
    chunks: int
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


def truncated_flags(
    last_tokens: list[int], output_len: int, max_new_tokens: int, pad_id: int | None
) -> list[bool]:
    """Detecta qué secuencias de un lote se cortaron por agotar ``max_new_tokens``.

    ``output_len`` incluye el token inicial del decoder, así que un lote que agotó el
    presupuesto mide ``max_new_tokens + 1``. Una secuencia que terminó antes queda rellena
    con ``pad`` hasta el final; una que sigue "activa" en la última posición se cortó.
    Nota: los modelos Marian fuerzan un EOS en el último paso (``forced_eos_token_id``),
    así que NO basta con mirar si el último token es EOS.
    """
    if output_len <= max_new_tokens:
        return [False] * len(last_tokens)
    return [token != pad_id for token in last_tokens]


def _release_memory() -> None:
    """Libera memoria de Python y, si torch ya está cargado, la caché de CUDA."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ignore_patterns(repo_files: list[str]) -> list[str]:
    patterns = list(_ALWAYS_IGNORE)
    if any(f.endswith(".safetensors") for f in repo_files):
        patterns += _IGNORE_WHEN_SAFETENSORS
    return patterns


class ModelManager:
    def __init__(
        self,
        catalog: dict[str, dict] | None = None,
        models_dir: Path | None = None,
        max_loaded: int | None = None,
    ):
        self._catalog = catalog if catalog is not None else load_catalog()
        self._models_dir = Path(models_dir) if models_dir else MODELS_BASE_DIR
        self._max_loaded = max_loaded or settings.max_loaded_models
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()  # del menos al más reciente
        self._lock = threading.Lock()

    # ---------------- Catálogo ----------------

    def get_model_info(self, model_id: str) -> dict:
        if model_id not in self._catalog:
            raise ModelNotAvailableError(f"Modelo '{model_id}' no definido en el catálogo.")
        return self._catalog[model_id]

    def list_models(self, local_only: bool = False) -> list[dict]:
        """Modelos del catálogo; con ``local_only`` solo los descargados por completo."""
        return [
            {"model_id": mid, **info}
            for mid, info in self._catalog.items()
            if not local_only or self.is_model_downloaded(mid)
        ]

    def list_pairs(self, local_only: bool = False) -> list[dict]:
        """Agrupa los modelos por par de idiomas para descubrimiento."""
        pairs: dict[tuple[str, str], list[str]] = {}
        for mid, info in self._catalog.items():
            if local_only and not self.is_model_downloaded(mid):
                continue
            if info["type"] == "pair":
                pairs.setdefault((info["source_lang"], info["target_lang"]), []).append(mid)
            else:
                for src, tgt in info.get("supported_pairs", []):
                    pairs.setdefault((src, tgt), []).append(mid)
        return [
            {"source_lang": s, "target_lang": t, "models": sorted(ids)}
            for (s, t), ids in sorted(pairs.items())
        ]

    def model_supports_pair(self, info: dict, src: str, tgt: str) -> bool:
        if info["type"] == "pair":
            return info["source_lang"] == src and info["target_lang"] == tgt
        return [src, tgt] in info.get("supported_pairs", [])

    def get_models_for_pair(
        self, src: str, tgt: str, tier: str | None = None, local_only: bool = False
    ) -> list[str]:
        """Devuelve los model_id que cubren el par, opcionalmente filtrados por tier."""
        return [
            mid
            for mid, info in self._catalog.items()
            if self.model_supports_pair(info, src, tgt)
            and (not tier or info.get("tier") == tier)
            and (not local_only or self.is_model_downloaded(mid))
        ]

    def _by_preference(self, model_ids: list[str]) -> list[str]:
        return sorted(
            model_ids,
            key=lambda mid: TIER_PREFERENCE.get(self._catalog[mid].get("tier", "balanced"), 99),
        )

    def resolve_model_id(
        self, src: str, tgt: str, tier: str | None = None, local_only: bool = False
    ) -> str:
        """Elige el mejor model_id para un par y tier opcional.

        Si no se especifica tier, prefiere 'fast' (menor latencia). Con ``local_only`` solo
        considera modelos descargados por completo: si ninguno cubre el par, falla indicando
        qué modelo del catálogo habría que descargar.
        """
        what = f"{src}→{tgt}" + (f" con tier='{tier}'" if tier else "")
        defined = self.get_models_for_pair(src, tgt, tier)
        if not defined:
            raise ModelNotAvailableError(f"No hay modelos definidos para {what}")
        candidates = self._by_preference(
            [m for m in defined if not local_only or self.is_model_downloaded(m)]
        )
        if not candidates:
            suggestion = self._by_preference(defined)[0]
            raise ModelNotAvailableError(
                f"Ningún modelo local cubre {what}. Descarga uno con el CLI: "
                f"python -m app.core.model_manager download {suggestion}"
            )
        return candidates[0]

    # ---------------- Comprobación local ----------------

    def _get_local_path(self, model_id: str) -> Path:
        info = self.get_model_info(model_id)
        return self._models_dir / info["hf_name"].replace("/", "__")

    @staticmethod
    def _has_weights(path: Path) -> bool:
        return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))

    def is_model_downloaded(self, model_id: str) -> bool:
        """True solo si la descarga terminó (marcador) y los archivos siguen ahí.

        Comprueba marcador + config.json + pesos, así que un modelo al que se le borró
        algo a mano tampoco cuenta como disponible.
        """
        path = self._get_local_path(model_id)
        return (
            (path / COMPLETE_MARKER).is_file()
            and (path / "config.json").is_file()
            and self._has_weights(path)
        )

    def get_download_state(self, model_id: str) -> str:
        """'ok' | 'incomplete' (hay archivos pero sin marcador) | 'missing'."""
        if self.is_model_downloaded(model_id):
            return "ok"
        path = self._get_local_path(model_id)
        has_files = path.is_dir() and any(p.name != ".cache" for p in path.iterdir())
        return "incomplete" if has_files else "missing"

    # ---------------- Descarga (solo desde CLI) ----------------

    def download_model(self, model_id: str, force: bool = False) -> None:
        info = self.get_model_info(model_id)
        hf_name = info["hf_name"]
        local_path = self._get_local_path(model_id)
        marker = local_path / COMPLETE_MARKER

        if self.is_model_downloaded(model_id) and not force:
            print(f"✅ '{model_id}' ya está descargado en {local_path}")
            return

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise ModelLoadError(
                "Falta 'huggingface_hub': pip install -r requirements.txt"
            ) from exc

        if local_path.exists() and not marker.exists():
            print(f"♻️  Descarga previa incompleta o sin verificar: se reanuda/valida '{model_id}'.")

        print(f"⬇️ Descargando '{hf_name}' en {local_path} (solo pesos PyTorch)...")
        # Los archivos van DIRECTAMENTE a local_path. Por si alguna versión de huggingface_hub
        # necesitara una caché temporal, se redirige a models/.hf_tmp (y se borra al terminar)
        # para no tocar nunca ~/.cache/huggingface.
        tmp_cache = self._models_dir / ".hf_tmp"
        os.environ.setdefault("HF_XET_CACHE", str(tmp_cache / "xet"))
        try:
            local_path.mkdir(parents=True, exist_ok=True)
            marker.unlink(missing_ok=True)
            ignore = _ignore_patterns(HfApi().list_repo_files(hf_name))
            snapshot_download(
                repo_id=hf_name,
                local_dir=str(local_path),
                cache_dir=str(tmp_cache / "hub"),
                ignore_patterns=ignore,
            )
            self._verify_download(local_path, model_id)
            marker.write_text(
                json.dumps(
                    {"hf_name": hf_name, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                ),
                encoding="utf-8",
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Error al descargar '{model_id}': {exc}") from exc
        finally:
            shutil.rmtree(tmp_cache, ignore_errors=True)
        print(f"✅ '{model_id}' descargado correctamente en {local_path}")

    @classmethod
    def _verify_download(cls, local_path: Path, model_id: str) -> None:
        if not (local_path / "config.json").exists():
            raise ModelLoadError(f"Descarga incompleta de '{model_id}': falta config.json.")
        if not cls._has_weights(local_path):
            raise ModelLoadError(f"Descarga incompleta de '{model_id}': no se encontraron pesos.")

    def download_all(self, force: bool = False) -> int:
        """Descarga todo el catálogo. Devuelve cuántos modelos fallaron."""
        failures = 0
        for mid in self._catalog:
            try:
                self.download_model(mid, force=force)
            except TranslationError as exc:
                failures += 1
                print(f"❌ Error con '{mid}': {exc}")
        return failures

    # ---------------- Estado en memoria ----------------

    @property
    def loaded_model_ids(self) -> list[str]:
        return list(self._loaded)

    @property
    def current_model_id(self) -> str | None:
        """El modelo usado más recientemente (o None si no hay ninguno cargado)."""
        return next(reversed(self._loaded), None)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    # ---------------- Carga (interno, asume lock tomado) ----------------

    def _evict_oldest_locked(self) -> None:
        model_id, _ = self._loaded.popitem(last=False)
        logger.info("Liberando '%s' de memoria...", model_id)
        _release_memory()

    def _resolve_device(self, torch: Any) -> str:
        wanted = settings.device
        has_cuda = torch.cuda.is_available()
        if wanted == "cuda" and not has_cuda:
            raise ModelLoadError("TRANSLATOR_DEVICE=cuda pero CUDA no está disponible.")
        return "cuda" if wanted == "cuda" or (wanted == "auto" and has_cuda) else "cpu"

    def _load_pretrained(self, local_path: Path) -> LoadedModel:
        """Carga tokenizador y modelo desde disco, sin red ni caché de HF."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self._resolve_device(torch)
        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(local_path), local_files_only=True, dtype=dtype
        )
        model.to(device)
        model.eval()
        return LoadedModel(model=model, tokenizer=tokenizer, device=device)

    def _ensure_model_loaded_locked(self, model_id: str) -> LoadedModel:
        """Asume lock tomado. Nunca descarga: falla si no está local."""
        loaded = self._loaded.get(model_id)
        if loaded is not None:
            self._loaded.move_to_end(model_id)
            return loaded

        if not self.is_model_downloaded(model_id):
            raise ModelNotAvailableError(
                f"'{model_id}' no está descargado (o la descarga quedó incompleta). "
                f"Usa el CLI: python -m app.core.model_manager download {model_id}"
            )

        while len(self._loaded) >= self._max_loaded:
            self._evict_oldest_locked()

        local_path = self._get_local_path(model_id)
        logger.info("Cargando '%s' desde %s...", model_id, local_path)
        started = time.perf_counter()
        try:
            loaded = self._load_pretrained(local_path)
        except ModelLoadError:
            _release_memory()
            raise
        except Exception as exc:
            _release_memory()
            logger.exception("Fallo al cargar '%s'", model_id)
            raise ModelLoadError(f"Error al cargar '{model_id}': {exc}") from exc

        self._loaded[model_id] = loaded
        logger.info(
            "'%s' listo en %.1fs (%s).", model_id, time.perf_counter() - started, loaded.device
        )
        return loaded

    def preload(self, model_id: str) -> None:
        """Carga un modelo por adelantado (evita la espera en la primera petición)."""
        with self._lock:
            self._ensure_model_loaded_locked(model_id)

    # ---------------- Inferencia ----------------

    def translate_batch(
        self,
        model_id: str,
        texts: list[str],
        source_lang: str | None = None,
        target_lang: str | None = None,
        max_length: int | None = None,
        cancel: threading.Event | None = None,
    ) -> list[ItemResult]:
        """Operación atómica: lock → asegurar modelo cargado → traducir todos los textos.

        Otra petición (incluso de otro modelo) esperará a que termine. Si ``cancel`` se activa
        (el cliente se fue), se abandona antes de cargar el modelo o entre dos lotes.
        """
        check_cancelled(cancel)
        with self._lock:
            loaded = self._ensure_model_loaded_locked(model_id)
            info = self.get_model_info(model_id)
            try:
                return self._translate_locked(
                    loaded, info, model_id, texts, source_lang, target_lang, max_length, cancel
                )
            except TranslationError:
                raise
            except Exception as exc:
                logger.exception("Fallo de inferencia con '%s'", model_id)
                _release_memory()
                raise InferenceError(
                    f"Error durante la inferencia con '{model_id}': {exc}"
                ) from exc

    def _translate_locked(
        self,
        loaded: LoadedModel,
        info: dict,
        model_id: str,
        texts: list[str],
        source_lang: str | None,
        target_lang: str | None,
        max_length: int | None,
        cancel: threading.Event | None = None,
    ) -> list[ItemResult]:
        forced_bos_token_id = None
        if info.get("family") == "nllb":
            forced_bos_token_id = self._prepare_nllb(
                loaded, info, model_id, source_lang, target_lang
            )

        max_new_tokens = max_length or info.get("default_max_length", 512)
        input_limit = min(
            int(getattr(loaded.tokenizer, "model_max_length", _MAX_INPUT_TOKENS)), _MAX_INPUT_TOKENS
        )
        target = max(8, min(settings.chunk_target_tokens, input_limit - _SPECIAL_TOKENS_MARGIN))

        def count(batch: Any) -> list[int]:
            return self._count_tokens(loaded, list(batch))

        segmentations = [build_chunks(text, count, target) for text in texts]
        check_cancelled(cancel)

        # Fragmentos de longitud parecida juntos => menos relleno (padding) en cada batch.
        order = sorted(
            ((i, j) for i, seg in enumerate(segmentations) for j in range(len(seg.chunks))),
            key=lambda ij: len(segmentations[ij[0]].chunks[ij[1]].text),
        )
        outputs: dict[tuple[int, int], tuple[str, bool]] = {}
        for k in range(0, len(order), settings.batch_size):
            check_cancelled(cancel)  # entre lotes: la parte más lenta ya no se desperdicia
            keys = order[k : k + settings.batch_size]
            batch_texts = [segmentations[i].chunks[j].text for i, j in keys]
            results = self._generate_batch(
                loaded,
                batch_texts,
                max_new_tokens=max_new_tokens,
                forced_bos_token_id=forced_bos_token_id,
            )
            outputs.update(zip(keys, results, strict=True))

        items: list[ItemResult] = []
        for i, seg in enumerate(segmentations):
            translations = [outputs[(i, j)][0].strip() for j in range(len(seg.chunks))]
            cut = sum(outputs[(i, j)][1] for j in range(len(seg.chunks)))
            warnings: list[str] = []
            if seg.forced_splits:
                warnings.append(
                    f"{seg.forced_splits} oración(es) demasiado larga(s) se dividieron por palabras; "
                    "la calidad en esos puntos puede verse afectada."
                )
            if cut:
                warnings.append(
                    f"{cut} fragmento(s) alcanzaron max_length={max_new_tokens} y su traducción puede "
                    "estar incompleta. Aumenta 'max_length'."
                )
            items.append(
                ItemResult(assemble(seg, translations), len(seg.chunks), cut > 0, warnings)
            )
        return items

    def _prepare_nllb(
        self,
        loaded: LoadedModel,
        info: dict,
        model_id: str,
        source_lang: str | None,
        target_lang: str | None,
    ) -> int:
        """Fija el idioma de origen en el tokenizador y devuelve el id del token de destino."""
        if not source_lang or not target_lang:
            raise InvalidRequestError(
                "El modelo NLLB requiere 'source_lang' y 'target_lang' en la petición."
            )
        lang_map = info.get("language_code_map", {})
        src_code, tgt_code = lang_map.get(source_lang), lang_map.get(target_lang)
        if not src_code or not tgt_code:
            raise InvalidRequestError(
                f"Par {source_lang}→{target_lang} no soportado por '{model_id}'."
            )

        tokenizer = loaded.tokenizer
        tokenizer.src_lang = src_code
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
        if forced_bos is None or forced_bos == getattr(tokenizer, "unk_token_id", None):
            raise ModelLoadError(
                f"El tokenizador de '{model_id}' no reconoce el código '{tgt_code}'."
            )
        return forced_bos

    def _count_tokens(self, loaded: LoadedModel, texts: list[str]) -> list[int]:
        encoded = loaded.tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def _generate_batch(
        self,
        loaded: LoadedModel,
        texts: list[str],
        *,
        max_new_tokens: int,
        forced_bos_token_id: int | None,
    ) -> list[tuple[str, bool]]:
        """Traduce un lote. Devuelve (texto, truncado) por entrada.

        ``truncado`` es True si la generación agotó ``max_new_tokens`` (ver ``truncated_flags``).
        """
        import torch

        tokenizer, model = loaded.tokenizer, loaded.model
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            loaded.device
        )
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        truncated = truncated_flags(
            outputs[:, -1].tolist(), outputs.shape[1], max_new_tokens, tokenizer.pad_token_id
        )
        return list(zip(decoded, truncated, strict=True))


# ---------------- CLI ----------------


def _fetch_server_models() -> list[dict] | None:
    """Pregunta al servidor en marcha (otro proceso) qué modelos tiene cargados."""
    import urllib.request

    request = urllib.request.Request(f"http://127.0.0.1:{settings.port}/models")
    if settings.api_key:
        request.add_header("X-API-Key", settings.api_key)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({})
    )  # sin proxys para localhost
    try:
        with opener.open(request, timeout=3) as response:
            return json.load(response)
    except Exception:
        return None


def _cli(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # consolas Windows sin UTF-8
    setup_logging(settings.log_level)

    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Uso: python -m app.core.model_manager <comando> [args]")
        print("Comandos:")
        print("  list                          Lista todos los modelos del catálogo")
        print("  pairs                         Lista pares de idiomas cubiertos")
        print(
            "  status                        Modelos cargados en el servidor y estado de descargas"
        )
        print(
            "  download <model_id> [--force] Descarga un modelo (--force: volver a validar/descargar)"
        )
        print("  download-all [--force]        Descarga todos los modelos del catálogo")
        return 0

    try:
        manager = ModelManager()
    except CatalogError as exc:
        print(f"❌ {exc}")
        return 1

    cmd, rest = args[0], args[1:]
    force = "--force" in rest
    positional = [a for a in rest if not a.startswith("--")]
    icons = {"ok": "✅", "incomplete": "⚠️ ", "missing": "⬇️ "}

    if cmd == "list":
        for m in manager.list_models():
            icon = icons[manager.get_download_state(m["model_id"])]
            print(f"  {icon} {m['model_id']:35s} [{m.get('tier', '?'):7s}] {m['description']}")
        if any(
            manager.get_download_state(m["model_id"]) == "incomplete" for m in manager.list_models()
        ):
            print(
                "\n⚠️  = descarga incompleta o hecha con una versión anterior: vuelve a ejecutar 'download <model_id>'."
            )
    elif cmd == "pairs":
        for p in manager.list_pairs():
            print(f"  {p['source_lang']}→{p['target_lang']}: {', '.join(p['models'])}")
    elif cmd == "status":
        server = _fetch_server_models()
        if server is None:
            print(
                f"Servidor no accesible en 127.0.0.1:{settings.port}: no se puede saber qué hay cargado en RAM."
            )
            loaded: set[str] = set()
        else:
            loaded = {m["model_id"] for m in server if m.get("loaded")}
            print(f"Modelos en memoria (servidor): {', '.join(sorted(loaded)) or 'Ninguno'}")
        for m in manager.list_models():
            mid = m["model_id"]
            state = {"ok": "descargado", "incomplete": "INCOMPLETO", "missing": "no descargado"}[
                manager.get_download_state(mid)
            ]
            print(
                f"  {icons[manager.get_download_state(mid)]} {mid:35s} {state}{'  [en RAM]' if mid in loaded else ''}"
            )
    elif cmd == "download":
        if not positional:
            print("Error: se requiere <model_id>")
            return 1
        try:
            manager.download_model(positional[0], force=force)
        except TranslationError as exc:
            print(f"❌ {exc}")
            return 1
    elif cmd == "download-all":
        return 1 if manager.download_all(force=force) else 0
    else:
        print(f"Comando desconocido: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

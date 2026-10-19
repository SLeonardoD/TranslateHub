"""Gestor centralizado de modelos de traducción.

- Catálogo plano con ID único por modelo (varios modelos por par de idiomas).
- Carga perezosa: un solo modelo en RAM a la vez.
- Thread-safe: el lock cubre carga + inferencia (operación atómica).
- La API NUNCA dispara descargas; eso es exclusivo del CLI.
"""
import threading
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import (
    MODELS_BASE_DIR,
    TIER_PREFERENCE,
    load_catalog,
)
from .exceptions import ModelNotAvailableError, ModelLoadError


class ModelManager:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_id: Optional[str] = None
        self._lock = threading.Lock()
        self._catalog = load_catalog()

    # ---------------- Catálogo ----------------

    def get_model_info(self, model_id: str) -> dict:
        if model_id not in self._catalog:
            raise ModelNotAvailableError(f"Modelo '{model_id}' no definido en el catálogo.")
        return self._catalog[model_id]

    def list_models(self) -> list[dict]:
        return [{"model_id": mid, **info} for mid, info in self._catalog.items()]

    def list_pairs(self) -> list[dict]:
        """Agrupa los modelos por par de idiomas para descubrimiento."""
        pairs: dict[tuple[str, str], list[str]] = {}
        for mid, info in self._catalog.items():
            if info["type"] == "pair":
                key = (info["source_lang"], info["target_lang"])
                pairs.setdefault(key, []).append(mid)
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
        self, src: str, tgt: str, tier: Optional[str] = None
    ) -> list[str]:
        """Devuelve los model_id que cubren el par, opcionalmente filtrados por tier."""
        results = []
        for mid, info in self._catalog.items():
            if not self.model_supports_pair(info, src, tgt):
                continue
            if tier and info.get("tier") != tier:
                continue
            results.append(mid)
        return results

    def resolve_model_id(
        self, src: str, tgt: str, tier: Optional[str] = None
    ) -> str:
        """
        Elige el mejor model_id para un par y tier opcional.
        Si no se especifica tier, prefiere 'fast' (menor latencia).
        """
        candidates = self.get_models_for_pair(src, tgt, tier)
        if not candidates:
            raise ModelNotAvailableError(
                f"No hay modelos definidos para {src}→{tgt}"
                + (f" con tier='{tier}'" if tier else "")
            )
        if not tier:
            candidates.sort(
                key=lambda mid: TIER_PREFERENCE.get(
                    self._catalog[mid].get("tier", "balanced"), 99
                )
            )
        return candidates[0]

    # ---------------- Comprobación local ----------------

    def _get_local_path(self, model_id: str) -> Path:
        info = self.get_model_info(model_id)
        safe_name = info["hf_name"].replace("/", "__")
        return MODELS_BASE_DIR / safe_name

    def is_model_downloaded(self, model_id: str) -> bool:
        local_path = self._get_local_path(model_id)
        return local_path.exists() and (local_path / "config.json").exists()

    # ---------------- Descarga (solo desde CLI) ----------------

    def download_model(self, model_id: str) -> None:
        info = self.get_model_info(model_id)
        hf_name = info["hf_name"]
        local_path = self._get_local_path(model_id)

        if self.is_model_downloaded(model_id):
            print(f"✅ '{model_id}' ya está descargado en {local_path}")
            return

        print(f"⬇️ Descargando '{hf_name}' en {local_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
            tokenizer.save_pretrained(str(local_path))
            model.save_pretrained(str(local_path))
            print(f"✅ '{model_id}' descargado correctamente.")
        except Exception as e:
            raise ModelLoadError(f"Error al descargar '{model_id}': {e}")

    def download_all(self) -> None:
        for mid in self._catalog:
            try:
                self.download_model(mid)
            except Exception as e:
                print(f"❌ Error con '{mid}': {e}")

    # ---------------- Carga (interno, asume lock tomado) ----------------

    def _unload_locked(self) -> None:
        if self._model is not None:
            print(f"🧹 Liberando '{self._current_model_id}' de memoria...")
            self._model = None
            self._tokenizer = None
            self._current_model_id = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_model_loaded_locked(self, model_id: str) -> None:
        """Asume lock tomado. Nunca descarga: falla si no está local."""
        if self._current_model_id == model_id:
            return

        if not self.is_model_downloaded(model_id):
            raise ModelNotAvailableError(
                f"'{model_id}' no está descargado localmente. "
                f"Usa el CLI: python -m app.core.model_manager download {model_id}"
            )

        self._unload_locked()

        local_path = self._get_local_path(model_id)
        print(f"📥 Cargando '{model_id}' desde {local_path}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(local_path))
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(local_path))
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._current_model_id = model_id
            print(f"✅ '{model_id}' listo.")
        except Exception as e:
            self._model = None
            self._tokenizer = None
            self._current_model_id = None
            raise ModelLoadError(f"Error al cargar '{model_id}': {e}")

    # ---------------- Inferencia ----------------

    def translate(
        self,
        model_id: str,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Operación atómica: lock → asegurar modelo cargado → traducir.
        Otra petición (incluso de otro modelo) esperará a que termine.
        """
        with self._lock:
            self._ensure_model_loaded_locked(model_id)
            info = self.get_model_info(model_id)
            family = info.get("family", "opus-mt")

            if family == "nllb":
                return self._translate_nllb(info, text, source_lang, target_lang, **kwargs)
            return self._translate_generic(info, text, **kwargs)

    def _translate_generic(self, info: dict, text: str, **kwargs) -> str:
        """Para OPUS-MT y similares: tokenizar → generate → decode."""
        max_length = kwargs.get("max_length") or info.get("default_max_length", 512)
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_length=max_length)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_nllb(
        self,
        info: dict,
        text: str,
        source_lang: Optional[str],
        target_lang: Optional[str],
        **kwargs,
    ) -> str:
        """Para NLLB: fija src_lang en tokenizer y forced_bos_token_id en generate."""
        if not source_lang or not target_lang:
            raise ModelLoadError(
                "El modelo NLLB requiere 'source_lang' y 'target_lang' en la petición."
            )
        lang_map = info.get("language_code_map", {})
        src_code = lang_map.get(source_lang)
        tgt_code = lang_map.get(target_lang)
        if not src_code or not tgt_code:
            raise ModelLoadError(
                f"Par {source_lang}→{target_lang} no soportado por '{self._current_model_id}'."
            )

        max_length = kwargs.get("max_length") or info.get("default_max_length", 512)
        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def current_model_id(self) -> Optional[str]:
        return self._current_model_id


# ---------------- CLI ----------------

def _cli() -> None:
    import sys

    manager = ModelManager()
    args = sys.argv[1:]

    if not args:
        print("Uso: python -m app.core.model_manager <comando> [args]")
        print("Comandos:")
        print("  list                     Lista todos los modelos del catálogo")
        print("  pairs                    Lista pares de idiomas cubiertos")
        print("  status                   Estado del modelo cargado y descargas")
        print("  download <model_id>      Descarga un modelo específico")
        print("  download-all             Descarga todos los modelos del catálogo")
        sys.exit(0)

    cmd = args[0]

    if cmd == "list":
        for m in manager.list_models():
            downloaded = "✅" if manager.is_model_downloaded(m["model_id"]) else "⬇️"
            print(f"  {downloaded} {m['model_id']:35s} [{m.get('tier','?'):7s}] {m['description']}")
    elif cmd == "pairs":
        for p in manager.list_pairs():
            print(f"  {p['source_lang']}→{p['target_lang']}: {', '.join(p['models'])}")
    elif cmd == "status":
        print(f"Modelo en memoria: {manager.current_model_id or 'Ninguno'}")
        for m in manager.list_models():
            status = "✅ descargado" if manager.is_model_downloaded(m["model_id"]) else "⬇️ no descargado"
            print(f"  {m['model_id']:35s} {status}")
    elif cmd == "download":
        if len(args) < 2:
            print("Error: se requiere <model_id>")
            sys.exit(1)
        manager.download_model(args[1])
    elif cmd == "download-all":
        manager.download_all()
    else:
        print(f"Comando desconocido: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
"""Fixtures compartidas. No necesitan torch ni transformers: usan un tokenizador y un modelo falsos."""

import copy
import json
import time
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, parse_catalog
from app.core.model_manager import COMPLETE_MARKER, LoadedModel, ModelManager

RAW_CATALOG = json.loads((PROJECT_ROOT / "model_info.json").read_text(encoding="utf-8"))


class FakeTokenizer:
    """Cuenta 1 token por palabra y conoce los códigos de idioma de NLLB."""

    model_max_length = 512
    unk_token_id = 3
    src_lang: str | None = None
    _codes = {"eng_Latn": 10, "spa_Latn": 11, "fra_Latn": 12, "deu_Latn": 13}

    def convert_tokens_to_ids(self, code: str) -> int:
        return self._codes.get(code, self.unk_token_id)


class FakeManager(ModelManager):
    """ModelManager sin torch: 'traduce' poniendo el texto en mayúsculas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loads: list[str] = []
        self.batches: list[list[str]] = []
        self.generate_kwargs: list[dict] = []
        self.delay = 0.0

    def _load_pretrained(self, local_path: Path) -> LoadedModel:
        self.loads.append(local_path.name)
        return LoadedModel(model=object(), tokenizer=FakeTokenizer(), device="cpu")

    def _count_tokens(self, loaded: LoadedModel, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]

    def _generate_batch(self, loaded, texts, *, max_new_tokens, forced_bos_token_id):
        self.batches.append(list(texts))
        self.generate_kwargs.append(
            {
                "max_new_tokens": max_new_tokens,
                "forced_bos_token_id": forced_bos_token_id,
                "src_lang": loaded.tokenizer.src_lang,
            }
        )
        if self.delay:
            time.sleep(self.delay)
        return [(t.upper(), "TRUNC" in t) for t in texts]


@pytest.fixture
def catalog() -> dict[str, dict]:
    return parse_catalog(copy.deepcopy(RAW_CATALOG))


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


def mark_downloaded(manager: ModelManager, model_id: str) -> None:
    path = manager._get_local_path(model_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"0")
    (path / COMPLETE_MARKER).write_text("{}", encoding="utf-8")


@pytest.fixture
def manager(catalog, models_dir) -> FakeManager:
    mgr = FakeManager(catalog=catalog, models_dir=models_dir, max_loaded=1)
    for model_id in catalog:
        mark_downloaded(mgr, model_id)
    return mgr

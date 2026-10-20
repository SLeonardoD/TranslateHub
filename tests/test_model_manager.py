from dataclasses import replace

import pytest

import app.core.model_manager as mm
from app.core.exceptions import (
    InferenceError,
    InvalidRequestError,
    ModelLoadError,
    ModelNotAvailableError,
)
from app.core.model_manager import COMPLETE_MARKER
from tests.conftest import FakeManager, mark_downloaded

# ---------- Descarga: el marcador decide ----------


def test_files_without_marker_do_not_count_as_downloaded(catalog, models_dir):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    path = mgr._get_local_path("opus-mt-en-es")
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")  # lo que antes bastaba para "descargado"
    assert not mgr.is_model_downloaded("opus-mt-en-es")
    assert mgr.get_download_state("opus-mt-en-es") == "incomplete"
    assert mgr.get_download_state("opus-mt-es-en") == "missing"
    with pytest.raises(ModelNotAvailableError, match="incompleta"):
        mgr.translate_batch("opus-mt-en-es", ["hi"])


def test_marker_marks_as_downloaded(manager):
    assert manager.is_model_downloaded("opus-mt-en-es")
    assert manager.get_download_state("opus-mt-en-es") == "ok"


def test_ignore_patterns_prefer_safetensors():
    with_st = mm._ignore_patterns(
        ["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5"]
    )
    assert "*.bin" in with_st and "*.h5" in with_st
    without_st = mm._ignore_patterns(["config.json", "pytorch_model.bin", "tf_model.h5"])
    assert "*.bin" not in without_st and "*.h5" in without_st


def test_verify_download_requires_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ModelLoadError, match="pesos"):
        mm.ModelManager._verify_download(tmp_path, "x")
    (tmp_path / "pytorch_model.bin").write_bytes(b"0")
    mm.ModelManager._verify_download(tmp_path, "x")


# ---------- Carga y LRU ----------


def test_lazy_loading_and_reuse(manager):
    assert manager.loaded_model_ids == []
    manager.translate_batch("opus-mt-en-es", ["hello"])
    manager.translate_batch("opus-mt-en-es", ["again"])
    assert manager.loads == ["Helsinki-NLP__opus-mt-en-es"]  # una sola carga
    assert manager.current_model_id == "opus-mt-en-es"


def test_default_keeps_one_model_in_memory(manager):
    manager.translate_batch("opus-mt-en-es", ["a"])
    manager.translate_batch("opus-mt-es-en", ["b"])
    assert manager.loaded_model_ids == ["opus-mt-es-en"]


def test_lru_keeps_the_most_recent_models(catalog, models_dir):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir, max_loaded=2)
    for mid in catalog:
        mark_downloaded(mgr, mid)
    mgr.translate_batch("opus-mt-en-es", ["a"])
    mgr.translate_batch("opus-mt-es-en", ["b"])
    mgr.translate_batch("opus-mt-en-es", ["c"])  # en-es pasa a ser el más reciente
    mgr.translate_batch("nllb-200-distilled-600M", ["d"], "en", "fr")  # expulsa a es-en
    assert mgr.loaded_model_ids == ["opus-mt-en-es", "nllb-200-distilled-600M"]
    assert mgr.loads.count("Helsinki-NLP__opus-mt-en-es") == 1


def test_load_failure_is_wrapped_and_leaves_no_state(manager, monkeypatch):
    def boom(path):
        raise OSError("disco roto")

    monkeypatch.setattr(manager, "_load_pretrained", boom)
    with pytest.raises(ModelLoadError, match="disco roto"):
        manager.translate_batch("opus-mt-en-es", ["x"])
    assert manager.loaded_model_ids == []


def test_preload(manager):
    manager.preload("opus-mt-en-es")
    assert manager.is_model_loaded("opus-mt-en-es")


# ---------- Inferencia ----------


def test_translation_result(manager):
    [item] = manager.translate_batch("opus-mt-en-es", ["hello world"])
    assert item.text == "HELLO WORLD"
    assert item.chunks == 1 and not item.truncated and item.warnings == []


@pytest.fixture
def small_chunks(monkeypatch):
    """Fragmentos de 16 palabras y lotes de 2, para forzar el troceado con textos cortos."""
    monkeypatch.setattr(mm, "settings", replace(mm.settings, chunk_target_tokens=16, batch_size=2))


def test_long_text_is_chunked_and_reassembled_in_order(manager, small_chunks):
    sentences = [f"Sentence number {i} is here." for i in range(12)]  # 5 palabras cada una
    text = " ".join(sentences)
    [item] = manager.translate_batch("opus-mt-en-es", [text])
    assert item.chunks == 4  # 3 oraciones (15 palabras) por fragmento
    assert item.text == text.upper()  # nada se pierde y el orden se conserva
    assert all(len(b) <= 2 for b in manager.batches)  # respeta batch_size


def test_newlines_survive_translation(manager):
    text = "first line.\nsecond line.\n\nthird."
    [item] = manager.translate_batch("opus-mt-en-es", [text])
    assert item.text == text.upper()


def test_batch_keeps_order_of_texts(manager, small_chunks):
    texts = ["short one", "a much longer text " * 6, "mid sized text here"]
    items = manager.translate_batch("opus-mt-en-es", texts)
    assert [i.text for i in items] == [t.upper() for t in texts]


def test_truncated_output_is_reported_not_silent(manager):
    [item] = manager.translate_batch("opus-mt-en-es", ["this will TRUNC here"], max_length=7)
    assert item.truncated
    assert "max_length=7" in item.warnings[0]
    assert manager.generate_kwargs[-1]["max_new_tokens"] == 7


def test_overlong_sentence_triggers_warning(manager, small_chunks):
    sentence = " ".join(f"w{i}" for i in range(60)) + "."
    [item] = manager.translate_batch("opus-mt-en-es", [sentence])
    assert item.chunks > 1
    assert "divid" in item.warnings[0]


def test_default_max_length_comes_from_catalog(manager):
    manager.translate_batch("opus-mt-en-es", ["hi"])
    assert manager.generate_kwargs[-1]["max_new_tokens"] == 512


def test_nllb_sets_source_language_and_forced_bos(manager):
    manager.translate_batch("nllb-200-distilled-600M", ["hello"], "en", "fr")
    kwargs = manager.generate_kwargs[-1]
    assert kwargs["src_lang"] == "eng_Latn"
    assert kwargs["forced_bos_token_id"] == 12  # fra_Latn


def test_nllb_without_languages_is_a_client_error(manager):
    with pytest.raises(InvalidRequestError, match="source_lang"):
        manager.translate_batch("nllb-200-distilled-600M", ["hello"])


def test_nllb_unsupported_language_is_a_client_error(manager):
    with pytest.raises(InvalidRequestError, match="no soportado"):
        manager.translate_batch("nllb-200-distilled-600M", ["hello"], "en", "xx")


def test_inference_failure_becomes_inference_error(manager, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(manager, "_generate_batch", boom)
    with pytest.raises(InferenceError, match="out of memory"):
        manager.translate_batch("opus-mt-en-es", ["x"])
    # el lock se liberó: la siguiente petición funciona
    monkeypatch.undo()
    assert manager.translate_batch("opus-mt-en-es", ["ok"])[0].text == "OK"


# ---------- Catálogo ----------


def test_resolve_prefers_fast_by_default(manager):
    assert manager.resolve_model_id("en", "es") == "opus-mt-en-es"
    assert manager.resolve_model_id("en", "es", "quality") == "nllb-200-distilled-600M"
    assert manager.resolve_model_id("en", "fr") == "nllb-200-distilled-600M"


def test_resolve_unknown_pair(manager):
    with pytest.raises(ModelNotAvailableError, match="en→ja"):
        manager.resolve_model_id("en", "ja")


def test_list_pairs(manager):
    pairs = {(p["source_lang"], p["target_lang"]): p["models"] for p in manager.list_pairs()}
    assert pairs[("en", "es")] == ["nllb-200-distilled-600M", "opus-mt-en-es"]
    assert pairs[("en", "de")] == ["nllb-200-distilled-600M"]


# ---------- Flujo de descarga (con huggingface_hub simulado) ----------


@pytest.fixture
def fake_hub(monkeypatch):
    """Simula huggingface_hub: lista de archivos del repo y una descarga que puede fallar."""
    import sys
    import types

    state = {"fail_after_config": False, "downloads": 0, "ignore": None}

    class HfApi:
        def list_repo_files(self, repo_id):
            return ["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5"]

    def snapshot_download(repo_id, local_dir, cache_dir=None, ignore_patterns=None):
        state["downloads"] += 1
        state["ignore"] = ignore_patterns
        state["cache_dir"] = cache_dir
        from pathlib import Path

        (Path(local_dir) / "config.json").write_text("{}")  # llega antes que los pesos
        if state["fail_after_config"]:
            raise ConnectionError("red caída a mitad de descarga")
        (Path(local_dir) / "model.safetensors").write_bytes(b"0")

    module = types.ModuleType("huggingface_hub")
    module.HfApi = HfApi
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return state


def test_download_writes_marker_only_at_the_end(catalog, models_dir, fake_hub):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    mgr.download_model("opus-mt-en-es")
    assert mgr.is_model_downloaded("opus-mt-en-es")
    assert "*.bin" in fake_hub["ignore"]  # hay safetensors: se omiten los .bin duplicados


def test_interrupted_download_is_not_reported_as_downloaded(catalog, models_dir, fake_hub):
    """El bug original: config.json ya existía, así que el modelo figuraba como descargado."""
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    fake_hub["fail_after_config"] = True
    with pytest.raises(ModelLoadError, match="red caída"):
        mgr.download_model("opus-mt-en-es")
    assert not mgr.is_model_downloaded("opus-mt-en-es")
    assert mgr.get_download_state("opus-mt-en-es") == "incomplete"

    # Volver a ejecutar 'download' lo repara (antes respondía "ya está descargado")
    fake_hub["fail_after_config"] = False
    mgr.download_model("opus-mt-en-es")
    assert mgr.is_model_downloaded("opus-mt-en-es")


def test_download_is_skipped_when_complete_unless_forced(catalog, models_dir, fake_hub):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    mgr.download_model("opus-mt-en-es")
    mgr.download_model("opus-mt-en-es")
    assert fake_hub["downloads"] == 1
    mgr.download_model("opus-mt-en-es", force=True)
    assert fake_hub["downloads"] == 2


def test_download_without_weights_does_not_write_marker(catalog, models_dir, fake_hub, monkeypatch):
    import huggingface_hub

    def only_config(repo_id, local_dir, cache_dir=None, ignore_patterns=None):
        from pathlib import Path

        (Path(local_dir) / "config.json").write_text("{}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", only_config)
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    with pytest.raises(ModelLoadError, match="pesos"):
        mgr.download_model("opus-mt-en-es")
    assert not mgr.is_model_downloaded("opus-mt-en-es")


def test_download_all_reports_failures(catalog, models_dir, fake_hub):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    fake_hub["fail_after_config"] = True
    assert mgr.download_all() == 3


# ---------- Detección de truncado (lógica pura) ----------


def test_truncated_flags_batch_finished_early_means_nothing_truncated():
    # el lote terminó antes de agotar el presupuesto: ninguna secuencia se cortó
    assert mm.truncated_flags([0, 59513], output_len=10, max_new_tokens=512, pad_id=59513) == [
        False,
        False,
    ]


def test_truncated_flags_distinguishes_rows_that_finished_from_rows_that_were_cut():
    # presupuesto agotado (len = max+1): la fila con pad terminó sola; la otra seguía activa
    assert mm.truncated_flags([59513, 0, 123], output_len=9, max_new_tokens=8, pad_id=59513) == [
        False,
        True,  # EOS forzado en la última posición (caso Marian) => cortada
        True,
    ]


def test_truncated_flags_without_pad_id():
    assert mm.truncated_flags([5], output_len=4, max_new_tokens=3, pad_id=None) == [True]


# ---------- Solo modelos locales y sin caché global ----------


def test_removed_weights_make_the_model_unavailable(manager):
    manager._get_local_path("opus-mt-en-es").joinpath("model.safetensors").unlink()
    assert not manager.is_model_downloaded("opus-mt-en-es")
    with pytest.raises(ModelNotAvailableError):
        manager.translate_batch("opus-mt-en-es", ["x"])


def test_local_only_listing_and_resolution(manager):
    (manager._get_local_path("opus-mt-en-es") / COMPLETE_MARKER).unlink()
    assert [m["model_id"] for m in manager.list_models(local_only=True)] == [
        "opus-mt-es-en",
        "nllb-200-distilled-600M",
    ]
    assert len(manager.list_models()) == 3  # el catálogo completo sigue disponible para el CLI
    assert manager.resolve_model_id("en", "es", local_only=True) == "nllb-200-distilled-600M"
    assert manager.resolve_model_id("en", "es") == "opus-mt-en-es"


def test_local_only_error_suggests_the_download(manager):
    (manager._get_local_path("opus-mt-en-es") / COMPLETE_MARKER).unlink()
    (manager._get_local_path("nllb-200-distilled-600M") / COMPLETE_MARKER).unlink()
    with pytest.raises(ModelNotAvailableError, match="download opus-mt-en-es"):
        manager.resolve_model_id("en", "es", local_only=True)
    with pytest.raises(ModelNotAvailableError, match="No hay modelos definidos"):
        manager.resolve_model_id("en", "ja", local_only=True)


def test_download_goes_straight_to_models_dir_and_leaves_no_cache(catalog, models_dir, fake_hub):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    mgr.download_model("opus-mt-en-es")
    # cualquier caché temporal vive dentro de models/ (nunca en ~/.cache) y se borra al terminar
    assert str(fake_hub["cache_dir"]).startswith(str(models_dir))
    assert not (models_dir / ".hf_tmp").exists()
    assert (models_dir / "Helsinki-NLP__opus-mt-en-es" / "model.safetensors").exists()


def test_failed_download_also_cleans_temp_cache(catalog, models_dir, fake_hub):
    mgr = FakeManager(catalog=catalog, models_dir=models_dir)
    fake_hub["fail_after_config"] = True
    with pytest.raises(ModelLoadError):
        mgr.download_model("opus-mt-en-es")
    assert not (models_dir / ".hf_tmp").exists()

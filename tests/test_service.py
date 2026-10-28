import pytest

from app.core.exceptions import InvalidRequestError, ModelNotAvailableError
from app.services import translation


@pytest.fixture(autouse=True)
def _use_fake_manager(manager, monkeypatch):
    monkeypatch.setattr(translation, "model_manager", manager)


def test_resolves_by_pair_and_fills_metadata():
    r = translation.translate_text("hello", source_lang="en", target_lang="es")
    assert r["model_id"] == "opus-mt-en-es"
    assert (r["source_lang"], r["target_lang"]) == ("en", "es")
    assert r["translated_text"] == "HELLO"
    assert r["chunks"] == 1 and r["truncated"] is False and r["warnings"] == []


def test_tier_quality_selects_nllb(manager):
    r = translation.translate_text("hello", source_lang="en", target_lang="es", tier="quality")
    assert r["model_id"] == "nllb-200-distilled-600M"


def test_language_codes_are_normalized():
    r = translation.translate_text("hola", source_lang=" ES-es ", target_lang="EN")
    assert r["model_id"] == "opus-mt-es-en"


def test_pair_model_rejects_wrong_source():
    with pytest.raises(ModelNotAvailableError, match="es→en"):
        translation.translate_text("x", model_id="opus-mt-es-en", source_lang="en")


def test_pair_model_rejects_wrong_target():
    """Antes se ignoraba en silencio y se traducía a otro idioma."""
    with pytest.raises(ModelNotAvailableError, match="en→fr"):
        translation.translate_text("x", model_id="opus-mt-en-es", target_lang="fr")


def test_pair_model_without_languages_uses_native_pair():
    r = translation.translate_text("x", model_id="opus-mt-en-es")
    assert (r["source_lang"], r["target_lang"]) == ("en", "es")


def test_multilingual_requires_languages_is_client_error():
    with pytest.raises(InvalidRequestError, match="multilingüe"):
        translation.translate_text("x", model_id="nllb-200-distilled-600M")


def test_multilingual_unsupported_pair():
    with pytest.raises(ModelNotAvailableError, match="no soporta"):
        translation.translate_text(
            "x", model_id="nllb-200-distilled-600M", source_lang="en", target_lang="ja"
        )


def test_unknown_model():
    with pytest.raises(ModelNotAvailableError, match="no definido"):
        translation.translate_text("x", model_id="nope")


def test_needs_model_or_pair():
    with pytest.raises(InvalidRequestError):
        translation.translate_text("x")


def test_not_downloaded_model_is_reported(manager):
    (manager._get_local_path("opus-mt-en-es") / ".download_complete").unlink()
    with pytest.raises(ModelNotAvailableError, match="download opus-mt-en-es"):
        translation.translate_text("x", model_id="opus-mt-en-es")


def test_batch_translation_with_metadata():
    r = translation.translate_texts(["uno", "dos"], source_lang="es", target_lang="en")
    assert [x["translated_text"] for x in r["results"]] == ["UNO", "DOS"]
    assert r["model_id"] == "opus-mt-es-en"


def test_models_status_marks_loaded(manager):
    manager.translate_batch("opus-mt-en-es", ["x"])
    status = {m["model_id"]: m for m in translation.get_models_status()}
    assert status["opus-mt-en-es"]["loaded"] and status["opus-mt-en-es"]["downloaded"]
    assert not status["opus-mt-es-en"]["loaded"]


# ---------- Cancelación (cliente desconectado) ----------


def test_cancelled_before_start_never_loads_the_model(manager):
    import threading

    import pytest

    from app.core.exceptions import ClientDisconnectedError

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ClientDisconnectedError):
        manager.translate_batch("opus-mt-en-es", ["hola"], cancel=cancel)
    assert manager.loads == [] and manager.batches == []


def test_cancelled_between_batches_stops_the_work(manager):
    import threading

    import pytest

    from app.core.exceptions import ClientDisconnectedError

    cancel = threading.Event()
    original = manager._generate_batch

    def generate_then_disconnect(*args, **kwargs):
        result = original(*args, **kwargs)
        cancel.set()  # el cliente se va durante el primer lote
        return result

    manager._generate_batch = generate_then_disconnect
    texts = [f"texto numero {i}" for i in range(30)]  # varios lotes (batch_size = 8)
    with pytest.raises(ClientDisconnectedError):
        manager.translate_batch("opus-mt-en-es", texts, cancel=cancel)
    assert len(manager.batches) == 1

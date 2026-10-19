"""Tests de ModelManager: catálogo, descargas, carga e inferencia (todo mockeado)."""
import pytest

from app.core.exceptions import ModelNotAvailableError, ModelLoadError


# ------------------------------------------------------------------
# Catálogo / consultas
# ------------------------------------------------------------------

def test_get_model_info_unknown_raises(manager):
    with pytest.raises(ModelNotAvailableError, match="no definido"):
        manager.get_model_info("no-existe")


def test_get_model_info_known(manager):
    info = manager.get_model_info("opus-mt-en-es")
    assert info["source_lang"] == "en"
    assert info["target_lang"] == "es"


def test_list_models(manager):
    models = manager.list_models()
    assert len(models) == 3
    ids = {m["model_id"] for m in models}
    assert ids == {"opus-mt-en-es", "opus-mt-es-en", "nllb-200-distilled-600M"}


def test_list_pairs_agrupa_por_par(manager):
    pairs = manager.list_pairs()
    indexed = {(p["source_lang"], p["target_lang"]): set(p["models"]) for p in pairs}
    assert indexed[("en", "es")] == {"opus-mt-en-es", "nllb-200-distilled-600M"}
    assert indexed[("es", "en")] == {"opus-mt-es-en", "nllb-200-distilled-600M"}
    assert indexed[("en", "fr")] == {"nllb-200-distilled-600M"}


def test_model_supports_pair_tipo_pair(manager):
    info = manager.get_model_info("opus-mt-en-es")
    assert manager.model_supports_pair(info, "en", "es") is True
    assert manager.model_supports_pair(info, "es", "en") is False


def test_model_supports_pair_multilingue(manager):
    info = manager.get_model_info("nllb-200-distilled-600M")
    assert manager.model_supports_pair(info, "en", "fr") is True
    assert manager.model_supports_pair(info, "fr", "en") is False


def test_get_models_for_pair(manager):
    assert set(manager.get_models_for_pair("en", "es")) == {
        "opus-mt-en-es", "nllb-200-distilled-600M"
    }


def test_get_models_for_pair_con_tier(manager):
    assert manager.get_models_for_pair("en", "es", tier="fast") == ["opus-mt-en-es"]
    assert manager.get_models_for_pair("en", "es", tier="quality") == ["nllb-200-distilled-600M"]


def test_resolve_model_id_prefiere_fast(manager):
    # Sin tier, la preferencia es "fast" < "balanced" < "quality"
    assert manager.resolve_model_id("en", "es") == "opus-mt-en-es"


def test_resolve_model_id_con_tier_quality(manager):
    assert manager.resolve_model_id("en", "fr", tier="quality") == "nllb-200-distilled-600M"


def test_resolve_model_id_sin_candidatos(manager):
    with pytest.raises(ModelNotAvailableError, match="No hay modelos"):
        manager.resolve_model_id("de", "ja")


def test_resolve_model_id_tier_inexistente_para_par(manager):
    with pytest.raises(ModelNotAvailableError):
        manager.resolve_model_id("en", "fr", tier="fast")


# ------------------------------------------------------------------
# Comprobación local
# ------------------------------------------------------------------

def test_is_model_downloaded_false(manager):
    assert manager.is_model_downloaded("opus-mt-en-es") is False


def test_is_model_downloaded_true(manager, models_dir, sample_catalog):
    info = sample_catalog["opus-mt-en-es"]
    d = models_dir / info["hf_name"].replace("/", "__")
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert manager.is_model_downloaded("opus-mt-en-es") is True


def test_get_local_path_sanitiza_hf_name(manager, models_dir):
    path = manager._get_local_path("nllb-200-distilled-600M")
    assert path.parent == models_dir
    assert path.name == "facebook__nllb-200-distilled-600M"


# ------------------------------------------------------------------
# Descarga (transformers mockeado)
# ------------------------------------------------------------------

def test_download_model_crea_archivos(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")
    assert manager.is_model_downloaded("opus-mt-en-es")
    mock_transformers["tok_class"].from_pretrained.assert_called_once_with(
        "Helsinki-NLP/opus-mt-en-es"
    )
    mock_transformers["mdl_class"].from_pretrained.assert_called_once_with(
        "Helsinki-NLP/opus-mt-en-es"
    )


def test_download_model_skips_si_ya_existe(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")
    assert mock_transformers["tok_class"].from_pretrained.call_count == 1

    manager.download_model("opus-mt-en-es")  # segunda vez
    assert mock_transformers["tok_class"].from_pretrained.call_count == 1


def test_download_all_descarga_todo(manager, mock_transformers):
    manager.download_all()
    for mid in ("opus-mt-en-es", "opus-mt-es-en", "nllb-200-distilled-600M"):
        assert manager.is_model_downloaded(mid)


def test_download_model_unknown_raises(manager):
    with pytest.raises(ModelNotAvailableError):
        manager.download_model("no-existe")


# ------------------------------------------------------------------
# Carga en memoria (transformers mockeado)
# ------------------------------------------------------------------

def test_ensure_loaded_falla_si_no_descargado(manager, mock_transformers):
    with pytest.raises(ModelNotAvailableError, match="no está descargado"):
        manager._ensure_model_loaded_locked("opus-mt-en-es")


def test_ensure_loaded_carga_modelo(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")
    manager._ensure_model_loaded_locked("opus-mt-en-es")
    assert manager.current_model_id == "opus-mt-en-es"
    assert mock_transformers["tok_class"].from_pretrained.call_count == 2  # download + load


def test_ensure_loaded_mismo_modelo_no_recarga(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")
    manager._ensure_model_loaded_locked("opus-mt-en-es")
    assert mock_transformers["tok_class"].from_pretrained.call_count == 2

    manager._ensure_model_loaded_locked("opus-mt-en-es")
    assert mock_transformers["tok_class"].from_pretrained.call_count == 2  # sin recarga


def test_ensure_loaded_cambia_modelo_libera_anterior(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")
    manager.download_model("opus-mt-es-en")

    manager._ensure_model_loaded_locked("opus-mt-en-es")
    assert manager.current_model_id == "opus-mt-en-es"

    manager._ensure_model_loaded_locked("opus-mt-es-en")
    assert manager.current_model_id == "opus-mt-es-en"
    # 4 llamadas = 2 downloads + 2 loads
    assert mock_transformers["tok_class"].from_pretrained.call_count == 4


# ------------------------------------------------------------------
# Inferencia (tokenizer y model mockeados)
# ------------------------------------------------------------------

def test_translate_generic(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")

    # Comportamiento del tokenizer y del model en inferencia
    mock_transformers["tok"].return_value = {"input_ids": [[1, 2]]}
    mock_transformers["tok"].decode.return_value = "Hola mundo"
    mock_transformers["mdl"].generate.return_value = [[1, 2, 3]]

    result = manager.translate("opus-mt-en-es", "Hello world")

    assert result == "Hola mundo"
    # generate fue llamado con max_length del catálogo
    kwargs = mock_transformers["mdl"].generate.call_args.kwargs
    assert kwargs["max_length"] == 512


def test_translate_generic_respeta_max_length_custom(manager, mock_transformers):
    manager.download_model("opus-mt-en-es")
    mock_transformers["tok"].return_value = {"input_ids": [[1, 2]]}
    mock_transformers["tok"].decode.return_value = "x"
    mock_transformers["mdl"].generate.return_value = [[1]]

    manager.translate("opus-mt-en-es", "Hello", max_length=42)
    kwargs = mock_transformers["mdl"].generate.call_args.kwargs
    assert kwargs["max_length"] == 42


def test_translate_nllb(manager, mock_transformers):
    manager.download_model("nllb-200-distilled-600M")

    mock_transformers["tok"].return_value = {"input_ids": [[1, 2]]}
    mock_transformers["tok"].decode.return_value = "Bonjour le monde"
    mock_transformers["tok"].convert_tokens_to_ids.return_value = 256057
    mock_transformers["mdl"].generate.return_value = [[1, 2, 3]]

    result = manager.translate(
        "nllb-200-distilled-600M", "Hello world",
        source_lang="en", target_lang="fr",
    )

    assert result == "Bonjour le monde"
    # El tokenizer recibió el src_lang ISO traducido a código interno
    assert mock_transformers["tok"].src_lang == "eng_Latn"
    # generate recibió el forced_bos_token_id del idioma destino
    kwargs = mock_transformers["mdl"].generate.call_args.kwargs
    assert kwargs["forced_bos_token_id"] == 256057


def test_translate_nllb_requiere_langs(manager, mock_transformers):
    manager.download_model("nllb-200-distilled-600M")
    with pytest.raises(ModelLoadError, match="requiere"):
        manager.translate("nllb-200-distilled-600M", "Hello")


def test_translate_nllb_par_no_soportado(manager, mock_transformers):
    manager.download_model("nllb-200-distilled-600M")
    with pytest.raises(ModelLoadError, match="no soportado"):
        manager.translate(
            "nllb-200-distilled-600M", "Hello",
            source_lang="en", target_lang="ja",  # ja no está en language_code_map
        )


def test_translate_modelo_no_descargado_nunca_descarga(manager, mock_transformers):
    """La API jamás debe disparar una descarga."""
    with pytest.raises(ModelNotAvailableError):
        manager.translate("opus-mt-en-es", "Hello")
    # from_pretrained no debe haberse llamado en absoluto
    mock_transformers["tok_class"].from_pretrained.assert_not_called()
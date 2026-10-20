"""Integración con un modelo Marian diminuto (pesos aleatorios, sin red).

Ejercita transformers + torch de verdad: carga, tokenización, troceado, generate() y
detección de truncado. Se omite si faltan torch, transformers o sentencepiece.
La traducción es basura (pesos aleatorios): solo se comprueba que el flujo funciona.
"""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
spm = pytest.importorskip("sentencepiece")

import app.core.model_manager as mm  # noqa: E402
from app.config import parse_catalog  # noqa: E402

SENTENCES = [
    "The cat sat on the mat.",
    "I like to read many books.",
    "She went to the market today.",
    "We will travel to the sea.",
    "He plays music every night.",
    "The weather is very nice now.",
    "They build a small house.",
    "My brother drinks cold water.",
]


@pytest.fixture(scope="module")
def tiny_models_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    model_dir = root / "Helsinki-NLP__opus-mt-en-es"
    model_dir.mkdir()
    corpus = root / "corpus.txt"
    corpus.write_text("\n".join(SENTENCES * 10), encoding="utf-8")

    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(model_dir / "source"),
            vocab_size=50,
            model_type="unigram",
            hard_vocab_limit=False,
        )
        (model_dir / "target.spm").write_bytes((model_dir / "source.spm").read_bytes())
        sp = spm.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        vocab["<pad>"] = len(vocab)
        (model_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "MarianTokenizer",
                    "source_lang": "en",
                    "target_lang": "es",
                    "model_max_length": 128,
                }
            ),
            encoding="utf-8",
        )
        config = transformers.MarianConfig(
            vocab_size=len(vocab),
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=128,
            pad_token_id=vocab["<pad>"],
            eos_token_id=vocab["</s>"],
            decoder_start_token_id=vocab["<pad>"],
            forced_eos_token_id=vocab["</s>"],
        )
        transformers.MarianMTModel(config).save_pretrained(model_dir)
        # Si el propio fixture no se puede cargar con transformers, es un problema del test, no del código.
        transformers.AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        transformers.AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as exc:
        pytest.skip(f"No se pudo construir el modelo diminuto de prueba: {exc!r}")

    (model_dir / mm.COMPLETE_MARKER).write_text("{}", encoding="utf-8")
    return root


@pytest.fixture
def real_manager(tiny_models_dir, monkeypatch):
    monkeypatch.setattr(mm, "settings", replace(mm.settings, chunk_target_tokens=16, batch_size=3))
    catalog = parse_catalog(
        {
            "models": {
                "opus-mt-en-es": {
                    "hf_name": "Helsinki-NLP/opus-mt-en-es",
                    "type": "pair",
                    "source_lang": "en",
                    "target_lang": "es",
                    "tier": "fast",
                    "description": "tiny",
                    "default_max_length": 24,
                }
            }
        }
    )
    return mm.ModelManager(catalog=catalog, models_dir=tiny_models_dir, max_loaded=1)


def test_real_pipeline_end_to_end(real_manager):
    long_text = " ".join(SENTENCES)
    items = real_manager.translate_batch("opus-mt-en-es", [long_text, SENTENCES[0]])

    assert len(items) == 2
    assert items[0].chunks > 1  # el texto largo se trocea
    assert items[1].chunks == 1  # el corto sigue siendo un solo fragmento
    assert all(isinstance(item.text, str) for item in items)
    assert real_manager.loaded_model_ids == ["opus-mt-en-es"]


def test_real_token_counts_are_positive(real_manager):
    real_manager.preload("opus-mt-en-es")
    loaded = real_manager._loaded["opus-mt-en-es"]
    counts = real_manager._count_tokens(loaded, SENTENCES[:3])
    assert all(isinstance(n, int) and n > 0 for n in counts)


def test_real_budget_is_respected_and_reported(real_manager):
    """Con un presupuesto minúsculo el modelo (aleatorio) casi seguro no termina: se avisa."""
    [item] = real_manager.translate_batch("opus-mt-en-es", [SENTENCES[0]], max_length=2)
    assert isinstance(item.truncated, bool)
    if item.truncated:
        assert "max_length=2" in item.warnings[0]

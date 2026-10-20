from app.core.text_utils import (
    assemble,
    build_chunks,
    fit_to_limit,
    is_sentence_end,
    split_units,
)


def words(texts):
    return [len(t.split()) for t in texts]


def roundtrip(text, target=5):
    seg = build_chunks(text, words, target)
    return seg, assemble(seg, [c.text for c in seg.chunks])


def test_short_text_is_a_single_chunk():
    seg, out = roundtrip("Hello world. How are you?", target=50)
    assert [c.text for c in seg.chunks] == ["Hello world. How are you?"]
    assert out == "Hello world. How are you?"


def test_long_text_is_packed_by_sentences():
    seg, out = roundtrip("One two three. Four five six. Seven eight nine.", target=6)
    assert [c.text for c in seg.chunks] == ["One two three. Four five six.", "Seven eight nine."]
    assert out == "One two three. Four five six. Seven eight nine."


def test_newlines_never_inside_a_chunk_and_are_restored():
    text = "Line one\nLine two\n\nParagraph two."
    seg, out = roundtrip(text, target=100)
    assert all("\n" not in c.text for c in seg.chunks)
    assert out == text


def test_leading_and_trailing_whitespace_preserved():
    _, out = roundtrip("  \n Hola mundo.  \n", target=100)
    assert out == "  \n Hola mundo.  \n"


def test_abbreviations_do_not_split_sentences():
    lead, units = split_units("Dr. Smith arrived. He sat down.")
    assert lead == ""
    assert [u.text for u in units] == ["Dr. Smith arrived.", "He sat down."]


def test_is_sentence_end():
    assert is_sentence_end("end.")
    assert is_sentence_end("¿qué?")
    assert is_sentence_end('dijo."')
    assert is_sentence_end("wait...")
    assert not is_sentence_end("Dr.")
    assert not is_sentence_end("J.")
    assert not is_sentence_end("3.")
    assert not is_sentence_end("word")


def test_overlong_sentence_is_split_by_words_and_flagged():
    sentence = " ".join(f"w{i}" for i in range(40)) + "."
    seg = build_chunks(sentence, words, target=10)
    assert seg.forced_splits == 1
    assert all(len(c.text.split()) <= 10 for c in seg.chunks)
    # no se pierde ni una palabra
    assert " ".join(c.text for c in seg.chunks).split() == sentence.split()


def test_unspaced_text_is_split_by_characters():
    parts = fit_to_limit("x" * 100, lambda xs: [len(x) for x in xs], 30)
    assert "".join(parts) == "x" * 100
    assert all(len(p) <= 30 for p in parts)


def test_whitespace_only_text_has_no_chunks():
    seg = build_chunks("   \n  ", words, 10)
    assert seg.chunks == []
    assert assemble(seg, []) == "   \n  "


def test_no_text_is_lost_in_long_mixed_text():
    text = "Primera frase. Segunda frase larga con varias palabras.\nTercera línea.\n\nCuarta: fin."
    seg, out = roundtrip(text, target=4)
    assert out.split() == text.split()


def test_trailing_whitespace_preserved_even_without_final_punctuation():
    _, out = roundtrip("hola mundo   ", target=100)
    assert out == "hola mundo   "

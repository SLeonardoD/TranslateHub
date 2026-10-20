"""Troceado de textos largos en fragmentos que caben en el modelo.

Funciones puras (sin torch): reciben un contador de tokens, así que se pueden
probar con cualquier tokenizador falso.

Reglas:
- Un texto corto sigue siendo UN solo fragmento (comportamiento idéntico al de siempre).
- Un texto largo se parte por oraciones y se agrupa hasta ``target`` tokens.
- Los saltos de línea nunca quedan dentro de un fragmento: se conservan al reensamblar.
- Una oración que por sí sola supera ``target`` se divide por palabras (se avisa de ello).
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

CountTokens = Callable[[Sequence[str]], list[int]]

_SENTENCE_ENDINGS = ".!?…"
_CLOSERS = "\"'”’»)]"
_OPENERS = "([\"'“‘«¿¡"
_WHITESPACE = re.compile(r"\s+")

# Abreviaturas cuyo punto NO cierra oración (es/en/fr/de). Si fallamos por exceso solo
# se fusionan dos oraciones en una unidad, lo cual es inofensivo.
_ABBREVIATIONS = frozenset(
    {
        "sr",
        "sra",
        "srta",
        "sres",
        "dr",
        "dra",
        "lic",
        "ing",
        "arq",
        "prof",
        "gral",
        "av",
        "avda",
        "pág",
        "págs",
        "ud",
        "uds",
        "vs",
        "cap",
        "núm",
        "tel",
        "aprox",
        "depto",
        "atte",
        "mr",
        "mrs",
        "ms",
        "jr",
        "st",
        "mt",
        "e.g",
        "i.e",
        "fig",
        "vol",
        "approx",
        "inc",
        "ltd",
        "corp",
        "dept",
        "pp",
        "mme",
        "mlle",
        "bzw",
        "usw",
        "z.b",
        "nr",
        "str",
        "etc",
    }
)


@dataclass(slots=True)
class Unit:
    """Oración (o línea) y el espacio en blanco que la seguía en el original."""

    text: str
    sep: str


@dataclass(slots=True)
class Chunk:
    """Fragmento que se envía al modelo y el separador a reponer tras traducirlo."""

    text: str
    sep: str


@dataclass(slots=True)
class Segmentation:
    lead: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    forced_splits: int = 0  # oraciones demasiado largas que hubo que partir por palabras


def is_sentence_end(word: str) -> bool:
    """¿La palabra (con su puntuación pegada) cierra una oración?"""
    core = word.rstrip(_CLOSERS)
    if not core or core[-1] not in _SENTENCE_ENDINGS:
        return False
    if core[-1] != "." or core.endswith(".."):
        return True
    stem = core.rstrip(".").lstrip(_OPENERS).lower()
    return not (stem in _ABBREVIATIONS or stem.isdigit() or (len(stem) == 1 and stem.isalpha()))


def split_units(text: str) -> tuple[str, list[Unit]]:
    """Parte el texto en oraciones/líneas. Devuelve (espacio inicial, unidades)."""
    lead = text[: len(text) - len(text.lstrip())]
    units: list[Unit] = []
    unit_start = word_start = len(lead)
    for match in _WHITESPACE.finditer(text):
        start, end = match.span()
        if end <= word_start:  # espacio inicial, ya contado en `lead`
            continue
        sep = match.group()
        if "\n" in sep or is_sentence_end(text[word_start:start]):
            units.append(Unit(text[unit_start:start], sep))
            unit_start = end
        word_start = end
    rest = text[unit_start:]
    if rest:
        body = rest.rstrip()  # el espacio final se conserva como separador
        units.append(Unit(body, rest[len(body) :]))
    return lead, units


def _split_middle(text: str) -> tuple[str, str]:
    """Divide en dos mitades, preferiblemente en el espacio más cercano al centro."""
    mid = len(text) // 2
    spaces = [m.start() for m in _WHITESPACE.finditer(text) if 0 < m.start() < len(text) - 1]
    if spaces:
        cut = min(spaces, key=lambda pos: abs(pos - mid))
        return text[:cut].rstrip(), text[cut:].lstrip()
    return text[:mid], text[mid:]


def fit_to_limit(text: str, count_tokens: CountTokens, limit: int) -> list[str]:
    """Divide ``text`` recursivamente hasta que cada parte tenga <= ``limit`` tokens."""
    if len(text) < 2 or count_tokens([text])[0] <= limit:
        return [text]
    left, right = _split_middle(text)
    return fit_to_limit(left, count_tokens, limit) + fit_to_limit(right, count_tokens, limit)


def build_chunks(text: str, count_tokens: CountTokens, target: int) -> Segmentation:
    """Trocea ``text`` en fragmentos de como máximo ``target`` tokens (salvo palabras gigantes)."""
    lead, units = split_units(text)
    seg = Segmentation(lead=lead)
    if not units:
        return seg

    counts = count_tokens([u.text for u in units])
    current: list[Unit] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            body = "".join(u.text + u.sep for u in current[:-1]) + current[-1].text
            seg.chunks.append(Chunk(body, current[-1].sep))
        current, current_tokens = [], 0

    for unit, n_tokens in zip(units, counts, strict=True):
        if n_tokens > target:
            flush()
            parts = fit_to_limit(unit.text, count_tokens, target)
            if len(parts) > 1:
                seg.forced_splits += 1
            for i, part in enumerate(parts):
                is_last = i == len(parts) - 1
                seg.chunks.append(Chunk(part, unit.sep if is_last else " "))
            continue
        if current and current_tokens + n_tokens > target:
            flush()
        current.append(unit)
        current_tokens += n_tokens
        if "\n" in unit.sep:
            flush()
    flush()
    return seg


def assemble(seg: Segmentation, translations: Sequence[str]) -> str:
    """Une las traducciones de los fragmentos reponiendo espacios y saltos de línea."""
    if len(translations) != len(seg.chunks):
        raise ValueError("Número de traducciones distinto al de fragmentos.")
    return seg.lead + "".join(t + c.sep for t, c in zip(translations, seg.chunks, strict=True))

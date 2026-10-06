"""Deterministic checks that keep LLM output faithful to the recording.

``check_refinement`` compares a raw transcript segment with the LLM's refined
version and returns a list of problems. Any problem means the refinement is
rejected and the raw text is used instead.

Both texts are first normalised to the same canonical token form (lowercase,
number words -> digits, "$42,000" -> "42000 dollars", "can't" -> "can not",
fillers kept), so formatting-only edits compare as equal and real content
changes stand out. Then:

1. Word-level diff: every inserted, deleted or replaced run of words must be
   explained as a filler/stutter removal or a sound-alike correction.
2. Invariants: numbers, negations, modal/commitment words, dates, currencies,
   percentages, question marks and technical identifiers must be preserved.
3. Overall similarity: wholesale rewrites are rejected.

Meeting-record grounding checks are added in a later phase.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum character similarity for a replaced run of words (sound-alike check)
REPLACE_SIMILARITY = 0.5
# Stricter threshold when the raw run contains a likely proper noun (a name)
PROPER_NOUN_SIMILARITY = 0.75
# A single replaced run may cover at most this many raw words ...
MAX_REPLACED_WORDS = 6
# ... and add at most this many words
MAX_ADDED_WORDS_PER_REPLACE = 2
# Whole-segment token similarity below this (for segments with enough words)
MIN_SEGMENT_SIMILARITY = 0.6
MIN_WORDS_FOR_SIMILARITY = 4

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

FILLERS = frozenset({"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "hm", "mm", "mhm"})

NEGATIONS = frozenset({
    "not", "no", "never", "nor", "neither", "none", "nothing", "nobody", "nowhere", "without",
})

# Words that carry certainty, possibility or commitment. Changing any of these
# can turn a proposal into a decision or a possibility into a fact.
MODALS = frozenset({
    "might", "may", "could", "would", "should", "will", "shall", "must", "can",
    "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "definitely", "certainly", "surely",
    "agree", "agreed", "agrees", "decide", "decided", "decides",
    "approve", "approved", "confirm", "confirmed",
    "propose", "proposed", "suggest", "suggested", "consider", "considering",
    "plan", "planned", "planning", "going", "want", "need", "needs",
})

MONTHS = {
    "january": "january", "jan": "january", "february": "february", "feb": "february",
    "march": "march", "april": "april", "apr": "april", "june": "june", "jun": "june",
    "july": "july", "jul": "july", "august": "august", "aug": "august",
    "september": "september", "sep": "september", "sept": "september",
    "october": "october", "oct": "october", "november": "november", "nov": "november",
    "december": "december", "dec": "december",
    # "may" is deliberately absent: it is far more often the modal verb
}
DATE_WORDS = frozenset(set(MONTHS.values()) | {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "tonight", "weekend", "eod", "eow",
})

CURRENCY_WORDS = {
    "dollar": "dollars", "dollars": "dollars", "usd": "dollars", "bucks": "dollars",
    "euro": "euros", "euros": "euros", "eur": "euros",
    "pound": "pounds", "pounds": "pounds", "gbp": "pounds",
    "rupee": "rupees", "rupees": "rupees", "inr": "rupees", "rs": "rupees",
    "yen": "yen", "cent": "cents", "cents": "cents",
}
_CURRENCY_SYMBOLS = {"$": "dollars", "€": "euros", "£": "pounds", "₹": "rupees", "¥": "yen"}

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine".split())}
_TEENS = {w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_ORD_UNITS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth".split())}
_ORD_TEENS = {w: i + 10 for i, w in enumerate(
    "tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
    "eighteenth nineteenth".split())}
_ORD_TENS = {w: (i + 2) * 10 for i, w in enumerate(
    "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth".split())}
_SCALES = {
    "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
    "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
    "k": 1_000,  # "forty k"; only counts when it follows a number
}
_SUFFIX_SCALES = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000}

_CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"),
    (r"\bshan't\b", "shall not"), (r"n't\b", " not"), (r"'ll\b", " will"),
    (r"'re\b", " are"), (r"'ve\b", " have"), (r"'m\b", " am"), (r"'d\b", " would"),
    (r"\bgonna\b", "going to"), (r"\bwanna\b", "want to"), (r"\bgotta\b", "got to"),
]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_ORDINAL_DIGITS_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
_SUFFIXED_DIGITS_RE = re.compile(r"^(\d+(?:\.\d+)?)(k|m|bn)$")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_tokens(text: str) -> list[str]:
    """Canonical comparison tokens for ``text``."""
    s = (text or "").translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                              "–": "-", "—": "-"}))
    s = s.lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)  # 42,000 / 2,00,000 -> plain digits
    for pattern, repl in _CONTRACTIONS:
        s = re.sub(pattern, repl, s)
    s = re.sub(
        r"([$€£₹¥])\s?(\d+(?:\.\d+)?)\s?(k|m|bn|thousand|million|billion|lakhs?|crores?)?\b",
        lambda m: f"{m.group(2)} {m.group(3) or ''} {_CURRENCY_SYMBOLS[m.group(1)]}",
        s,
    )
    s = s.replace("%", " percent ").replace("per cent", "percent")
    s = s.replace("-", " ")

    tokens = []
    for tok in _TOKEN_RE.findall(s):
        tok = tok.strip("'")
        tok = CURRENCY_WORDS.get(tok, tok)
        tok = MONTHS.get(tok, tok)
        tokens.append(tok)
    return _numbers_to_digits(tokens)


def compact(tokens: list[str]) -> str:
    return "".join(tokens)


def _canon_number(value: Decimal) -> str:
    value = value.normalize()
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _digit_token_value(tok: str) -> Decimal | None:
    m = _ORDINAL_DIGITS_RE.match(tok)
    if m:
        return Decimal(m.group(1))
    m = _SUFFIXED_DIGITS_RE.match(tok)
    if m:
        return Decimal(m.group(1)) * _SUFFIX_SCALES[m.group(2)]
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        try:
            return Decimal(tok)
        except InvalidOperation:
            return None
    return None


def _is_number_word(tok: str) -> bool:
    return (tok in _UNITS or tok in _TEENS or tok in _TENS or tok in _ORD_UNITS
            or tok in _ORD_TEENS or tok in _ORD_TENS or tok in ("hundred", "hundredth")
            or tok in _SCALES)


def _numbers_to_digits(tokens: list[str]) -> list[str]:
    """Replace spelled-out numbers (and digit+scale) with canonical digit tokens."""
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        value, used = _parse_number(tokens, i)
        if used:
            out.append(_canon_number(value))
            i += used
        else:
            out.append(tokens[i])
            i += 1
    return out


def _parse_number(toks: list[str], start: int) -> tuple[Decimal, int]:
    """Parse a number phrase at ``start``. Returns (value, tokens consumed)."""
    n = len(toks)
    total = Decimal(0)
    current = Decimal(0)
    last: str | None = None
    j = start

    first = toks[start]
    digit_value = _digit_token_value(first)
    if digit_value is not None:
        current, last, j = digit_value, "digit", start + 1
    elif first == "a" and start + 1 < n and (toks[start + 1] == "hundred" or toks[start + 1] in _SCALES):
        current, last, j = Decimal(1), "a", start + 1
    elif not _is_number_word(first) or first in _SCALES or first == "hundred":
        return Decimal(0), 0

    while j < n:
        w = toks[j]
        if w in _UNITS and last in (None, "tens", "hundred", "scale"):
            current += _UNITS[w]; last = "unit"
        elif w in _TEENS and last in (None, "hundred", "scale"):
            current += _TEENS[w]; last = "teen"
        elif w in _TENS and last in (None, "hundred", "scale"):
            current += _TENS[w]; last = "tens"
        elif w in _ORD_UNITS and last in (None, "tens", "hundred", "scale"):
            current += _ORD_UNITS[w]; j += 1; break
        elif (w in _ORD_TEENS or w in _ORD_TENS) and last in (None, "hundred", "scale"):
            current += _ORD_TEENS.get(w) or _ORD_TENS[w]; j += 1; break
        elif w in ("hundred", "hundredth") and last in ("unit", "teen", "tens", "a", "digit"):
            current = (current or 1) * 100; last = "hundred"
            if w == "hundredth":
                j += 1; break
        elif w in _SCALES and last in ("unit", "teen", "tens", "hundred", "a", "digit"):
            total += (current or 1) * _SCALES[w]; current = Decimal(0); last = "scale"
        elif (w == "and" and last in ("hundred", "scale") and j + 1 < n
              and (toks[j + 1] in _UNITS or toks[j + 1] in _TEENS or toks[j + 1] in _TENS
                   or toks[j + 1] in _ORD_UNITS or toks[j + 1] in _ORD_TEENS)):
            pass  # "one hundred and five"
        elif (w == "point" and last in ("unit", "teen", "tens", "digit") and j + 1 < n
              and toks[j + 1] in _UNITS):
            digits = ""
            k = j + 1
            while k < n and toks[k] in _UNITS:
                digits += str(_UNITS[toks[k]]); k += 1
            current = Decimal(f"{int(total + current)}.{digits}")
            total = Decimal(0)
            j, last = k, "digit"  # a scale may follow: "one point five million"
            continue
        else:
            break
        j += 1

    if last == "a" and j == start + 1:
        return Decimal(0), 0
    return total + current, j - start


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def numbers_in(tokens: list[str]) -> Counter[str]:
    found: Counter[str] = Counter()
    for tok in tokens:
        for num in _NUMBER_RE.findall(tok):
            found[_canon_number(Decimal(num))] += 1
    return found


def _count(tokens: list[str], vocab: frozenset[str] | set[str]) -> Counter[str]:
    return Counter(t for t in tokens if t in vocab)


def _identifiers(tokens: list[str]) -> set[str]:
    """Tokens mixing letters and digits, e.g. k8s, s3, v1.29, q3."""
    return {t for t in tokens if re.search(r"[a-z]", t) and re.search(r"\d", t)}


def proper_nouns(text: str) -> set[str]:
    """Lowercased capitalised words that do not start a sentence (likely names)."""
    names = set()
    for m in re.finditer(r"[A-Z][a-zA-Z]+", text or ""):
        before = text[: m.start()].rstrip()
        if not before or before[-1] in ".!?:\"'":
            continue
        word = m.group(0).lower()
        if word not in DATE_WORDS and word not in MONTHS:
            names.add(word)
    return names


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def check_refinement(
    raw_text: str,
    refined_text: str,
    changes: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Return problems with ``refined_text`` as a cleanup of ``raw_text``.

    ``changes`` are the (original_span, refined_span) pairs the model reported.
    An empty list means the refinement is safe to use.
    """
    issues: list[str] = []
    raw_stripped, refined_stripped = (raw_text or "").strip(), (refined_text or "").strip()

    if raw_stripped and not refined_stripped:
        return ["refined text is empty"]

    for original_span, refined_span in changes or []:
        if not _contains(raw_stripped, original_span):
            issues.append(f"reported change {original_span!r} not found in the raw text")
        if refined_span and not _contains(refined_stripped, refined_span):
            issues.append(f"reported replacement {refined_span!r} not found in the refined text")

    raw_toks, ref_toks = normalize_tokens(raw_stripped), normalize_tokens(refined_stripped)
    if raw_toks == ref_toks:
        issues += _check_question_marks(raw_stripped, refined_stripped)
        return issues

    # 1. Overall size and similarity
    if len(raw_toks) >= MIN_WORDS_FOR_SIMILARITY:
        # Word merges ("post gres" -> "postgres") lower word similarity but not
        # character similarity, so a rewrite must look large on both measures.
        word_ratio = SequenceMatcher(None, raw_toks, ref_toks, autojunk=False).ratio()
        char_ratio = _similarity(" ".join(raw_toks), " ".join(ref_toks))
        if max(word_ratio, char_ratio) < MIN_SEGMENT_SIMILARITY:
            issues.append(f"rewrite too large (similarity {max(word_ratio, char_ratio):.2f})")
    if len(ref_toks) > len(raw_toks) * 1.5 + 3:
        issues.append("refined text is much longer than the original")

    # 2. Every edited run of words must be explainable
    names = proper_nouns(raw_stripped)
    matcher = SequenceMatcher(None, raw_toks, ref_toks, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        removed, added = raw_toks[a0:a1], ref_toks[b0:b1]
        if op == "delete":
            if not _removable(raw_toks, a0, a1):
                issues.append(f"removed words: {' '.join(removed)!r}")
        elif op == "insert":
            issues.append(f"added words not in the original: {' '.join(added)!r}")
        else:
            issues += _check_replacement(removed, added, names)

    # 3. Invariants that must survive any cleanup
    issues += _check_invariants(raw_toks, ref_toks)
    issues += _check_question_marks(raw_stripped, refined_stripped)
    return issues


def _contains(text: str, span: str) -> bool:
    squash = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return squash(span) in squash(text)


def _removable(raw_toks: list[str], a0: int, a1: int) -> bool:
    for k in range(a0, a1):
        tok = raw_toks[k]
        stutter = (k > 0 and raw_toks[k - 1] == tok) or (k + 1 < len(raw_toks) and raw_toks[k + 1] == tok)
        if tok not in FILLERS and not stutter:
            return False
    return True


def _check_replacement(removed: list[str], added: list[str], names: set[str]) -> list[str]:
    core = [t for t in removed if t not in FILLERS] or removed
    label = f"{' '.join(removed)!r} -> {' '.join(added)!r}"
    if len(core) > MAX_REPLACED_WORDS:
        return [f"replaced too many words at once: {label}"]
    if len(added) > len(core) + MAX_ADDED_WORDS_PER_REPLACE:
        return [f"replacement adds words: {label}"]
    threshold = PROPER_NOUN_SIMILARITY if names & set(core) else REPLACE_SIMILARITY
    score = _similarity(compact(core), compact(added))
    if score < threshold:
        kind = "name changed" if threshold == PROPER_NOUN_SIMILARITY else "unsupported substitution"
        return [f"{kind} (similarity {score:.2f}): {label}"]
    return []


def _check_invariants(raw_toks: list[str], ref_toks: list[str]) -> list[str]:
    issues = []

    def compare(label: str, raw: Counter, ref: Counter) -> None:
        if raw != ref:
            lost = sorted((raw - ref).elements())
            gained = sorted((ref - raw).elements())
            parts = []
            if lost:
                parts.append(f"removed {lost}")
            if gained:
                parts.append(f"added {gained}")
            issues.append(f"{label} changed: {', '.join(parts)}")

    compare("numbers", numbers_in(raw_toks), numbers_in(ref_toks))
    compare("negation", _count(raw_toks, NEGATIONS), _count(ref_toks, NEGATIONS))
    compare("certainty/commitment words", _count(raw_toks, MODALS), _count(ref_toks, MODALS))
    compare("dates", _count(raw_toks, DATE_WORDS), _count(ref_toks, DATE_WORDS))
    currencies = set(CURRENCY_WORDS.values())
    compare("monetary units", _count(raw_toks, currencies), _count(ref_toks, currencies))
    compare("percentages", _count(raw_toks, {"percent"}), _count(ref_toks, {"percent"}))

    raw_compact = compact(raw_toks)
    for ident in sorted(_identifiers(ref_toks) - _identifiers(raw_toks)):
        if ident.replace(".", "") not in raw_compact.replace(".", ""):
            issues.append(f"new technical identifier not supported by the original: {ident!r}")
    return issues


def _check_question_marks(raw: str, refined: str) -> list[str]:
    if refined.count("?") < raw.count("?"):
        return ["a question was turned into a statement"]
    return []

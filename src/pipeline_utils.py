"""Pure-Python utility functions for the FlowSure pipeline.

Isolated from Spark so they can be unit-tested in CI without a Spark session.
Used by the Databricks notebooks via Pandas UDFs or direct imports.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

# ---------- Text cleaning & PII masking ----------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\s-]?){7,15}\b")
_ORDER_RE = re.compile(r"\b(?:order|invoice|ticket)[#\s:]*[\w-]+\b", flags=re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\{\{[^}]+\}\}")  # bitext placeholders like {{Order Number}}
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip URLs & mentions, collapse whitespace.

    Bitext placeholders like ``{{Order Number}}`` are kept as ``<order>`` so the
    classifier still sees a signal instead of raw braces.
    """
    if text is None:
        return ""
    t = str(text).lower()
    t = _URL_RE.sub(" <url> ", t)
    t = _MENTION_RE.sub(" <user> ", t)
    t = _TEMPLATE_RE.sub(lambda m: f" <{m.group(0)[2:-2].strip().lower().replace(' ', '_')}> ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def mask_pii(text: str) -> str:
    """GDPR-friendly masking of emails, phone numbers and order refs."""
    if text is None:
        return ""
    t = str(text)
    t = _EMAIL_RE.sub("<email>", t)
    t = _PHONE_RE.sub("<phone>", t)
    t = _ORDER_RE.sub("<order_ref>", t)
    return t


def detect_language(text: str) -> str:
    """Very small language heuristic — good enough as a routing hint.

    Real deployments should swap this for ``langdetect`` or fastText. Kept
    stdlib-only so the unit tests don't need extra deps.
    """
    if not text:
        return "unknown"
    t = text.lower()

    nl_words = [" de ", " het ", " een ", " ik ", " je ", " niet ", " met ", " voor "]
    en_words = [" the ", " a ", " an ", " i ", " you ", " not ", " with ", " for "]
    de_words = [" der ", " die ", " das ", " ich ", " nicht ", " und ", " mit ", " für "]
    fr_words = [" le ", " la ", " les ", " je ", " ne ", " pas ", " avec ", " pour "]

    nl = sum(w in t for w in nl_words)
    en = sum(w in t for w in en_words)
    de = sum(w in t for w in de_words)
    fr = sum(w in t for w in fr_words)
    scores = {"nl": nl, "en": en, "de": de, "fr": fr}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"  # default English


# ---------- Data drift metrics ----------


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two probability vectors.

    Both inputs must sum to 1 and have the same length (aligned bins).
    Values commonly interpreted as: <0.1 stable, 0.1-0.25 moderate, >0.25 drift.
    """
    e = list(expected)
    a = list(actual)
    if len(e) != len(a):
        raise ValueError("expected and actual must have same length")
    psi = 0.0
    for ei, ai in zip(e, a, strict=False):
        ei_s = ei if ei > 0 else eps
        ai_s = ai if ai > 0 else eps
        psi += (ai_s - ei_s) * math.log(ai_s / ei_s)
    return psi


def to_probabilities(counts: dict) -> dict:
    """Normalize a count dict to a probability dict."""
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def align_distributions(expected: dict, actual: dict) -> tuple[list[float], list[float]]:
    """Align two count/prob dicts on the same key set (missing keys -> 0)."""
    keys = sorted(set(expected) | set(actual))
    e = [expected.get(k, 0.0) for k in keys]
    a = [actual.get(k, 0.0) for k in keys]
    return e, a


# ---------- Simple retrieval helpers used by the responder model ----------


def top_k_cosine(query_vec, matrix, k: int = 3):
    """Return indices of top-k cosine-similar rows in ``matrix`` for ``query_vec``.

    ``matrix`` is expected to be a 2-D iterable of already L2-normalized vectors
    and ``query_vec`` a 1-D iterable of the same dim, also L2-normalized. Kept
    dependency-free for unit tests.
    """
    scores = []
    for i, row in enumerate(matrix):
        scores.append((i, sum(q * r for q, r in zip(query_vec, row, strict=False))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:k]]

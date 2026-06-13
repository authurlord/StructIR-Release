"""Self-contained QA scoring (SQuAD-style normalization + EM + token F1).

Used by the fixed readers to score answers against gold strings. No external
dependency; mirrors the standard open-domain QA metric.
"""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import List, Tuple


def normalize_answer(s) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD norm)."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_precision_recall_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    p_tok = normalize_answer(pred).split()
    g_tok = normalize_answer(gold).split()
    if not p_tok or not g_tok:
        v = float(p_tok == g_tok)
        return v, v, v
    common = Counter(p_tok) & Counter(g_tok)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(p_tok)
    recall = overlap / len(g_tok)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def parse_answer(text: str) -> str:
    """Best-effort answer extraction from a CoT/free-form generation: prefer an
    explicit 'Answer:' line, else the last non-empty line."""
    s = (text or "").strip()
    m = re.search(r"(?:final\s+)?answer\s*[:\-]\s*(.+)", s, flags=re.I)
    if m:
        return m.group(1).splitlines()[0].strip().strip(". ")
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return lines[-1].strip(". ") if lines else s

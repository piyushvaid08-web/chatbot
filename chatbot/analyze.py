"""Deterministic micro-analysis for spreadsheet questions.

LLMs are unreliable at arithmetic, so when the user asks a typical table
question (totals, averages, counts, min/max), we compute the answer in Python
over the retrieved chunks and hand the result to the model as ground truth.

``try_compute`` returns a short factual sentence when it can answer
confidently, otherwise ``None`` (and the LLM answers as usual).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+")
_INTENTS = [
    (re.compile(r"\b(total|sum)\b", re.I), "sum"),
    (re.compile(r"\b(average|mean)\b", re.I), "mean"),
    (re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b", re.I), "count"),
    (re.compile(r"\b(highest|most|maximum|max|top)\b", re.I), "max"),
    (re.compile(r"\b(lowest|least|minimum|min)\b", re.I), "min"),
]

# Words that carry no column meaning in aggregation questions.
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "and", "or", "is",
    "was", "are", "were", "what", "which", "who", "how", "many", "much", "do",
    "does", "did", "across", "all", "total", "sum", "average", "mean", "count",
    "number", "highest", "most", "maximum", "max", "top", "lowest", "least",
    "minimum", "min", "overall", "please", "answer", "from", "data", "based",
    "according", "with", "have", "has", "their", "our", "your", "can", "you",
    "tell", "me", "show", "us", "about", "employee", "employees", "person",
    "people", "row", "rows", "value", "values", "figure", "figures", "info",
    "sheet", "sales", "sold", "revenue", "salary", "units", "region", "product",
    "department", "name", "amount", "total", "per", "each", "any", "every",
}


def try_compute(question: str, chunks: List[Tuple[str, str, float]]) -> Optional[str]:
    """Answer a spreadsheet aggregation question deterministically, or return None."""
    tables = []
    for source, text, _score in chunks:
        if not source.lower().endswith((".xls", ".xlsx", ".csv")):
            continue
        parsed = _parse_table(text)
        if parsed:
            tables.append((source, parsed[0], parsed[1]))
    if not tables:
        return None

    intent = _detect_intent(question)
    if not intent:
        return None

    q_lower = question.lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))

    # Scope by sheet: keep only sources whose label mentions a query token
    # (e.g. "q1" -> "sample.xlsx [Sheet: Q1 Sales]").
    sheet_scoped = None
    for token in q_tokens:
        if token in _STOP or token.isdigit():
            continue
        hits = [t for t in tables if token in t[0].lower()]
        if len(hits) == 1:
            sheet_scoped = [hits[0]]
            break

    active = sheet_scoped or tables

    # Find the target numeric column across the active tables.
    target = _match_column(active, q_tokens)
    if target is None:
        return None

    header, col_idx = target

    # Filter rows: a query token matching a cell value (e.g. region "west").
    filtered_rows: List[List[str]] = []
    for _src, hdr, rows in active:
        for row in rows:
            keep = True
            for token in q_tokens:
                if token in _STOP or token.isdigit():
                    continue
                if any(token == cell.lower() for cell in row):
                    keep = True
                    break
                if token == hdr[col_idx].lower():
                    continue
                # A non-numeric filter value mentioned -> must match somewhere
                # in this row to keep it (e.g. "widget").
                if _looks_like_value(token) and not any(
                    token in cell.lower() for cell in row
                ):
                    keep = False
                    break
            if keep:
                filtered_rows.append(row)

    if not filtered_rows:
        # Fall back to unfiltered rows rather than refusing.
        filtered_rows = [row for _src, _hdr, rows in active for row in rows]

    values = [_to_number(row[col_idx]) for row in filtered_rows]
    values = [v for v in values if v is not None]
    if not values:
        return None

    sources = sorted({src for src, _h, _r in active})
    label = sources[0] if len(sources) == 1 else f"{len(sources)} sheets"

    if intent == "count":
        return f"[Computed from {label}] {len(filtered_rows)} matching row(s)."

    if intent == "sum":
        total = _fmt(sum(values))
        detail = ", ".join(_fmt(v) for v in values[:12])
        more = f" ({len(values)} values: {detail})" if len(values) <= 12 else ""
        return f"[Computed from {label}] sum of the {len(values)} matching {header[col_idx]} value(s){more} = {total}."

    if intent == "mean":
        avg = _fmt(sum(values) / len(values))
        return f"[Computed from {label}] average {header[col_idx]} over {len(values)} matching row(s) = {avg}."

    if intent == "max":
        best = max(values)
        row = next(r for r in filtered_rows if _to_number(r[col_idx]) == best)
        detail = " | ".join(f"{h}: {c}" for h, c in zip(header, row))
        return f"[Computed from {label}] max {header[col_idx]} = {_fmt(best)} in row [{detail}]."

    if intent == "min":
        best = min(values)
        row = next(r for r in filtered_rows if _to_number(r[col_idx]) == best)
        detail = " | ".join(f"{h}: {c}" for h, c in zip(header, row))
        return f"[Computed from {label}] min {header[col_idx]} = {_fmt(best)} in row [{detail}]."

    return None


def _parse_table(text: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    lines = [ln.strip() for ln in text.splitlines() if "|" in ln]
    if not lines:
        return None
    header = [c.strip() for c in lines[0].split("|")]
    if len(header) < 2:
        return None
    rows = [[c.strip() for c in ln.split("|")] for ln in lines[1:]]
    rows = [r for r in rows if len(r) >= 2 and any(r)]
    return (header, rows) if rows else None


def _detect_intent(question: str) -> Optional[str]:
    for pattern, name in _INTENTS:
        if pattern.search(question):
            return name
    return None


def _match_column(tables, q_tokens: set) -> Optional[Tuple[List[str], int]]:
    """Pick the numeric column the question asks about."""
    candidates = []
    for _src, header, rows in tables:
        for idx, cell in enumerate(header):
            cell_tokens = set(re.findall(r"[a-z0-9]+", cell.lower()))
            overlap = len(cell_tokens & q_tokens)
            numeric = any(_to_number(r[idx]) is not None for r in rows)
            candidates.append((overlap, numeric, cell, idx, header))
    # Prefer a column with a header match and numeric values.
    scored = [c for c in candidates if c[0] > 0 and c[1]]
    if scored:
        scored.sort(key=lambda c: c[0], reverse=True)
        return scored[0][4], scored[0][3]
    # No header match: use the single numeric column if unambiguous.
    numeric_cols = {(c[3], c[4]) for c in candidates if c[1]}
    if len(numeric_cols) == 1:
        idx, header = next(iter(numeric_cols))
        return header, idx
    return None


def _looks_like_value(token: str) -> bool:
    return len(token) >= 3 and not token.isdigit()


def _to_number(value: str) -> Optional[float]:
    cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()
    m = _NUM_RE.search(cleaned)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _fmt(value: float) -> str:
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"
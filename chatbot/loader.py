"""Load text out of documents: Excel (.xls/.xlsx), CSV, plain text, PDF, Word.

Every loader returns a list of ``(source, text)`` pairs, where ``source`` is a
short human-readable label like ``sales.xlsx [Sheet: Q1]`` or ``guide.pdf [page 3]``.
The text is later split into overlapping chunks by the knowledge base.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Iterable, List, Tuple

from .config import SUPPORTED_EXTENSIONS

RawDocument = Tuple[str, str]  # (source label, extracted text)


def _read_plain_text(path: str, source: str) -> List[RawDocument]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as fh:
                return [(source, fh.read())]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with a supported encoding.")


def _read_csv(path: str, source: str) -> List[RawDocument]:
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(fh, dialect))
    if not rows:
        return [(source, "(empty file)")]
    lines = [" | ".join(cell.strip() for cell in row if cell is not None and cell.strip()) for row in rows]
    lines = [line for line in lines if line]
    return [(source, "\n".join(lines))]


def _read_xlsx(path: str, source: str) -> List[RawDocument]:
    from openpyxl import load_workbook

    docs: List[RawDocument] = []
    wb = load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        lines = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append(" | ".join(cells))
        if lines:
            docs.append((f"{source} [Sheet: {ws.title}]", "\n".join(lines)))
    wb.close()
    if not docs:
        docs.append((source, "(spreadsheet with no data)"))
    return docs


def _read_xls(path: str, source: str) -> List[RawDocument]:
    import xlrd

    docs: List[RawDocument] = []
    book = xlrd.open_workbook(path, on_demand=True)
    for sheet in book.sheets():
        lines = []
        for row_idx in range(sheet.nrows):
            cells = [
                str(sheet.cell_value(row_idx, col_idx)).strip()
                for col_idx in range(sheet.ncols)
                if str(sheet.cell_value(row_idx, col_idx)).strip()
            ]
            if cells:
                lines.append(" | ".join(cells))
        if lines:
            docs.append((f"{source} [Sheet: {sheet.name}]", "\n".join(lines)))
    if not docs:
        docs.append((source, "(spreadsheet with no data)"))
    return docs


def _read_pdf(path: str, source: str) -> List[RawDocument]:
    from PyPDF2 import PdfReader

    reader = PdfReader(path)
    docs: List[RawDocument] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append((f"{source} [page {idx}]", text))
    if not docs:
        docs.append((source, "(PDF with no extractable text)"))
    return docs


def _read_docx(path: str, source: str) -> List[RawDocument]:
    from docx import Document

    doc = Document(path)
    parts: List[str] = []

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if paragraphs:
        parts.append("\n".join(paragraphs))

    for t_idx, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"[Table {t_idx}]\n" + "\n".join(rows))

    if not parts:
        parts.append("(Word document with no extractable text)")
    return [(source, "\n\n".join(parts))]


def _read_image(path: str, source: str) -> List[RawDocument]:
    import base64
    from openai import OpenAI
    from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

    if not OPENAI_API_KEY:
        return [(source, "(Image cannot be read because OPENAI_API_KEY is not set)")]

    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    ext = os.path.splitext(path)[1].lower().replace('.', '')
    if ext == 'jpg': ext = 'jpeg'

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail so it can be searched and queried later. Extract all visible text, numbers, data from charts, and describe the visual elements thoroughly."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        description = response.choices[0].message.content or ""
        return [(source, f"[Image Description]\n{description.strip()}")]
    except Exception as e:
        return [(source, f"(Failed to read image: {e})")]


_LOADERS = {
    ".txt": _read_plain_text,
    ".md": _read_plain_text,
    ".log": _read_plain_text,
    ".csv": _read_csv,
    ".xlsx": _read_xlsx,
    ".xls": _read_xls,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".png": _read_image,
    ".jpg": _read_image,
    ".jpeg": _read_image,
    ".gif": _read_image,
    ".webp": _read_image,
}


def load_file(path: str) -> List[RawDocument]:
    """Extract ``(source, text)`` pairs from a single file."""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    source = os.path.basename(path)
    return _LOADERS[ext](path, source)


def load_paths(paths: Iterable[str]) -> List[RawDocument]:
    """Load many files or directories. Directories are scanned recursively."""
    docs: List[RawDocument] = []
    for raw in paths:
        p = os.path.abspath(raw)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in sorted(files):
                    ext = os.path.splitext(name)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        docs.extend(load_file(os.path.join(root, name)))
        else:
            docs.extend(load_file(p))
    return docs


def extract_markdown_table(text: str, max_chars: int = 2500) -> str:
    """Helper for the chat prompt: collapse a raw sheet dump into a small table."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:40]
    if not lines:
        return text[:max_chars]
    return "\n".join(lines)[:max_chars]
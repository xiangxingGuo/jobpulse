from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_text_from_txt_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore").strip()


def extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages).strip()


def extract_text_from_docx_bytes(data: bytes) -> str:
    doc = Document(BytesIO(data))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras).strip()


def extract_resume_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return extract_text_from_txt_bytes(data)
    if suffix == ".pdf":
        return extract_text_from_pdf_bytes(data)
    if suffix == ".docx":
        return extract_text_from_docx_bytes(data)

    raise ValueError(f"unsupported resume file type: {suffix}")

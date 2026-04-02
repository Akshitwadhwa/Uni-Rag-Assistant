from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    source: str
    metadata: dict


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to ingest PDF documents.") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return path.read_text(encoding="utf-8")


def extract_field(text: str, labels: list[str]) -> str | None:
    for label in labels:
        pattern = rf"(?im)^{re.escape(label)}\s*:\s*(.+)$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def infer_course_metadata(text: str, source_name: str) -> dict:
    course_code = extract_field(text, ["Course Code", "Subject Code", "Paper Code"])
    subject_name = extract_field(text, ["Course Title", "Subject Name", "Course Name", "Title"])
    credits = extract_field(text, ["Credits", "Credit"])
    ltp = extract_field(text, ["L-T-P", "L-T-P-C", "Contact Hours", "Lecture-Tutorial-Practical"])
    faculty = extract_field(text, ["Instructor", "Faculty", "Course Coordinator"])
    semester = extract_field(text, ["Semester", "Term"])

    metadata = {
        "filename": source_name,
        "suffix": Path(source_name).suffix.lower(),
        "doc_type": "course_handout",
        "course_code": course_code or Path(source_name).stem.upper(),
        "subject_name": subject_name or Path(source_name).stem.replace("_", " ").title(),
        "credits": credits or "unknown",
        "ltp": ltp or "unknown",
        "faculty": faculty or "unknown",
        "semester": semester or "unknown",
    }
    return metadata


def load_documents(data_dir: str | Path) -> list[Document]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = normalize_whitespace(read_text_file(path))
        metadata = infer_course_metadata(text, path.name)
        documents.append(
            Document(
                doc_id=path.stem,
                text=text,
                source=str(path),
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(f"No supported documents found in {root}")
    return documents


def preview_documents(documents: Iterable[Document], limit: int = 2) -> str:
    lines: list[str] = []
    for document in list(documents)[:limit]:
        lines.append(f"- {document.doc_id}: {document.text[:120]}...")
    return "\n".join(lines)

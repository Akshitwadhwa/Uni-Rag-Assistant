from __future__ import annotations

from dataclasses import dataclass
import re

from rag_assignment.ingestion import Document


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    metadata: dict


def split_sentences(text: str) -> list[str]:
    parts = text.replace("?", ".").replace("!", ".").split(".")
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        start += step
    return chunks


def sentence_window_chunk(text: str, window_size: int, stride: int) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    for start in range(0, len(sentences), stride):
        window = sentences[start : start + window_size]
        if not window:
            continue
        chunks.append(". ".join(window).strip() + ".")
    return chunks


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.endswith(":") and len(stripped) <= 80:
        return True
    if re.fullmatch(r"[A-Z0-9 /&()-]{4,}", stripped):
        return True
    return False


def normalize_heading(line: str) -> str:
    return line.strip().lstrip("#").strip(" :")


def infer_section_type(title: str) -> str:
    normalized = title.lower()
    if any(keyword in normalized for keyword in ["credit", "ltp", "contact hour"]):
        return "credits"
    if any(keyword in normalized for keyword in ["evaluation", "assessment", "grading", "marks"]):
        return "evaluation"
    if any(keyword in normalized for keyword in ["unit", "syllabus", "module", "course content", "topics"]):
        return "syllabus"
    if any(keyword in normalized for keyword in ["objective", "outcome"]):
        return "outcomes"
    if any(keyword in normalized for keyword in ["reference", "textbook"]):
        return "references"
    if any(keyword in normalized for keyword in ["faculty", "instructor", "coordinator"]):
        return "faculty"
    return "general"


def extract_sections(text: str) -> list[tuple[str, str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    sections: list[tuple[str, str]] = []
    current_title = "Document Overview"
    buffer: list[str] = []

    for line in lines:
        if is_heading(line):
            if buffer:
                sections.append((current_title, "\n".join(buffer).strip()))
                buffer = []
            current_title = normalize_heading(line)
            continue
        buffer.append(line)

    if buffer:
        sections.append((current_title, "\n".join(buffer).strip()))

    return [(title, body) for title, body in sections if body]


def build_metadata_prefix(document: Document, section_title: str, section_type: str) -> str:
    return (
        f"Course: {document.metadata.get('subject_name', document.doc_id)}\n"
        f"Course Code: {document.metadata.get('course_code', document.doc_id)}\n"
        f"Credits: {document.metadata.get('credits', 'unknown')}\n"
        f"Section: {section_title}\n"
        f"Section Type: {section_type}\n"
    )


def section_aware_chunks(document: Document, chunk_size: int, chunk_overlap: int) -> list[tuple[str, dict]]:
    chunks: list[tuple[str, dict]] = []
    sections = extract_sections(document.text)
    if not sections:
        fallback_text = build_metadata_prefix(document, "Document Overview", "general") + "\n" + document.text
        return [(fallback_text, {"section_title": "Document Overview", "section_type": "general"})]

    for section_title, section_body in sections:
        section_type = infer_section_type(section_title)
        prefix = build_metadata_prefix(document, section_title, section_type)
        if len(section_body.split()) <= chunk_size:
            chunks.append(
                (
                    f"{prefix}\n{section_body}".strip(),
                    {"section_title": section_title, "section_type": section_type},
                )
            )
            continue

        for part in chunk_text(section_body, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append(
                (
                    f"{prefix}\n{part}".strip(),
                    {"section_title": section_title, "section_type": section_type},
                )
            )
    return chunks


def build_chunks(
    documents: list[Document],
    strategy: str = "section_fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    sentence_window_size: int = 5,
    sentence_stride: int = 3,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        if strategy == "fixed":
            parts = [
                (
                    build_metadata_prefix(document, "Document Overview", "general") + "\n" + part,
                    {"section_title": "Document Overview", "section_type": "general"},
                )
                for part in chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            ]
        elif strategy == "sentence_window":
            parts = [
                (
                    build_metadata_prefix(document, "Sentence Window", "general") + "\n" + part,
                    {"section_title": "Sentence Window", "section_type": "general"},
                )
                for part in sentence_window_chunk(
                    document.text,
                    window_size=sentence_window_size,
                    stride=sentence_stride,
                )
            ]
        elif strategy == "section_fixed":
            parts = section_aware_chunks(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        for index, (part, part_metadata) in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{index}",
                    doc_id=document.doc_id,
                    text=part,
                    source=document.source,
                    metadata={
                        **document.metadata,
                        **part_metadata,
                        "chunk_index": index,
                        "chunk_strategy": strategy,
                    },
                )
            )
    return chunks

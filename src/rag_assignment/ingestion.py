from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
FIELD_LABELS = [
    "Course Code",
    "Subject Code",
    "Paper Code",
    "Course Title",
    "Course Name",
    "Subject Name",
    "Course Name",
    "Title",
    "Credits",
    "Credit",
    "L-D-P",
    "L-T-P",
    "L-T-P-C",
    "Contact Hours",
    "Lecture-Tutorial-Practical",
    "Instructor",
    "Faculty",
    "Course Faculty",
    "Course Coordinator",
    "Semester",
    "Term",
    "Assessment Pattern",
    "Evaluation Scheme",
    "Evaluation Component",
]


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


def label_pattern(label: str) -> str:
    words = []
    for word in label.split():
        letters = [re.escape(char) for char in word]
        words.append(r"\s*".join(letters))
    return r"\s+".join(words)


def build_search_text(text: str) -> str:
    flattened = re.sub(r"\s+", " ", text)
    # Repair common PDF extraction artifacts such as "A ssessment" and "A im".
    flattened = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", flattened)
    return flattened.strip()


def clean_extracted_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" :,-")
    return value


def extract_field(text: str, labels: list[str]) -> str | None:
    inline_value = extract_inline_field(text, labels)
    if inline_value:
        return inline_value
    for label in labels:
        pattern = rf"(?im)^{label_pattern(label)}\s*:\s*(.+)$"
        match = re.search(pattern, text)
        if match:
            return clean_extracted_value(match.group(1))
    return None


def extract_inline_field(text: str, labels: list[str], stop_labels: list[str] | None = None) -> str | None:
    search_text = build_search_text(text)
    stop_labels = stop_labels or FIELD_LABELS
    stop_patterns = [label_pattern(label) for label in stop_labels]
    stop_union = "|".join(stop_patterns)

    for label in labels:
        pattern = rf"(?i){label_pattern(label)}\s*:\s*(.+?)(?=\s+(?:{stop_union})\s*:|$)"
        match = re.search(pattern, search_text)
        if match:
            return clean_extracted_value(match.group(1))
    return None


def extract_evaluation_summary(text: str) -> str | None:
    search_text = build_search_text(text)
    start_labels = [
        "Assessment Pattern",
        "Evaluation Scheme",
        "Evaluation Component",
        "Evaluation Criteria",
        "Assessment Scheme",
    ]
    end_labels = [
        "Student Responsibilities",
        "Attendance Policy",
        "Recourse Examination Policy",
        "Make-up policy",
        "Behavior Expect",
        "Academic Dishonesty",
        "Reference Books",
        "Video lectures",
    ]

    start_match = None
    for label in start_labels:
        pattern = rf"(?i){label_pattern(label)}\s*:\s*"
        start_match = re.search(pattern, search_text)
        if start_match:
            break

    if not start_match:
        quiz_match = re.search(r"(?i)\bquiz\b.{0,120}\b\d{1,3}\s*%", search_text)
        assignment_match = re.search(r"(?i)\bassignment\b.{0,160}\b\d{1,3}\s*%", search_text)
        if quiz_match and assignment_match:
            start_index = min(quiz_match.start(), assignment_match.start())
            snippet = search_text[start_index : start_index + 1200]
            return clean_extracted_value(snippet)
        return None

    end_index = len(search_text)
    tail = search_text[start_match.end() :]
    for label in end_labels:
        match = re.search(rf"(?i)\b{label_pattern(label)}\s*:", tail)
        if match:
            end_index = min(end_index, start_match.end() + match.start())

    snippet = search_text[start_match.end() : end_index]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if not snippet:
        return None
    return snippet[:2500]


def infer_course_metadata(text: str, source_name: str) -> dict:
    course_code = extract_inline_field(
        text,
        ["Course Code", "Subject Code", "Paper Code"],
        stop_labels=[
            "Course Title",
            "Course Name",
            "Subject Name",
            "Credits",
            "Credit",
            "L-D-P",
            "L-T-P",
            "L-T-P-C",
            "Contact Hours",
            "Batch",
            "Semester",
        ],
    ) or extract_field(text, ["Course Code", "Subject Code", "Paper Code"])
    subject_name = extract_inline_field(
        text,
        ["Course Title", "Subject Name", "Course Name", "Title"],
        stop_labels=["Credits", "Credit", "L-D-P", "L-T-P", "L-T-P-C", "Contact Hours", "Batch", "Semester", "Course Faculty", "Course Coordinator"],
    )
    credits = extract_inline_field(
        text,
        ["Credits", "Credit"],
        stop_labels=["Contact Hours", "Batch", "Semester", "Course Faculty", "Course Coordinator"],
    ) or extract_field(text, ["Credits", "Credit"])
    ltp = extract_field(text, ["L-T-P", "L-T-P-C", "Contact Hours", "Lecture-Tutorial-Practical"])
    faculty = extract_field(text, ["Instructor", "Faculty", "Course Faculty", "Course Coordinator"])
    semester = extract_field(text, ["Semester", "Term"])
    evaluation_summary = extract_evaluation_summary(text)

    if credits:
        credit_match = re.search(r"\d+(?:\.\d+)?(?:\s*\([^)]*\))?", credits)
        if credit_match:
            credits = credit_match.group(0).strip()
    if not credits:
        credit_match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*credits?\b", build_search_text(text))
        if credit_match:
            credits = credit_match.group(1)

    if subject_name:
        subject_name = re.sub(r"\bCoursehandout\b", "", subject_name, flags=re.I).strip(" -")
        if " Credits " in subject_name:
            subject_name = subject_name.split(" Credits ", 1)[0].strip()

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
        "evaluation_summary": evaluation_summary or "unknown",
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

from __future__ import annotations

from rag_assignment.vectorstores import SearchResult


SYSTEM_PROMPT = """You are a course handout assistant.
Answer only from the supplied course handout context.
Prefer exact factual extraction for credits, evaluation criteria, course code, course outcomes, syllabus units, and faculty details.
If the answer is not supported by the context, say that the information is not available in the uploaded handouts.
If the question mentions a specific subject, keep the answer limited to that subject."""


def format_context(results: list[SearchResult]) -> str:
    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        sections.append(
            f"[Context {index} | score={result.score:.4f} | subject={result.chunk.metadata.get('subject_name', 'unknown')} | "
            f"course_code={result.chunk.metadata.get('course_code', 'unknown')} | "
            f"section={result.chunk.metadata.get('section_title', 'unknown')} | "
            f"source={result.chunk.metadata.get('filename', result.chunk.source)}]\n"
            f"{result.chunk.text}"
        )
    return "\n\n".join(sections)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = format_context(results)
    return f"""{SYSTEM_PROMPT}

Retrieved context:
{context}

User question:
{question}

Instructions:
1. Write the response in plain English prose, not markdown and not bullet points.
2. Do not use labels such as "Answer", "Justification", or "Sources".
3. Start with the direct answer in one sentence.
4. Then add one or two short sentences explaining which retrieved context supports it.
5. If marks, credits, or percentages are mentioned, preserve the exact values.
6. Mention the relevant source filenames naturally in the final sentence.
"""

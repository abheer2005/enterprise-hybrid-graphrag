import re
from typing import Any


MIN_CHUNK_CHARS = 80


def clean_text(text: str) -> str:
    """
    Clean common extraction artifacts while preserving document structure.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove PDF page markers such as "Page 3 of 10"
    text = re.sub(
        r"(?im)^\s*page\s+\d+\s+of\s+\d+\s*$",
        "",
        text,
    )

    # Normalize spaces without destroying line structure
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    text = "\n".join(lines)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_heading(text: str) -> bool:
    """
    Detect likely document headings.
    """

    text = text.strip()

    if not text:
        return False

    # Standalone numbering such as:
    # 1.
    # 2.
    # 4.1
    if re.fullmatch(r"\d+(?:\.\d+)*\.?", text):
        return True

    # Numbered headings:
    # 2. DEFINITIONS
    # 4. ETHICAL CONDUCT
    if re.fullmatch(
        r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Z0-9 &/(),.'\-]{2,}",
        text,
    ):
        return True

    # Uppercase headings:
    # INTRODUCTION
    # ETHICAL CONDUCT
    if (
        len(text) <= 150
        and re.fullmatch(r"[A-Z][A-Z0-9 &/(),.'\-]{2,}", text)
    ):
        return True

    return False


def split_into_blocks(text: str) -> list[str]:
    """
    Convert extracted text into structural blocks while keeping headings
    attached to the content that follows them.
    """

    text = clean_text(text)

    if not text:
        return []

    lines = text.split("\n")

    blocks = []
    current_heading = []
    current_body = []

    def flush():
        nonlocal current_heading, current_body

        parts = []

        if current_heading:
            parts.append(" ".join(current_heading))

        if current_body:
            parts.append(" ".join(current_body))

        block = "\n\n".join(parts).strip()

        if block:
            blocks.append(block)

        current_heading = []
        current_body = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if is_heading(line):
            # If body already exists, this heading begins a new block.
            if current_body:
                flush()

            current_heading.append(line)

        else:
            current_body.append(line)

    flush()

    return blocks


def split_long_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Split oversized text while trying to break at sentence boundaries.
    """

    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0

    while start < len(text):
        target_end = min(start + max_chars, len(text))

        if target_end < len(text):
            search_start = max(
                start,
                target_end - 400,
            )

            sentence_breaks = [
                text.rfind(". ", search_start, target_end),
                text.rfind("; ", search_start, target_end),
                text.rfind("\n", search_start, target_end),
            ]

            best_break = max(sentence_breaks)

            if best_break > start:
                target_end = best_break + 1

        piece = text[start:target_end].strip()

        if piece:
            pieces.append(piece)

        if target_end >= len(text):
            break

        new_start = max(
            target_end - overlap_chars,
            start + 1,
        )

        start = new_start

    return pieces


def chunk_record(
    record: dict[str, Any],
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[dict[str, Any]]:
    """
    Create retrieval-ready chunks while preserving provenance.
    """

    text = clean_text(record["text"])

    if not text:
        return []

    blocks = split_into_blocks(text)

    raw_chunks = []
    current_chunk = ""

    for block in blocks:

        if len(block) > max_chars:

            if current_chunk:
                raw_chunks.append(current_chunk.strip())
                current_chunk = ""

            raw_chunks.extend(
                split_long_text(
                    block,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )

            continue

        candidate = (
            f"{current_chunk}\n\n{block}".strip()
            if current_chunk
            else block
        )

        if len(candidate) <= max_chars:
            current_chunk = candidate

        else:
            if current_chunk:
                raw_chunks.append(current_chunk.strip())

            current_chunk = block

    if current_chunk:
        raw_chunks.append(current_chunk.strip())

    # Merge tiny chunks into a neighboring chunk where possible.
    merged_chunks = []

    for chunk in raw_chunks:

        if (
            len(chunk) < MIN_CHUNK_CHARS
            and merged_chunks
            and len(merged_chunks[-1]) + len(chunk) + 2 <= max_chars
        ):
            merged_chunks[-1] = (
                merged_chunks[-1] + "\n\n" + chunk
            ).strip()

        else:
            merged_chunks.append(chunk)

    # If the first chunk is tiny, merge it forward.
    if (
        len(merged_chunks) >= 2
        and len(merged_chunks[0]) < MIN_CHUNK_CHARS
        and len(merged_chunks[0]) + len(merged_chunks[1]) + 2
        <= max_chars
    ):
        merged_chunks[1] = (
            merged_chunks[0]
            + "\n\n"
            + merged_chunks[1]
        )

        merged_chunks.pop(0)

    output = []

    total_chunks = len(merged_chunks)

    for chunk_index, chunk_text in enumerate(
        merged_chunks,
        start=1,
    ):
        metadata = dict(record["metadata"])

        metadata.update(
            {
                "chunk_index": chunk_index,
                "chunks_in_record": total_chunks,
                "character_count": len(chunk_text),
            }
        )

        output.append(
            {
                "text": chunk_text,
                "metadata": metadata,
            }
        )

    return output


def chunk_documents(
    records: list[dict[str, Any]],
    max_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[dict[str, Any]]:

    all_chunks = []

    for record in records:

        chunks = chunk_record(
            record,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        all_chunks.extend(chunks)

    return all_chunks
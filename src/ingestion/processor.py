import hashlib
import json
from pathlib import Path
from typing import Any

from src.ingestion.document_loader import (
    SUPPORTED_EXTENSIONS,
    build_document_metadata,
    load_document,
)
from src.ingestion.chunker import chunk_documents


def create_chunk_id(chunk: dict[str, Any]) -> str:
    """
    Create a deterministic unique ID for a chunk.

    Same source + location + chunk position + text
    will generate the same ID on repeated processing.
    """

    metadata = chunk["metadata"]

    identity = "|".join(
        [
            metadata.get("document_id", metadata.get("source", "")),
            metadata.get("document_version", ""),
            str(metadata.get("page", "")),
            str(metadata.get("slide", "")),
            str(metadata.get("sheet", "")),
            str(metadata.get("chunk_index", "")),
            chunk["text"],
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:24]


def prepare_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add stable identifiers to processed chunks.
    """

    prepared = []

    for chunk in chunks:
        item = {
            "chunk_id": create_chunk_id(chunk),
            "text": chunk["text"],
            "metadata": chunk["metadata"],
        }

        prepared.append(item)

    return prepared


def save_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save processed chunks as JSON Lines.

    JSONL is convenient because each line represents
    one independent chunk.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def merge_incremental_chunks(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace changed document versions while retaining untouched documents."""
    incoming_document_ids = {
        item["metadata"].get("document_id") for item in incoming
        if item["metadata"].get("document_id")
    }
    retained = [
        item for item in existing
        if item.get("metadata", {}).get("document_id") not in incoming_document_ids
    ]
    merged = retained + incoming
    # Protect against duplicate records from retries or overlapping input roots.
    return list({item["chunk_id"]: item for item in merged}.values())


def process_corpus(
    raw_directory: Path,
    output_path: Path,
    incremental: bool = True,
) -> list[dict[str, Any]]:
    """
    Complete offline ingestion pipeline:

    raw files
        -> extraction
        -> cleaning/chunking
        -> IDs
        -> persisted JSONL
    """

    print("=" * 70)
    print("IOCL OFFLINE DOCUMENT PROCESSOR")
    print("=" * 70)

    existing = load_jsonl(output_path) if incremental else []
    existing_by_document: dict[str, list[dict[str, Any]]] = {}
    for item in existing:
        document_id = item.get("metadata", {}).get("document_id")
        if document_id:
            existing_by_document.setdefault(document_id, []).append(item)

    files = sorted(
        path for path in raw_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    retained_chunks: list[dict[str, Any]] = []
    changed_records: list[dict[str, Any]] = []
    unchanged_documents = 0

    for file_path in files:
        identity = build_document_metadata(file_path, raw_directory)
        prior = existing_by_document.get(identity["document_id"], [])
        prior_hash = (
            prior[0].get("metadata", {}).get("content_hash") if prior else None
        )
        if incremental and prior_hash == identity["content_hash"]:
            retained_chunks.extend(prior)
            unchanged_documents += 1
            continue
        changed_records.extend(load_document(file_path, corpus_root=raw_directory))

    print(
        f"Found {len(files)} document(s): {unchanged_documents} unchanged, "
        f"{len(files) - unchanged_documents} new/updated."
    )
    records = changed_records

    print("\nCreating chunks...")

    chunks = chunk_documents(
        records,
        max_chars=1800,
        overlap_chars=250,
    )

    print(f"Created {len(chunks)} chunks.")

    print("\nAssigning chunk IDs...")

    prepared_chunks = prepare_chunks(chunks)

    # This is a corpus snapshot: absent documents are removed, unchanged
    # documents are retained without reparsing, and changed versions replace old ones.
    final_chunks = merge_incremental_chunks(retained_chunks, prepared_chunks)

    save_jsonl(
        final_chunks,
        output_path,
    )

    print(
        f"\nSaved {len(final_chunks)} processed chunks to:"
    )
    print(output_path)

    return final_chunks

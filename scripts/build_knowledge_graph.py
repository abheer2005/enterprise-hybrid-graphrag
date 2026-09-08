import json
import time
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph.entity_extractor import KnowledgeExtractor
from src.graph.neo4j_store import Neo4jStore
from src.graph.extraction_cache import ExtractionCache


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

CACHE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kg"
    / "processed_chunks.jsonl"
)


# This controls how many currently-unprocessed chunks
# are attempted in a single script run.
#
# It is NOT tied to any document, department or domain.
BATCH_SIZE = 5

MAX_RETRIES = 4
BASE_RETRY_SECONDS = 10


def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            chunks.append(
                json.loads(line)
            )

    return chunks


def extract_with_retry(
    extractor: KnowledgeExtractor,
    text: str,
):
    """
    Run KG extraction with exponential backoff for
    temporary API/rate-limit failures.

    Permanent quota exhaustion may still fail after
    all retries. Such chunks are NOT cached, so they
    can be retried in a later run.
    """

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            return extractor.extract(text)

        except Exception as error:
            error_text = str(error).lower()

            retryable = any(
                marker in error_text
                for marker in [
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "rate_limit",
                    "timeout",
                    "temporarily unavailable",
                    "503",
                ]
            )

            if (
                not retryable
                or attempt == MAX_RETRIES
            ):
                raise

            wait_seconds = (
                BASE_RETRY_SECONDS
                * (2 ** (attempt - 1))
            )

            print(
                f"Temporary API/rate limit error. "
                f"Retrying in {wait_seconds}s "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            time.sleep(wait_seconds)


def main():
    print("=" * 70)
    print("IOCL INCREMENTAL KNOWLEDGE GRAPH BUILDER")
    print("=" * 70)

    chunks = load_chunks()

    print(
        f"\nCorpus chunks:       {len(chunks)}"
    )

    cache = ExtractionCache(
        CACHE_FILE
    )

    # --------------------------------------------------
    # Determine which chunks still require KG processing
    # --------------------------------------------------

    all_pending = [
        chunk
        for chunk in chunks
        if not cache.contains(
            chunk["chunk_id"]
        )
    ]

    processed_count = (
        len(chunks) - len(all_pending)
    )

    # Only take a small batch for this run.
    # Later this can come from configuration/environment.
    batch = all_pending[:BATCH_SIZE]

    print(
        f"Already processed:   {processed_count}"
    )

    print(
        f"Total pending:       {len(all_pending)}"
    )

    print(
        f"This batch:          {len(batch)}"
    )

    if not batch:
        print(
            "\nKnowledge graph is already up to date."
        )
        return

    extractor = KnowledgeExtractor()
    store = Neo4jStore()

    success_count = 0
    failure_count = 0

    total_entities = 0
    total_relationships = 0

    try:
        store.create_semantic_constraints()

        for index, chunk in enumerate(
            batch,
            start=1,
        ):
            metadata = chunk.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "unknown",
            )

            page = metadata.get(
                "page"
            )

            chunk_id = chunk[
                "chunk_id"
            ]

            print("\n" + "-" * 70)

            print(
                f"[{index}/{len(batch)}] "
                f"{source}"
            )

            print(
                f"Page:     {page}"
            )

            print(
                f"Chunk ID: {chunk_id}"
            )

            try:
                extraction = extract_with_retry(
                    extractor=extractor,
                    text=chunk["text"],
                )

                entity_count = len(
                    extraction.entities
                )

                relationship_count = len(
                    extraction.relationships
                )

                print(
                    f"Entities:      "
                    f"{entity_count}"
                )

                print(
                    f"Relationships: "
                    f"{relationship_count}"
                )

                # Write only validated extraction
                # to Neo4j.
                store.upsert_extraction(
                    chunk,
                    extraction,
                )

                # Cache ONLY after successful graph write.
                cache.mark_processed(
                    chunk_id=chunk_id,
                    source=source,
                )

                success_count += 1

                total_entities += (
                    entity_count
                )

                total_relationships += (
                    relationship_count
                )

                print(
                    "Status: SUCCESS"
                )

            except Exception as error:
                failure_count += 1

                print(
                    "Status: FAILED"
                )

                print(
                    f"Error: {error}"
                )

                print(
                    "Chunk was NOT marked as processed "
                    "and can be retried later."
                )

            # Small spacing between successful API calls.
            time.sleep(0.25)

        print("\n" + "=" * 70)
        print("KNOWLEDGE GRAPH BUILD COMPLETE")
        print("=" * 70)

        print(
            f"Successful chunks:   "
            f"{success_count}"
        )

        print(
            f"Failed chunks:       "
            f"{failure_count}"
        )

        print(
            f"Extracted entities:  "
            f"{total_entities}"
        )

        print(
            f"Extracted relations: "
            f"{total_relationships}"
        )

        remaining = (
            len(all_pending)
            - success_count
        )

        print(
            f"Still pending:       "
            f"{remaining}"
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()
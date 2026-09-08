import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph.entity_extractor import (
    KnowledgeExtractor,
)

from src.graph.neo4j_store import (
    Neo4jStore,
)


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)


TARGETS = [
    (
        "IOC_S&CSR_Policy.pdf",
        4,
    ),
    (
        "RPT_Policy.pdf",
        2,
    ),
]


def load_test_chunks():
    selected = []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            record = json.loads(line)

            source = record["metadata"].get(
                "source"
            )

            page = record["metadata"].get(
                "page"
            )

            if (source, page) in TARGETS:
                selected.append(record)

    return selected[:3]


def main():
    print("=" * 70)
    print("IOCL SEMANTIC GRAPH TEST")
    print("=" * 70)

    chunks = load_test_chunks()

    print(
        f"\nSelected {len(chunks)} chunks."
    )

    extractor = KnowledgeExtractor()
    store = Neo4jStore()

    try:
        store.create_semantic_constraints()

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            metadata = chunk["metadata"]

            print("\n" + "-" * 70)

            print(
                f"[{index}/{len(chunks)}] "
                f"{metadata.get('source')} "
                f"page {metadata.get('page')}"
            )

            extraction = extractor.extract(
                chunk["text"]
            )

            print(
                f"Entities: "
                f"{len(extraction.entities)}"
            )

            print(
                f"Relationships: "
                f"{len(extraction.relationships)}"
            )

            store.upsert_extraction(
                chunk,
                extraction,
            )

            print("Written to Neo4j.")

        print("\n" + "=" * 70)
        print("SEMANTIC GRAPH TEST COMPLETE")
        print("=" * 70)

    finally:
        store.close()


if __name__ == "__main__":
    main()
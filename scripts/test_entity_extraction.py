import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph.entity_extractor import (
    KnowledgeExtractor,
)


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)


TARGETS = [
    (
        "Whistle_Blower_policy.pdf",
        7,
    ),
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

    # Only three API calls for this test.
    return selected[:3]


def main():
    print("=" * 70)
    print("IOCL KNOWLEDGE EXTRACTION TEST")
    print("=" * 70)

    chunks = load_test_chunks()

    print(
        f"\nSelected {len(chunks)} test chunks."
    )

    extractor = KnowledgeExtractor()

    for number, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk["metadata"]

        print("\n" + "=" * 70)
        print(f"TEST CHUNK {number}")
        print("=" * 70)

        print(
            f"Source: {metadata.get('source')}"
        )

        print(
            f"Page:   {metadata.get('page')}"
        )

        print(
            f"ID:     {chunk['chunk_id']}"
        )

        extraction = extractor.extract(
            chunk["text"]
        )

        print("\nENTITIES")

        for entity in extraction.entities:
            print(
                f"  [{entity.type}] "
                f"{entity.name}"
            )

        print("\nRELATIONSHIPS")

        for relationship in (
            extraction.relationships
        ):
            print(
                "  "
                f"{relationship.source}"
                f" --{relationship.relation}--> "
                f"{relationship.target}"
            )

            if relationship.evidence:
                print(
                    f"      Evidence: "
                    f"{relationship.evidence}"
                )


if __name__ == "__main__":
    main()
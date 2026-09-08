import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph.neo4j_store import Neo4jStore


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)


def load_chunks():
    records = []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def main():
    print("=" * 70)
    print("IOCL BASE KNOWLEDGE GRAPH BUILDER")
    print("=" * 70)

    records = load_chunks()

    print(f"\nLoaded {len(records)} chunks.")

    store = Neo4jStore()

    try:
        print("Creating constraints...")
        store.create_constraints()

        print("Writing documents and chunks...")

        for index, record in enumerate(
            records,
            start=1,
        ):
            store.upsert_document_and_chunk(
                record
            )

            if (
                index % 25 == 0
                or index == len(records)
            ):
                print(
                    f"  Processed "
                    f"{index}/{len(records)}"
                )

        stats = store.get_graph_stats()

        print("\n" + "=" * 70)
        print("BASE GRAPH COMPLETE")
        print("=" * 70)

        print(
            f"Nodes:         "
            f"{stats['nodes']}"
        )

        print(
            f"Relationships: "
            f"{stats['relationships']}"
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.vector_store import VectorStore


CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

VECTOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vector"
)

INDEX_FILE = VECTOR_DIR / "chunks.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.json"


def main():
    print("=" * 70)
    print("IOCL VECTOR INDEX BUILDER")
    print("=" * 70)

    store = VectorStore()

    store.build_incremental(
        CHUNKS_FILE,
        INDEX_FILE,
        METADATA_FILE,
    )

    store.save(
        INDEX_FILE,
        METADATA_FILE,
    )

    print("\n" + "=" * 70)
    print("VECTOR INDEX COMPLETE")
    print("=" * 70)

    print(f"Vectors: {store.index.ntotal}")
    print(f"Index:   {INDEX_FILE}")
    print(f"Metadata:{METADATA_FILE}")


if __name__ == "__main__":
    main()

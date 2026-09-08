from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.vector_store import VectorStore


VECTOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vector"
)

INDEX_FILE = VECTOR_DIR / "chunks.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.json"


def print_results(query: str, results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nRESULT #{rank}")
        print("-" * 80)

        print(f"Score:    {result['score']:.4f}")
        print(f"Source:   {metadata.get('source')}")

        if metadata.get("page") is not None:
            print(f"Page:     {metadata.get('page')}")

        if metadata.get("slide") is not None:
            print(f"Slide:    {metadata.get('slide')}")

        if metadata.get("sheet") is not None:
            print(f"Sheet:    {metadata.get('sheet')}")

        print(f"Chunk ID: {result['chunk_id']}")

        print("\nTEXT:")
        print(result["text"][:700])


def main():
    print("=" * 80)
    print("IOCL VECTOR RETRIEVAL TEST")
    print("=" * 80)

    store = VectorStore()

    store.load(
        INDEX_FILE,
        METADATA_FILE,
    )

    test_queries = [
        "What standards of ethical conduct should senior management follow?",

        "How does the whistle blower mechanism protect employees who report wrongdoing?",

        "What is IOCL's policy regarding preservation of documents?",

        "What are the rules concerning related party transactions?",

        "What are the company's sustainability and CSR responsibilities?",
    ]

    for query in test_queries:

        results = store.search(
            query,
            top_k=3,
        )

        print_results(
            query,
            results,
        )


if __name__ == "__main__":
    main()
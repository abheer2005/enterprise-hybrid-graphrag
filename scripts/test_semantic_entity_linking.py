from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.retrieval.vector_store import (
    VectorStore,
)

from src.retrieval.semantic_entity_linker import (
    SemanticEntityLinker,
)


VECTOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vector"
)

INDEX_FILE = (
    VECTOR_DIR
    / "chunks.faiss"
)

METADATA_FILE = (
    VECTOR_DIR
    / "metadata.json"
)


def main():

    print("=" * 80)
    print(
        "IOCL SEMANTIC ENTITY LINKING TEST"
    )
    print("=" * 80)

    vector_store = VectorStore()

    vector_store.load(
        INDEX_FILE,
        METADATA_FILE,
    )

    linker = SemanticEntityLinker(
        embedding_model=vector_store.model
    )

    print(
        f"\nLoaded graph entities: "
        f"{len(linker.entities)}"
    )

    test_queries = [

        "Who do Senior Management Personnel report to?",

        "What obligations do Officers have?",

        "What does the Code say about Whole Time Directors?",

        "Who are the senior executives accountable to?",

        "What duties apply to company officers?",

        "What rules govern related party transactions?",

        "How are whistleblowers protected?",

        "What are the company's CSR responsibilities?",
    ]

    try:

        for query in test_queries:

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"QUERY: {query}"
            )

            print(
                "=" * 80
            )

            results = linker.search(
                query=query,
                top_k=8,
                min_score=0.20,
            )

            if not results:

                print(
                    "\nNo semantic entities found."
                )

                continue

            for rank, result in enumerate(
                results,
                start=1,
            ):

                print(
                    f"\nRESULT #{rank}"
                )

                print(
                    "-" * 80
                )

                print(
                    f"Score: "
                    f"{result['semantic_score']:.4f}"
                )

                print(
                    f"Entity: "
                    f"{result.get('name')}"
                )

                print(
                    f"Type:   "
                    f"{result.get('type')}"
                )

                description = (
                    result.get(
                        "description"
                    )
                )

                if description:

                    print(
                        f"Description: "
                        f"{description}"
                    )

    finally:

        linker.close()


if __name__ == "__main__":
    main()
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from src.generation.context_builder import (
    ContextBuilder,
)


QUERIES = [
    "How are whistleblowers protected?",
    "What are the company's CSR responsibilities?",
    "What rules govern related party transactions?",
]


def main():

    print("=" * 80)
    print("IOCL CONTEXT BUILDER TEST")
    print("=" * 80)

    retriever = HybridRetriever()

    builder = ContextBuilder(
        max_chunks=5,
        max_chars=10000,
    )

    try:

        for query in QUERIES:

            print("\n\n" + "=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)

            retrieval = retriever.retrieve(
                query=query,
                vector_top_k=8,
                entity_limit=8,
                relation_limit=20,
                semantic_min_score=0.25,
                final_top_k=8,
            )

            result = builder.build(
                retrieval["candidates"]
            )

            print("\nCONTEXT STATS")

            for key, value in result[
                "stats"
            ].items():

                print(
                    f"  {key}: {value}"
                )

            print("\nSELECTED EVIDENCE")

            for evidence in result[
                "evidence"
            ]:

                metadata = evidence.get(
                    "metadata",
                    {},
                )

                print("\n" + "-" * 80)

                print(
                    f"Evidence "
                    f"{evidence['evidence_id']}"
                )

                print(
                    f"Source: "
                    f"{metadata.get('source')}"
                )

                print(
                    f"Page: "
                    f"{metadata.get('page')}"
                )

                print(
                    f"Chunk: "
                    f"{evidence.get('chunk_id')}"
                )

                score = evidence.get(
                    "rerank_score"
                )

                if score is not None:
                    print(
                        f"Rerank score: "
                        f"{score:.4f}"
                    )

                text = (
                    evidence.get("text")
                    or ""
                )

                print("\nTEXT:")
                print(
                    text[:400]
                )

    finally:

        retriever.close()

    print("\n" + "=" * 80)
    print("CONTEXT BUILDER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
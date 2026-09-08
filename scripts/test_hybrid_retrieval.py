from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.hybrid_retriever import HybridRetriever


QUERIES = [
    "Who do Senior Management Personnel report to?",
    "Who are the senior executives accountable to?",
    "What duties apply to company officers?",
    "How are whistleblowers protected?",
    "What are the company's CSR responsibilities?",
    "What rules govern related party transactions?",
]


def main():

    print("=" * 80)
    print("IOCL HYBRID RETRIEVAL TEST")
    print("=" * 80)

    retriever = HybridRetriever()

    try:

        for query in QUERIES:

            print("\n\n" + "=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)

            result = retriever.retrieve(
                query=query,
                vector_top_k=8,
                entity_limit=8,
                relation_limit=20,
                semantic_min_score=0.25,
                final_top_k=8,
            )

            stats = result["stats"]

            print("\nRETRIEVAL STATS")

            print(
                f"  Vector candidates: "
                f"{stats['vector_candidates']}"
            )

            print(
                f"  Graph chunks:      "
                f"{stats['graph_chunks']}"
            )

            print(
                f"  Unique candidates: "
                f"{stats['unique_candidates']}"
            )

            print(
                f"  Final candidates:  "
                f"{stats['returned_candidates']}"
            )

            print("\nLINKED GRAPH ENTITIES")

            entities = (
                result["graph"].get(
                    "entities",
                    [],
                )
            )

            if not entities:
                print("  None")

            for entity in entities[:5]:

                print(
                    f"  [{entity.get('type')}] "
                    f"{entity.get('name')} "
                    f"| link={entity.get('entity_link_score', 0):.4f} "
                    f"| sources={entity.get('entity_link_sources')}"
                )

            print("\nFINAL HYBRID CANDIDATES")

            for rank, candidate in enumerate(
                result["candidates"],
                start=1,
            ):

                metadata = candidate.get(
                    "metadata",
                    {},
                )

                print("\n" + "-" * 80)

                print(f"RANK #{rank}")

                print(
                    f"Rerank score: "
                    f"{candidate.get('rerank_score', 0.0):.4f}"
                )

                print(
                    f"Semantic score: "
                    f"{candidate.get('semantic_rerank_score', 0.0):.4f}"
                )

                print(
                    f"Old hybrid score: "
                    f"{candidate.get('hybrid_score', 0.0):.4f}"
                )

                print(
                    f"Graph support: "
                    f"{candidate.get('graph_support_score', 0.0):.4f}"
                )

                print(
                    f"Retrieval agreement: "
                    f"{candidate.get('retrieval_agreement', 0.0):.4f}"
                )

                vector_score = candidate.get(
                    "vector_score"
                )

                if vector_score is not None:
                    print(
                        f"Vector score: "
                        f"{vector_score:.4f}"
                    )
                else:
                    print(
                        "Vector score: None"
                    )

                print(
                    f"Graph relations: "
                    f"{candidate.get('graph_relation_count', 0)}"
                )

                print(
                    f"Signals: "
                    f"{candidate.get('retrieval_sources')}"
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
                    f"{candidate.get('chunk_id')}"
                )

                text = (
                    candidate.get("text")
                    or ""
                )

                print("\nTEXT:")
                print(
                    text[:500]
                )

        print("\n\n" + "=" * 80)
        print("HYBRID RETRIEVAL TEST COMPLETE")
        print("=" * 80)

    finally:

        retriever.close()


if __name__ == "__main__":
    main()
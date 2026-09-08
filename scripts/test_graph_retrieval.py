from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_retriever import GraphRetriever


QUERIES = [
    # Exact / near-exact terminology
    "What obligations do Officers have?",

    "What is required from Senior Management Personnel?",

    "Who do Senior Management Personnel report to?",

    "What does the Code say about Whole Time Directors?",

    # Paraphrase tests
    "Who are the senior executives accountable to?",

    "What duties apply to company officers?",

    # Cross-domain tests.
    # These may currently be weak because the KG is
    # not yet complete. That is useful diagnostic evidence.
    "What rules govern related party transactions?",

    "How are whistleblowers protected?",

    "What are the company's CSR responsibilities?",
]


def main():

    print("=" * 80)
    print("IOCL GRAPH RETRIEVAL V2 TEST")
    print("LITERAL + SEMANTIC ENTITY LINKING")
    print("=" * 80)

    # -----------------------------------------------------
    # Load ONE embedding model.
    #
    # The same model is reused by the semantic entity
    # linker instead of loading a second copy.
    # -----------------------------------------------------

    vector_store = VectorStore()

    retriever = GraphRetriever(
        embedding_model=vector_store.model
    )

    try:

        for query in QUERIES:

            print("\n\n")
            print("=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)

            result = retriever.retrieve(
                query=query,
                entity_limit=8,
                relation_limit=30,
                semantic_min_score=0.25,
            )

            # -------------------------------------------------
            # ENTITY LINKING
            # -------------------------------------------------

            print("\nMATCHED ENTITIES")

            if not result["entities"]:
                print("  None")

            for rank, entity in enumerate(
                result["entities"],
                start=1,
            ):

                print(
                    f"\n  #{rank} "
                    f"[{entity.get('type')}] "
                    f"{entity.get('name')}"
                )

                print(
                    f"     Link sources: "
                    f"{entity.get('entity_link_sources')}"
                )

                print(
                    f"     Literal: "
                    f"{entity.get('literal_match')}"
                )

                semantic_score = entity.get(
                    "semantic_score"
                )

                if semantic_score is not None:

                    print(
                        f"     Semantic score: "
                        f"{semantic_score:.4f}"
                    )

                print(
                    f"     Final link score: "
                    f"{entity.get('entity_link_score', 0.0):.4f}"
                )

                description = (
                    entity.get("description")
                )

                if description:

                    print(
                        f"     Description: "
                        f"{description}"
                    )

            # -------------------------------------------------
            # GRAPH RELATIONSHIPS
            # -------------------------------------------------

            print("\nGRAPH RELATIONSHIPS")

            if not result["relationships"]:
                print("  None")

            for relation in result[
                "relationships"
            ][:20]:

                print(
                    f"\n  "
                    f"{relation.get('source_entity')} "
                    f"--{relation.get('relationship')}--> "
                    f"{relation.get('target_entity')}"
                )

                print(
                    f"    Evidence: "
                    f"{relation.get('evidence')}"
                )

                print(
                    f"    Source: "
                    f"{relation.get('source')} "
                    f"(page {relation.get('page')})"
                )

            # -------------------------------------------------
            # PROVENANCE
            # -------------------------------------------------

            print("\nPROVENANCE CHUNKS")

            print(
                f"  Retrieved: "
                f"{len(result['chunks'])}"
            )

            for chunk in result[
                "chunks"
            ][:5]:

                text = (
                    chunk.get("text")
                    or ""
                )

                preview = text[:350]

                print("\n  " + "-" * 70)

                print(
                    f"  Source: "
                    f"{chunk.get('source')}"
                )

                print(
                    f"  Page: "
                    f"{chunk.get('page')}"
                )

                print(
                    f"  Chunk: "
                    f"{chunk.get('chunk_id')}"
                )

                print(
                    f"\n  {preview}"
                )

        print("\n\n")
        print("=" * 80)
        print("GRAPH RETRIEVAL V2 TEST COMPLETE")
        print("=" * 80)

    finally:

        retriever.close()


if __name__ == "__main__":
    main()
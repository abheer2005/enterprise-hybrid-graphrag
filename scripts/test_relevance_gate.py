from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from src.retrieval.reranker import (
    HybridReranker,
)

from src.retrieval.relevance_gate import (
    RelevanceGate,
)


QUERIES = [
    # Direct in-domain
    "How are whistleblowers protected?",
    "What are the company's CSR responsibilities?",
    "What rules govern related party transactions?",
    "Who do Senior Management Personnel report to?",
    "What duties apply to company officers?",

    # Paraphrased / conversational in-domain
    "If an employee reports wrongdoing, what protection do they get?",
    "Explain what IndianOil is supposed to do under CSR.",
    "When does an RPT need approval?",
    "What responsibilities does an officer have?",
    "Tell me about rules for transactions with related parties.",

    # Broad in-domain
    "Explain the whistleblower policy.",
    "Summarize the CSR policy.",
    "What does the code of conduct say?",
    "Explain IndianOil's related party transaction policy.",

    # Negative / absence questions
    "Does the whistleblower policy mention financial rewards?",
    "Does the CSR policy allow normal business activities to use CSR funds?",
    "Does the RPT policy require Audit Committee approval?",

    # Potentially partially covered
    "What happens if a whistleblower is unhappy with how the complaint was handled?",
    "How does IndianOil deal with unspent CSR money?",
    "What environmental responsibilities are mentioned in company policies?",

    # Completely unrelated
    "Who is M. S. Dhoni?",
    "What is the capital of France?",
    "How do I make chocolate cake?",
    "Who won the FIFA World Cup?",
    "Explain photosynthesis.",
    "Write a Python program to sort numbers.",
    "What is the distance between Earth and Mars?",
]


def main():

    print("=" * 80)
    print("IOCL RETRIEVAL RELEVANCE GATE TEST")
    print("=" * 80)

    retriever = HybridRetriever()

    reranker = HybridReranker(
        embedding_model=(
            retriever.vector_store.model
        ),
    )

    gate = RelevanceGate()

    try:

        for query in QUERIES:

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"QUESTION: {query}"
            )

            print(
                "=" * 80
            )

            retrieval = retriever.retrieve(
                query=query,
                vector_top_k=10,
                entity_limit=10,
                relation_limit=30,
                semantic_min_score=0.25,
                final_top_k=12,
            )

            candidates = retrieval.get(
                "candidates",
                [],
            )

            reranked = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=8,
            )

            decision = gate.evaluate(
                reranked
            )

            print(
                f"Status:                  "
                f"{decision.status}"
            )

            print(
                f"Allow generation:        "
                f"{decision.allow_generation}"
            )

            print(
                f"Top semantic score:      "
                f"{decision.top_semantic_score:.4f}"
            )

            print(
                f"Mean top semantic score: "
                f"{decision.mean_top_semantic_score:.4f}"
            )

            print(
                f"Top rerank score:        "
                f"{decision.top_rerank_score:.4f}"
            )

            print(
                f"Mean top rerank score:   "
                f"{decision.mean_top_rerank_score:.4f}"
            )

            print(
                f"Strong matches:          "
                f"{decision.strong_semantic_matches}"
            )

            print(
                f"Moderate matches:        "
                f"{decision.moderate_semantic_matches}"
            )

            print(
                f"Vector+graph agreements: "
                f"{decision.vector_graph_agreements}"
            )

            print(
                f"Reason: {decision.reason}"
            )

            print(
                "\nTOP RETRIEVED CHUNKS"
            )

            print(
                "-" * 80
            )

            for index, item in enumerate(
                reranked[:3],
                start=1,
            ):

                metadata = (
                    item.get(
                        "metadata",
                        {},
                    )
                    or {}
                )

                print(
                    f"\n#{index}"
                )

                print(
                    "Semantic: "
                    f"{item.get('semantic_rerank_score', 0):.4f}"
                )

                print(
                    "Rerank:   "
                    f"{item.get('rerank_score', 0):.4f}"
                )

                print(
                    "Source:   "
                    f"{metadata.get('source')}"
                )

                text = (
                    item.get("text")
                    or ""
                )

                print(
                    "Text:     "
                    f"{text[:250]}"
                )

    finally:

        retriever.close()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RELEVANCE GATE TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
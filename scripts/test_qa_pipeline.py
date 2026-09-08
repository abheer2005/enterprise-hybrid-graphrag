from pathlib import Path
import os
import sys

from dotenv import load_dotenv


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


from src.generation.qa_pipeline import (
    QAPipeline,
)


load_dotenv(
    PROJECT_ROOT / ".env"
)


# =========================================================
# TEST QUERIES
# =========================================================
#
# These deliberately cover:
#
# 1. Whistleblower policy
# 2. CSR policy
# 3. Related Party Transactions
# 4. Senior Management Personnel
# 5. Officer duties
# 6. Completely out-of-corpus question
#
# The final query is important because the system should
# NOT answer from general model knowledge when the IOCL
# corpus contains no supporting evidence.
# =========================================================

QUERIES = [
    "How are whistleblowers protected?",

    "What are the company's CSR responsibilities?",

    "What rules govern related party transactions?",

    "Who do Senior Management Personnel report to?",

    "What duties apply to company officers?",

    "Who is M. S. Dhoni?",
]


def print_separator():
    print(
        "\n"
        + "=" * 80
    )


def main():

    print("=" * 80)
    print("IOCL END-TO-END GRAPHRAG QA TEST")
    print("=" * 80)

    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in .env"
        )

    # -----------------------------------------------------
    # INITIALIZE PIPELINE ONCE
    #
    # Important:
    # Do NOT recreate this for every question because the
    # vector embedding model and NLI model are expensive
    # to load.
    # -----------------------------------------------------

    pipeline = QAPipeline(
        api_key=api_key,
        max_context_chunks=5,
        max_context_chars=10000,
        min_entailment_score=0.70,
    )

    try:

        for query in QUERIES:

            print_separator()

            print(
                f"QUESTION: {query}"
            )

            print(
                "=" * 80
            )

            result = pipeline.answer(
                query=query,
                vector_top_k=10,
                entity_limit=10,
                relation_limit=30,
                semantic_min_score=0.25,
                hybrid_top_k=12,
                rerank_top_k=8,
            )

            # ---------------------------------------------
            # FINAL ANSWER
            # ---------------------------------------------

            print(
                "\nFINAL ANSWER"
            )

            print(
                "-" * 80
            )

            print(
                result.answer
            )

            # ---------------------------------------------
            # DECISION
            # ---------------------------------------------

            print(
                "\nDECISION"
            )

            print(
                "-" * 80
            )

            print(
                f"Accepted:              "
                f"{result.accepted}"
            )

            print(
                f"Grounded:              "
                f"{result.grounded}"
            )

            print(
                f"Insufficient evidence: "
                f"{result.insufficient_evidence}"
            )

            # ---------------------------------------------
            # CITATION VALIDATION
            # ---------------------------------------------

            citation = (
                result.citation_validation
            )

            print(
                "\nCITATION VALIDATION"
            )

            print(
                "-" * 80
            )

            print(
                f"Passed:            "
                f"{citation.get('validation_passed')}"
            )

            print(
                f"Citation valid:    "
                f"{citation.get('citation_valid')}"
            )

            print(
                f"Cited evidence:    "
                f"{citation.get('cited_evidence_ids')}"
            )

            print(
                f"Invalid citations: "
                f"{citation.get('invalid_citation_ids')}"
            )

            # ---------------------------------------------
            # NLI GROUNDING
            # ---------------------------------------------

            grounding = (
                result.grounding_validation
            )

            print(
                "\nNLI GROUNDING"
            )

            print(
                "-" * 80
            )

            print(
                f"Passed:             "
                f"{grounding.get('passed')}"
            )

            print(
                f"Claims checked:     "
                f"{grounding.get('claims_checked')}"
            )

            print(
                f"Supported claims:   "
                f"{grounding.get('supported_claims')}"
            )

            print(
                f"Unsupported claims: "
                f"{grounding.get('unsupported_claims')}"
            )

            if grounding.get(
                "skipped"
            ):

                print(
                    f"Skipped:            "
                    f"{grounding.get('reason')}"
                )

            # ---------------------------------------------
            # CLAIM DETAILS
            # ---------------------------------------------

            claims = grounding.get(
                "claims",
                [],
            )

            if claims:

                print(
                    "\nCLAIMS"
                )

                for index, claim in enumerate(
                    claims,
                    start=1,
                ):

                    print(
                        "\n"
                        + "." * 80
                    )

                    print(
                        f"Claim #{index}"
                    )

                    print(
                        f"Text: "
                        f"{claim.get('claim')}"
                    )

                    print(
                        f"Citations: "
                        f"{claim.get('citations')}"
                    )

                    print(
                        f"Label: "
                        f"{claim.get('label')}"
                    )

                    print(
                        f"Entailment: "
                        f"{claim.get('entailment_score', 0):.4f}"
                    )

                    print(
                        f"Contradiction: "
                        f"{claim.get('contradiction_score', 0):.4f}"
                    )

                    print(
                        f"Neutral: "
                        f"{claim.get('neutral_score', 0):.4f}"
                    )

                    print(
                        f"Supported: "
                        f"{claim.get('supported')}"
                    )

            # ---------------------------------------------
            # REPAIR
            # ---------------------------------------------

            print(
                "\nREPAIR"
            )

            print(
                "-" * 80
            )

            repair = (
                result.repair
                or {}
            )

            print(
                f"Attempted:          "
                f"{repair.get('attempted', False)}"
            )

            print(
                f"Changed:            "
                f"{repair.get('changed', False)}"
            )

            print(
                f"Accepted repair:    "
                f"{repair.get('accepted', False)}"
            )

            print(
                f"Supported claims:   "
                f"{repair.get('supported_claims', 0)}"
            )

            print(
                f"Unsupported claims: "
                f"{repair.get('unsupported_claims', 0)}"
            )

            removed = repair.get(
                "removed_fragments",
                [],
            )

            print(
                f"Removed fragments:  "
                f"{len(removed)}"
            )

            for fragment in removed:
                print(
                    f"  - {fragment}"
                )

            # ---------------------------------------------
            # SOURCES
            # ---------------------------------------------

            print(
                "\nCITED SOURCES"
            )

            print(
                "-" * 80
            )

            if not result.sources:
                print(
                    "None"
                )

            for source in result.sources:

                source_name = (
                    source.get("source")
                    or "Unknown source"
                )

                location = []

                if (
                    source.get("page")
                    is not None
                ):

                    location.append(
                        f"page "
                        f"{source.get('page')}"
                    )

                if (
                    source.get("slide")
                    is not None
                ):

                    location.append(
                        f"slide "
                        f"{source.get('slide')}"
                    )

                if (
                    source.get("sheet")
                    is not None
                ):

                    location.append(
                        f"sheet "
                        f"{source.get('sheet')}"
                    )

                location_text = (
                    " | ".join(location)
                )

                if location_text:

                    print(
                        f"- {source_name} "
                        f"| {location_text}"
                    )

                else:

                    print(
                        f"- {source_name}"
                    )

            # ---------------------------------------------
            # PIPELINE STATS
            # ---------------------------------------------

            print(
                "\nPIPELINE STATS"
            )

            print(
                "-" * 80
            )

            for key, value in (
                result.retrieval_stats.items()
            ):

                print(
                    f"{key}: {value}"
                )

            for key, value in (
                result.context_stats.items()
            ):

                print(
                    f"context_{key}: {value}"
                )

            # ---------------------------------------------
            # FAILURE REASONS
            # ---------------------------------------------

            if result.failure_reasons:

                print(
                    "\nFAILURE REASONS"
                )

                print(
                    "-" * 80
                )

                for reason in (
                    result.failure_reasons
                ):

                    print(
                        f"- {reason}"
                    )

        # -------------------------------------------------
        # ALL QUERIES COMPLETE
        # -------------------------------------------------

        print_separator()

        print(
            "END-TO-END QA TEST COMPLETE"
        )

        print(
            "=" * 80
        )

    finally:

        pipeline.close()


if __name__ == "__main__":
    main()
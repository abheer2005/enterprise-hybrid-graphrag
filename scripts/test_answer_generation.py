from pathlib import Path
import os
import sys

from dotenv import load_dotenv


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

from src.generation.answer_generator import (
    AnswerGenerator,
)

from src.generation.answer_validator import (
    AnswerValidator,
)


load_dotenv(
    PROJECT_ROOT / ".env"
)


QUERIES = [
    "How are whistleblowers protected?",
    "What are the company's CSR responsibilities?",
    "What rules govern related party transactions?",
]


def print_source(
    source: dict,
) -> None:
    """
    Print one validated source reference.
    """

    source_name = (
        source.get("source")
        or "Unknown source"
    )

    parts = [
        source_name
    ]

    if source.get("page") is not None:
        parts.append(
            f"page {source.get('page')}"
        )

    if source.get("slide") is not None:
        parts.append(
            f"slide {source.get('slide')}"
        )

    if source.get("sheet") is not None:
        parts.append(
            f"sheet {source.get('sheet')}"
        )

    chunk_id = source.get(
        "chunk_id"
    )

    if chunk_id:
        parts.append(
            f"chunk {chunk_id}"
        )

    print(
        "  - "
        + " | ".join(parts)
    )


def main():

    print("=" * 80)
    print(
        "IOCL GROUNDED ANSWER GENERATION "
        "+ VALIDATION TEST"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # 1. LOAD GEMINI API KEY
    # -----------------------------------------------------

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in .env"
        )

    # -----------------------------------------------------
    # 2. INITIALIZE PIPELINE COMPONENTS
    # -----------------------------------------------------

    retriever = HybridRetriever()

    context_builder = ContextBuilder(
        max_chunks=5,
        max_chars=10000,
    )

    generator = AnswerGenerator(
        api_key=api_key,
    )

    validator = AnswerValidator(
        require_citations=True,
    )

    try:

        for query in QUERIES:

            print(
                "\n\n"
                + "=" * 80
            )

            print(
                f"QUESTION: {query}"
            )

            print(
                "=" * 80
            )

            # -------------------------------------------------
            # 3. HYBRID RETRIEVAL
            # -------------------------------------------------

            retrieval = retriever.retrieve(
                query=query,
                vector_top_k=8,
                entity_limit=8,
                relation_limit=20,
                semantic_min_score=0.25,
                final_top_k=8,
            )

            print(
                "\nHYBRID CANDIDATES: "
                f"{len(retrieval['candidates'])}"
            )

            # -------------------------------------------------
            # 4. CONTEXT CONSTRUCTION
            # -------------------------------------------------

            context_result = (
                context_builder.build(
                    candidates=(
                        retrieval[
                            "candidates"
                        ]
                    ),
                )
            )

            selected_evidence = (
                context_result[
                    "evidence"
                ]
            )

            stats = (
                context_result[
                    "stats"
                ]
            )

            print(
                "\nCONTEXT STATS"
            )

            print(
                f"  Input candidates: "
                f"{stats['input_candidates']}"
            )

            print(
                f"  Selected chunks:  "
                f"{stats['selected_chunks']}"
            )

            print(
                f"  Context chars:     "
                f"{stats['context_chars']}"
            )

            print(
                f"  Max chunks:        "
                f"{stats['max_chunks']}"
            )

            print(
                f"  Max chars:         "
                f"{stats['max_chars']}"
            )

            # -------------------------------------------------
            # 5. SHOW SELECTED EVIDENCE
            # -------------------------------------------------

            print(
                "\nSELECTED EVIDENCE"
            )

            if not selected_evidence:
                print(
                    "  None"
                )

            for evidence in selected_evidence:

                metadata = (
                    evidence.get(
                        "metadata",
                        {},
                    )
                    or {}
                )

                print(
                    "\n"
                    + "-" * 80
                )

                print(
                    f"Evidence "
                    f"{evidence.get('evidence_id')}"
                )

                print(
                    f"Source: "
                    f"{metadata.get('source')}"
                )

                if (
                    metadata.get("page")
                    is not None
                ):
                    print(
                        f"Page: "
                        f"{metadata.get('page')}"
                    )

                if (
                    metadata.get("slide")
                    is not None
                ):
                    print(
                        f"Slide: "
                        f"{metadata.get('slide')}"
                    )

                if (
                    metadata.get("sheet")
                    is not None
                ):
                    print(
                        f"Sheet: "
                        f"{metadata.get('sheet')}"
                    )

                print(
                    f"Chunk: "
                    f"{evidence.get('chunk_id')}"
                )

                rerank_score = (
                    evidence.get(
                        "rerank_score"
                    )
                )

                if rerank_score is not None:

                    print(
                        f"Rerank score: "
                        f"{rerank_score:.4f}"
                    )

            # -------------------------------------------------
            # 6. GROUNDED ANSWER GENERATION
            # -------------------------------------------------

            generated = generator.generate(
                query=query,
                selected_chunks=selected_evidence,
            )

            print(
                "\n"
                + "=" * 80
            )

            print(
                "GENERATED ANSWER"
            )

            print(
                "=" * 80
            )

            print(
                generated.answer
            )

            print(
                "\nGENERATOR GROUNDED: "
                f"{generated.grounded}"
            )

            # -------------------------------------------------
            # 7. VALIDATE GENERATED ANSWER
            # -------------------------------------------------

            validation = (
                validator.validate(
                    answer=generated.answer,
                    evidence=selected_evidence,
                )
            )

            print(
                "\n"
                + "=" * 80
            )

            print(
                "ANSWER VALIDATION"
            )

            print(
                "=" * 80
            )

            print(
                "Validation passed:   "
                f"{validation['validation_passed']}"
            )

            print(
                "Citation valid:      "
                f"{validation['citation_valid']}"
            )

            print(
                "Insufficient:        "
                f"{validation['insufficient_evidence']}"
            )

            print(
                "Cited evidence IDs:  "
                f"{validation['cited_evidence_ids']}"
            )

            print(
                "Invalid citations:   "
                f"{validation['invalid_citation_ids']}"
            )

            print(
                "Unused evidence IDs: "
                f"{validation['unused_evidence_ids']}"
            )

            # -------------------------------------------------
            # 8. VALIDATION ERRORS
            # -------------------------------------------------

            print(
                "\nVALIDATION ERRORS"
            )

            if not validation["errors"]:
                print(
                    "  None"
                )

            for error in validation[
                "errors"
            ]:
                print(
                    f"  - {error}"
                )

            # -------------------------------------------------
            # 9. VALIDATION WARNINGS
            # -------------------------------------------------

            print(
                "\nVALIDATION WARNINGS"
            )

            if not validation["warnings"]:
                print(
                    "  None"
                )

            for warning in validation[
                "warnings"
            ]:
                print(
                    f"  - {warning}"
                )

            # -------------------------------------------------
            # 10. FINAL SOURCE SELECTION
            # -------------------------------------------------
            #
            # IMPORTANT:
            #
            # Do NOT use:
            #
            #     generated.sources
            #
            # here.
            #
            # generated.sources contains all chunks supplied
            # to the LLM.
            #
            # validation["cited_sources"] contains only the
            # evidence actually cited by the generated answer.
            # -------------------------------------------------

            final_sources = (
                validation[
                    "cited_sources"
                ]
            )

            print(
                "\nVALIDATED SOURCES"
            )

            if not final_sources:
                print(
                    "  None"
                )

            for source in final_sources:
                print_source(
                    source
                )

            # -------------------------------------------------
            # 11. FINAL PIPELINE STATUS
            # -------------------------------------------------

            print(
                "\n"
                + "-" * 80
            )

            if validation[
                "validation_passed"
            ]:

                if validation[
                    "insufficient_evidence"
                ]:

                    print(
                        "PIPELINE STATUS: "
                        "VALID ABSTENTION"
                    )

                else:

                    print(
                        "PIPELINE STATUS: "
                        "CITATION VALIDATED"
                    )

            else:

                print(
                    "PIPELINE STATUS: "
                    "VALIDATION FAILED"
                )

                print(
                    "The generated answer must NOT "
                    "be returned as a trusted final "
                    "answer."
                )

        print(
            "\n\n"
            + "=" * 80
        )

        print(
            "ANSWER GENERATION + VALIDATION "
            "TEST COMPLETE"
        )

        print(
            "=" * 80
        )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()
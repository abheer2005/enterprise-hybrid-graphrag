import os
import time
import traceback

from dotenv import load_dotenv

from src.generation.qa_pipeline import QAPipeline


QUERY = "How are whistleblowers protected?"


def separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found."
        )

    separator(
        "IOCL REAL GENERATION DIAGNOSTIC"
    )

    print("\nQuestion:")
    print(QUERY)

    pipeline = QAPipeline(
        api_key=api_key,
    )

    try:

        # ==================================================
        # 1. RETRIEVAL
        # ==================================================

        separator("1. REAL HYBRID RETRIEVAL")

        started = time.perf_counter()

        retrieval_result = (
            pipeline.retriever.retrieve(
                query=QUERY,
                vector_top_k=10,
                entity_limit=10,
                relation_limit=30,
                semantic_min_score=0.25,
                hybrid_top_k=12,
            )
        )

        print(
            "Retrieval seconds:",
            round(
                time.perf_counter()
                - started,
                4,
            ),
        )

        print(
            "Retrieval result type:",
            type(
                retrieval_result
            ).__name__,
        )

        # ==================================================
        # 2. GET RETRIEVAL CANDIDATES
        # ==================================================

        if isinstance(
            retrieval_result,
            dict,
        ):

            candidates = (
                retrieval_result.get(
                    "results"
                )
                or retrieval_result.get(
                    "candidates"
                )
                or retrieval_result.get(
                    "chunks"
                )
                or []
            )

        elif isinstance(
            retrieval_result,
            list,
        ):

            candidates = (
                retrieval_result
            )

        else:

            candidates = []

        print(
            "Retrieved candidates:",
            len(candidates),
        )

        if not candidates:

            print(
                "\nNo candidates were found."
            )

            print(
                "Retrieval result representation:"
            )

            print(
                repr(retrieval_result)[:5000]
            )

            return

        # ==================================================
        # 3. RERANK
        # ==================================================

        separator("2. REAL RERANKING")

        started = time.perf_counter()

        rerank_result = (
            pipeline.reranker.rerank(
                query=QUERY,
                candidates=candidates,
                top_k=8,
            )
        )

        print(
            "Reranking seconds:",
            round(
                time.perf_counter()
                - started,
                4,
            ),
        )

        if isinstance(
            rerank_result,
            dict,
        ):

            reranked_candidates = (
                rerank_result.get(
                    "results"
                )
                or rerank_result.get(
                    "candidates"
                )
                or rerank_result.get(
                    "reranked"
                )
                or []
            )

        else:

            reranked_candidates = (
                rerank_result
            )

        print(
            "Reranked candidates:",
            len(
                reranked_candidates
            ),
        )

        # ==================================================
        # 4. REAL CONTEXT BUILDER
        # ==================================================

        separator("3. REAL CONTEXT BUILDING")

        context_result = (
            pipeline.context_builder.build(
                candidates=(
                    reranked_candidates
                ),
            )
        )

        evidence = (
            context_result.get(
                "evidence",
                [],
            )
        )

        print(
            "Evidence chunks:",
            len(evidence),
        )

        total_chars = sum(
            len(
                (
                    item.get("text")
                    or ""
                )
            )
            for item in evidence
        )

        print(
            "Total evidence characters:",
            total_chars,
        )

        print(
            "Configured max chunks:",
            pipeline.context_builder.max_chunks,
        )

        print(
            "Configured max chars:",
            pipeline.context_builder.max_chars,
        )

        # ==================================================
        # 5. INSPECT REAL EVIDENCE
        # ==================================================

        separator("4. EVIDENCE INSPECTION")

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            metadata = (
                item.get(
                    "metadata",
                    {},
                )
                or {}
            )

            text = (
                item.get("text")
                or ""
            )

            print(
                f"\nEVIDENCE #{index}"
            )

            print(
                "Chunk ID:",
                item.get(
                    "chunk_id"
                ),
            )

            print(
                "Evidence ID:",
                item.get(
                    "evidence_id"
                ),
            )

            print(
                "Source:",
                metadata.get(
                    "source"
                ),
            )

            print(
                "Page:",
                metadata.get(
                    "page"
                ),
            )

            print(
                "Characters:",
                len(text),
            )

            print(
                "Preview:"
            )

            print(
                text[:700]
            )

            if len(text) > 700:
                print("...")

        # ==================================================
        # 6. BUILD EXACT PRODUCTION PROMPT
        # ==================================================

        separator("5. PRODUCTION PROMPT DIAGNOSTICS")

        prompt = (
            pipeline.generator._build_prompt(
                query=QUERY,
                selected_chunks=evidence,
            )
        )

        print(
            "Prompt characters:",
            len(prompt),
        )

        print(
            "Approximate tokens:",
            round(
                len(prompt) / 4
            ),
        )

        print(
            "Model:",
            pipeline.generator.model_name,
        )

        # We deliberately do NOT print the complete
        # production prompt because it may contain
        # enterprise document content.

        # ==================================================
        # 7. REAL GENERATOR ONLY
        # ==================================================

        separator("6. REAL GENERATOR CALL")

        print(
            "Calling the same AnswerGenerator used "
            "by QAPipeline..."
        )

        started = time.perf_counter()

        try:

            generated = (
                pipeline.generator.generate(
                    query=QUERY,
                    selected_chunks=evidence,
                )
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            print(
                "\nGENERATION SUCCESS"
            )

            print(
                "Generation seconds:",
                round(
                    elapsed,
                    4,
                ),
            )

            print(
                "Grounded flag:",
                generated.grounded,
            )

            print(
                "Sources:",
                len(
                    generated.sources
                ),
            )

            print("\nANSWER")
            print("-" * 80)

            print(
                generated.answer
            )

        except Exception as error:

            elapsed = (
                time.perf_counter()
                - started
            )

            print(
                "\nGENERATION FAILED"
            )

            print(
                "Generation seconds:",
                round(
                    elapsed,
                    4,
                ),
            )

            print(
                "Wrapper type:",
                type(error).__name__,
            )

            print(
                "Wrapper message:",
                str(error),
            )

            cause = error.__cause__

            if cause is not None:

                print(
                    "\nORIGINAL PROVIDER ERROR"
                )

                print("-" * 80)

                print(
                    "Type:",
                    type(cause).__name__,
                )

                print(
                    "Module:",
                    type(cause).__module__,
                )

                print(
                    "Message:",
                    str(cause),
                )

                for attribute in (
                    "status_code",
                    "code",
                    "status",
                    "message",
                ):

                    value = getattr(
                        cause,
                        attribute,
                        None,
                    )

                    if value is not None:

                        print(
                            f"{attribute}:",
                            value,
                        )

            else:

                print(
                    "\nNo chained provider "
                    "exception found."
                )

            print("\nTRACEBACK")
            print("-" * 80)

            traceback.print_exception(
                type(error),
                error,
                error.__traceback__,
            )

    finally:

        pipeline.close()


if __name__ == "__main__":
    main()
"""
Test production-safe answer-generation failure handling.

Purpose
-------
Verify that if the answer generator raises an exception,
QAPipeline:

1. Does not crash.
2. Returns accepted=False.
3. Returns grounded=False.
4. Does NOT incorrectly classify an API failure as
   insufficient document evidence.
5. Skips citation validation and NLI grounding.
6. Records a useful failure reason.
7. Preserves performance information.

The real Gemini API is NOT called.
"""

import os

from dotenv import load_dotenv

from src.generation.qa_pipeline import QAPipeline


class ForcedGenerationFailure:
    """
    Test-only generator that simulates a provider failure
    after generator-level retry handling has been exhausted.
    """

    def generate(
        self,
        query,
        selected_chunks,
    ):
        raise RuntimeError(
            "Simulated Gemini 503 UNAVAILABLE"
        )


def main():

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )

    print("=" * 80)
    print("IOCL GENERATION FAILURE HANDLING TEST")
    print("=" * 80)

    pipeline = QAPipeline(
        api_key=api_key,
    )

    original_generator = (
        pipeline.generator
    )

    try:

        pipeline.generator = (
            ForcedGenerationFailure()
        )

        query = (
            "How are whistleblowers protected?"
        )

        print("\nQUESTION")
        print("-" * 80)
        print(query)

        result = pipeline.answer(
            query=query
        )

        print("\nFINAL ANSWER")
        print("-" * 80)
        print(result.answer)

        print("\nDECISION")
        print("-" * 80)
        print(
            f"Accepted:     {result.accepted}"
        )
        print(
            f"Grounded:     {result.grounded}"
        )
        print(
            "Insufficient: "
            f"{result.insufficient_evidence}"
        )

        print("\nVALIDATION")
        print("-" * 80)

        citation = (
            result.citation_validation
            or {}
        )

        grounding = (
            result.grounding_validation
            or {}
        )

        print(
            "Citation skipped: "
            f"{citation.get('skipped')}"
        )

        print(
            "NLI skipped:      "
            f"{grounding.get('skipped')}"
        )

        print("\nFAILURE REASONS")
        print("-" * 80)

        for reason in (
            result.failure_reasons
        ):
            print(
                f"- {reason}"
            )

        # =================================================
        # ASSERTIONS
        # =================================================

        assert (
            result.accepted is False
        ), (
            "Generation failure must not be accepted."
        )

        assert (
            result.grounded is False
        ), (
            "Generation failure must not be grounded."
        )

        assert (
            result.insufficient_evidence
            is False
        ), (
            "Provider failure must not be classified as "
            "insufficient document evidence."
        )

        assert (
            citation.get(
                "skipped"
            )
            is True
        ), (
            "Citation validation should be skipped when "
            "generation fails."
        )

        assert (
            grounding.get(
                "skipped"
            )
            is True
        ), (
            "NLI grounding should be skipped when "
            "generation fails."
        )

        assert (
            len(
                result.failure_reasons
            )
            >= 1
        ), (
            "Generation failure should produce a failure "
            "reason."
        )

        assert (
            result.performance.get(
                "generation_seconds"
            )
            is not None
        ), (
            "Generation timing should be preserved."
        )

        assert (
            result.performance.get(
                "total_seconds"
            )
            is not None
        ), (
            "Total timing should be preserved."
        )

        print("\nRESULT")
        print("-" * 80)
        print("PASS")

    finally:

        pipeline.generator = (
            original_generator
        )

        pipeline.close()

    print("\n" + "=" * 80)
    print(
        "GENERATION FAILURE HANDLING TEST COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
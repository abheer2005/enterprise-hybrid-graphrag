"""
Failure-path tests for the IOCL GraphRAG QA pipeline.

Purpose
-------
The normal end-to-end QA test checks whether the complete system can
answer realistic questions.

This file checks something different:

    Does QAPipeline behave safely when one of its downstream components
    produces a bad, incomplete, unsupported, or insufficient answer?

We deliberately replace ONLY the answer generator so that retrieval,
reranking, relevance gating, context construction, citation validation,
NLI grounding, repair, and the final acceptance gate continue to use
the real production implementation.

This allows us to test:

1. Completely unrelated query -> relevance gate blocks generation.
2. Explicit insufficient-evidence response.
3. Missing citation -> deterministic rejection.
4. Invalid citation ID -> deterministic rejection.
5. Fabricated claim with valid citation -> NLI rejection.
6. Mixed supported + fabricated answer -> unsupported fragment repair.
7. Fully supported answer -> normal acceptance.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORT PIPELINE
# ============================================================

from src.generation.qa_pipeline import QAPipeline


# ============================================================
# TEST GENERATOR RESULT
# ============================================================

@dataclass
class FakeGeneratedAnswer:
    """
    Minimal object compatible with:

        generated.answer

    inside QAPipeline.
    """

    answer: str


class FixedAnswerGenerator:
    """
    Test-only generator.

    QAPipeline normally calls:

        self.generator.generate(
            query=query,
            selected_chunks=evidence,
        )

    This replacement returns exactly the answer supplied
    by the test.

    Everything after generation still runs normally.
    """

    def __init__(
        self,
        answer: str,
    ):
        self.answer = answer

        self.called = False

        self.last_query = None

        self.last_selected_chunks = None

    def generate(
        self,
        query: str,
        selected_chunks,
    ) -> FakeGeneratedAnswer:

        self.called = True

        self.last_query = query

        self.last_selected_chunks = selected_chunks

        return FakeGeneratedAnswer(
            answer=self.answer,
        )


# ============================================================
# ASSERTION HELPERS
# ============================================================

def require(
    condition: bool,
    message: str,
) -> None:
    """
    Raise a readable test failure.
    """

    if not condition:
        raise AssertionError(
            message
        )


def print_header(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_result(
    result,
) -> None:
    """
    Print the most important QAResult fields.
    """

    print()
    print("FINAL ANSWER")
    print("-" * 80)
    print(result.answer)

    print()
    print("DECISION")
    print("-" * 80)

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

    print()
    print("RELEVANCE")
    print("-" * 80)

    print(
        "Status:           "
        f"{result.relevance.get('status')}"
    )

    print(
        "Allow generation: "
        f"{result.relevance.get('allow_generation')}"
    )

    print(
        "Reason:           "
        f"{result.relevance.get('reason')}"
    )

    print()
    print("CITATION VALIDATION")
    print("-" * 80)

    citation = (
        result.citation_validation
        or {}
    )

    print(
        "Passed:            "
        f"{citation.get('validation_passed')}"
    )

    print(
        "Citation valid:    "
        f"{citation.get('citation_valid')}"
    )

    print(
        "Insufficient:      "
        f"{citation.get('insufficient_evidence')}"
    )

    print(
        "Cited evidence:    "
        f"{citation.get('cited_evidence_ids')}"
    )

    print(
        "Invalid citations: "
        f"{citation.get('invalid_citation_ids')}"
    )

    errors = (
        citation.get(
            "errors",
            [],
        )
        or []
    )

    if errors:

        print(
            "Validation errors:"
        )

        for error in errors:

            print(
                f"  - {error}"
            )

    print()
    print("GROUNDING")
    print("-" * 80)

    grounding = (
        result.grounding_validation
        or {}
    )

    print(
        "Passed:             "
        f"{grounding.get('passed')}"
    )

    print(
        "Skipped:            "
        f"{grounding.get('skipped', False)}"
    )

    print(
        "Claims checked:     "
        f"{grounding.get('claims_checked', 0)}"
    )

    print(
        "Supported claims:   "
        f"{grounding.get('supported_claims', 0)}"
    )

    print(
        "Unsupported claims: "
        f"{grounding.get('unsupported_claims', 0)}"
    )

    print()
    print("REPAIR")
    print("-" * 80)

    repair = (
        result.repair
        or {}
    )

    print(
        "Attempted:          "
        f"{repair.get('attempted')}"
    )

    print(
        "Changed:            "
        f"{repair.get('changed')}"
    )

    print(
        "Usable:             "
        f"{repair.get('usable')}"
    )

    print(
        "Accepted repair:    "
        f"{repair.get('accepted', False)}"
    )

    print(
        "Supported claims:   "
        f"{repair.get('supported_claims', 0)}"
    )

    print(
        "Unsupported claims: "
        f"{repair.get('unsupported_claims', 0)}"
    )

    removed = (
        repair.get(
            "removed_fragments",
            [],
        )
        or []
    )

    print(
        "Removed fragments:  "
        f"{len(removed)}"
    )

    for fragment in removed:

        print(
            f"  - {fragment}"
        )

    print()
    print("SOURCES")
    print("-" * 80)

    if result.sources:

        for source in result.sources:

            print(
                f"- {source}"
            )

    else:

        print(
            "None"
        )

    print()
    print("FAILURE REASONS")
    print("-" * 80)

    if result.failure_reasons:

        for reason in result.failure_reasons:

            print(
                f"- {reason}"
            )

    else:

        print(
            "None"
        )


# ============================================================
# TEST RUNNER
# ============================================================

def run_test(
    number: int,
    title: str,
    pipeline: QAPipeline,
    query: str,
    fake_answer: str | None,
    assertions,
    expect_generator_call: bool = True,
):
    """
    Run one failure-path test.

    If fake_answer is None, the production generator is not
    replaced. This is useful for relevance-gate tests because
    generation should never be reached.
    """

    print_header(
        f"TEST {number}: {title}"
    )

    original_generator = (
        pipeline.generator
    )

    fake_generator = None

    try:

        if fake_answer is not None:

            fake_generator = (
                FixedAnswerGenerator(
                    answer=fake_answer,
                )
            )

            pipeline.generator = (
                fake_generator
            )

        result = pipeline.answer(
            query=query,
        )

        print_result(
            result
        )

        if fake_generator is not None:

            require(
                fake_generator.called
                == expect_generator_call,
                (
                    "Generator call expectation failed. "
                    f"Expected called="
                    f"{expect_generator_call}, "
                    f"actual="
                    f"{fake_generator.called}."
                ),
            )

        assertions(
            result,
        )

        print()
        print(
            "RESULT: PASS"
        )

        return True

    except Exception as exc:

        print()
        print(
            "RESULT: FAIL"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return False

    finally:

        pipeline.generator = (
            original_generator
        )


# ============================================================
# TEST ASSERTIONS
# ============================================================

def assert_irrelevant_query(
    result,
):

    require(
        result.accepted is False,
        "Irrelevant query must not be accepted.",
    )

    require(
        result.grounded is False,
        "Irrelevant query must not be marked grounded.",
    )

    require(
        result.insufficient_evidence is True,
        (
            "Relevance-gate rejection should be marked "
            "as insufficient evidence."
        ),
    )

    require(
        result.relevance.get(
            "allow_generation"
        )
        is False,
        (
            "Relevance gate should block generation."
        ),
    )

    require(
        result.relevance.get(
            "status"
        )
        == "irrelevant",
        (
            "Expected relevance status 'irrelevant'."
        ),
    )

    require(
        len(result.sources)
        == 0,
        (
            "Rejected unrelated query must expose "
            "no sources."
        ),
    )


def assert_insufficient_response(
    result,
):

    require(
        result.insufficient_evidence
        is True,
        (
            "Generator's explicit insufficient-evidence "
            "response should be detected."
        ),
    )

    require(
        result.accepted
        is True,
        (
            "A valid explicit insufficient-evidence "
            "response should pass deterministic validation."
        ),
    )

    require(
        result.grounded
        is False,
        (
            "Insufficient-evidence response should not "
            "be marked as a grounded factual answer."
        ),
    )

    require(
        result.grounding_validation.get(
            "skipped"
        )
        is True,
        (
            "NLI should be skipped for explicit "
            "insufficient-evidence responses."
        ),
    )


def assert_missing_citation(
    result,
):

    require(
        result.accepted
        is False,
        (
            "Factual answer without citations must "
            "not be accepted."
        ),
    )

    require(
        result.grounded
        is False,
        (
            "Rejected answer must not be marked grounded."
        ),
    )

    require(
        result.citation_validation.get(
            "validation_passed"
        )
        is False,
        (
            "Missing citation should fail citation "
            "validation."
        ),
    )

    require(
        len(result.sources)
        == 0,
        (
            "Rejected answer must expose no final sources."
        ),
    )


def assert_invalid_citation(
    result,
):

    require(
        result.accepted
        is False,
        (
            "Answer with invalid citation must not "
            "be accepted."
        ),
    )

    require(
        result.citation_validation.get(
            "validation_passed"
        )
        is False,
        (
            "Invalid citation ID should fail "
            "citation validation."
        ),
    )

    invalid_ids = (
        result.citation_validation.get(
            "invalid_citation_ids",
            [],
        )
        or []
    )

    require(
        999 in invalid_ids,
        (
            "Citation [999] should be reported as invalid."
        ),
    )

    require(
        len(result.sources)
        == 0,
        (
            "Rejected answer must expose no final sources."
        ),
    )


def assert_fabricated_claim(
    result,
):

    require(
        result.accepted
        is False,
        (
            "Fabricated factual claim must not "
            "be accepted."
        ),
    )

    require(
        result.grounded
        is False,
        (
            "Fabricated claim must not be marked grounded."
        ),
    )

    require(
        result.citation_validation.get(
            "validation_passed"
        )
        is True,
        (
            "The citation itself should be structurally "
            "valid so NLI gets tested."
        ),
    )

    require(
        result.grounding_validation.get(
            "passed"
        )
        is False,
        (
            "NLI should reject the fabricated claim."
        ),
    )

    require(
        result.grounding_validation.get(
            "unsupported_claims",
            0,
        )
        >= 1,
        (
            "At least one unsupported claim should "
            "be detected."
        ),
    )

    require(
        result.repair.get(
            "attempted"
        )
        is True,
        (
            "Repair should be attempted after grounding "
            "failure when citation validation passed."
        ),
    )

    require(
        len(result.sources)
        == 0,
        (
            "Rejected fabricated answer must expose "
            "no final sources."
        ),
    )


def assert_mixed_repair(
    result,
):

    require(
        result.repair.get(
            "attempted"
        )
        is True,
        (
            "Mixed answer should trigger repair."
        ),
    )

    require(
        result.repair.get(
            "changed"
        )
        is True,
        (
            "Repair should remove the unsupported "
            "fragment."
        ),
    )

    require(
        result.repair.get(
            "accepted"
        )
        is True,
        (
            "Repair should be accepted when supported "
            "content remains."
        ),
    )

    require(
        result.accepted
        is True,
        (
            "Successfully repaired answer should "
            "be accepted."
        ),
    )

    require(
        result.grounded
        is True,
        (
            "Successfully repaired answer should "
            "be marked grounded."
        ),
    )

    require(
        "monetary reward"
        not in result.answer.lower(),
        (
            "Unsupported monetary-reward claim should "
            "not survive repair."
        ),
    )

    require(
        "victimization"
        in result.answer.lower(),
        (
            "Supported victimization claim should "
            "remain after repair."
        ),
    )

    require(
        len(result.sources)
        >= 1,
        (
            "Accepted repaired answer should retain "
            "its cited source."
        ),
    )


def assert_supported_answer(
    result,
):

    require(
        result.accepted
        is True,
        (
            "Fully supported answer should be accepted."
        ),
    )

    require(
        result.grounded
        is True,
        (
            "Fully supported answer should be grounded."
        ),
    )

    require(
        result.insufficient_evidence
        is False,
        (
            "Supported factual answer must not be marked "
            "insufficient."
        ),
    )

    require(
        result.citation_validation.get(
            "validation_passed"
        )
        is True,
        (
            "Supported answer should pass citation "
            "validation."
        ),
    )

    require(
        result.grounding_validation.get(
            "passed"
        )
        is True,
        (
            "Supported answer should pass NLI grounding."
        ),
    )

    require(
        result.repair.get(
            "attempted"
        )
        is False,
        (
            "Repair should not run for a fully grounded "
            "answer."
        ),
    )

    require(
        len(result.sources)
        >= 1,
        (
            "Accepted answer should expose its cited "
            "source."
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "IOCL QA PIPELINE FAILURE-PATH TEST"
    )
    print("=" * 80)

    api_key = (
        os.getenv(
            "GOOGLE_API_KEY"
        )
        or os.getenv(
            "GEMINI_API_KEY"
        )
        or "TEST_ONLY_KEY"
    )

    pipeline = None

    passed = 0
    failed = 0

    try:

        pipeline = QAPipeline(
            api_key=api_key,
        )

        # ----------------------------------------------------
        # TEST 1
        #
        # Completely unrelated query.
        #
        # The relevance gate should stop the request BEFORE
        # answer generation.
        # ----------------------------------------------------

        ok = run_test(
            number=1,
            title=(
                "IRRELEVANT QUERY BLOCKED BEFORE GENERATION"
            ),
            pipeline=pipeline,
            query=(
                "Who is M. S. Dhoni?"
            ),
            fake_answer=None,
            assertions=(
                assert_irrelevant_query
            ),
        )

        passed += int(ok)
        failed += int(not ok)

        # ----------------------------------------------------
        # TEST 2
        #
        # Relevant topic, but generator explicitly says the
        # retrieved evidence is insufficient.
        #
        # This tests the pipeline's special insufficient-
        # evidence branch.
        # ----------------------------------------------------

        ok = run_test(
            number=2,
            title=(
                "EXPLICIT INSUFFICIENT-EVIDENCE RESPONSE"
            ),
            pipeline=pipeline,
            query=(
                "Does the whistleblower policy provide "
                "financial rewards?"
            ),
            fake_answer=(
                "I could not find sufficient information "
                "in the provided documents to answer this "
                "question reliably."
            ),
            assertions=(
                assert_insufficient_response
            ),
        )

        passed += int(ok)
        failed += int(not ok)

        # ----------------------------------------------------
        # TEST 3
        #
        # Correct-looking factual statement but NO citation.
        #
        # Citation validator must reject it.
        # ----------------------------------------------------

        ok = run_test(
            number=3,
            title=(
                "FACTUAL ANSWER WITHOUT CITATION"
            ),
            pipeline=pipeline,
            query=(
                "How are whistleblowers protected?"
            ),
            fake_answer=(
                "Whistle-blowers are entitled to "
                "protection against victimization."
            ),
            assertions=(
                assert_missing_citation
            ),
        )

        passed += int(ok)
        failed += int(not ok)

        # ----------------------------------------------------
        # TEST 4
        #
        # Citation syntax exists but evidence ID does not.
        # ----------------------------------------------------

        ok = run_test(
            number=4,
            title=(
                "INVALID CITATION ID"
            ),
            pipeline=pipeline,
            query=(
                "How are whistleblowers protected?"
            ),
            fake_answer=(
                "Whistle-blowers are entitled to "
                "protection against victimization [999]."
            ),
            assertions=(
                assert_invalid_citation
            ),
        )

        passed += int(ok)
        failed += int(not ok)

        # ----------------------------------------------------
        # TEST 5
        #
        # Valid citation, fabricated claim.
        #
        # This is critical:
        #
        # citation validation should succeed structurally,
        # but NLI should determine that the evidence does not
        # support the monetary reward claim.
        # ----------------------------------------------------

        ok = run_test(
            number=5,
            title=(
                "FABRICATED CLAIM WITH VALID CITATION"
            ),
            pipeline=pipeline,
            query=(
                "How are whistleblowers protected?"
            ),
            fake_answer=(
                "Whistle-blowers receive a monetary reward "
                "for reporting misconduct [1]."
            ),
            assertions=(
                assert_fabricated_claim
            ),
        )

        passed += int(ok)
        failed += int(not ok)

        # ----------------------------------------------------
        # TEST 6
        #
        # One supported claim + one fabricated claim.
        #
        # Expected:
        #
        # NLI fails complete grounding.
        # Repair runs.
        # Fabricated fragment is removed.
        # Supported fragment remains.
        # Repaired answer is accepted.
        # ----------------------------------------------------

        ok = run_test(
            number=6,
            title=(
                "PARTIAL GROUNDING AND SAFE REPAIR"
            ),
            pipeline=pipeline,
            query=(
                "How are whistleblowers protected?"
            ),
            fake_answer=(
                "Whistle-blowers are entitled to "
                "protection against victimization [1].\n\n"
                "Whistle-blowers receive a monetary reward "
                "for reporting misconduct [1]."
            ),
            assertions=(
                assert_mixed_repair
            ),
        )

        passed += int(ok)
        failed += int(not ok)

        # ----------------------------------------------------
        # TEST 7
        #
        # Fully supported factual answer.
        #
        # This acts as the control test to prove that the
        # safety gates do not reject a valid answer.
        # ----------------------------------------------------

        ok = run_test(
            number=7,
            title=(
                "FULLY SUPPORTED CONTROL ANSWER"
            ),
            pipeline=pipeline,
            query=(
                "How are whistleblowers protected?"
            ),
            fake_answer=(
                "Whistle-blowers are entitled to "
                "protection against victimization [1]."
            ),
            assertions=(
                assert_supported_answer
            ),
        )

        passed += int(ok)
        failed += int(not ok)

    finally:

        if pipeline is not None:

            pipeline.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FAILURE-PATH TEST SUMMARY"
    )
    print("=" * 80)

    total = (
        passed
        + failed
    )

    print(
        f"Total:  {total}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print("=" * 80)

    if failed:

        raise SystemExit(
            1
        )

    print(
        "ALL QA FAILURE-PATH TESTS PASSED"
    )


if __name__ == "__main__":
    main()
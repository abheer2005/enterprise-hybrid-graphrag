import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from dotenv import load_dotenv

from src.generation.qa_pipeline import QAPipeline


# ============================================================
# CONFIGURATION
# ============================================================

SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80

# Keep this False initially.
# This avoids running every expensive test while verifying
# that the evaluation framework itself works.
RUN_FULL_SUITE = False

# Number of questions used for the quick smoke evaluation.
QUICK_TEST_COUNT = 5


# ============================================================
# TEST CASE MODEL
# ============================================================

@dataclass
class EvaluationCase:
    number: int
    category: str
    question: str

    # Expected high-level behaviour.
    expect_accepted: Optional[bool] = None
    expect_insufficient: Optional[bool] = None
    expect_relevance: Optional[str] = None

    # Optional semantic checks on the final answer.
    required_terms: tuple[str, ...] = ()

    # Optional custom assertion.
    custom_check: Optional[
        Callable[[object], tuple[bool, str]]
    ] = None


# ============================================================
# TEST SUITE
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # DIRECT DOCUMENT QUESTIONS
    # --------------------------------------------------------

    EvaluationCase(
        number=1,
        category="direct",
        question="How are whistleblowers protected?",
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
        required_terms=(
            "protection",
        ),
    ),

    EvaluationCase(
        number=2,
        category="direct",
        question="What are the company's CSR responsibilities?",
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
        required_terms=(
            "CSR",
        ),
    ),

    EvaluationCase(
        number=3,
        category="direct",
        question="What rules govern related party transactions?",
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
        required_terms=(
            "related party",
        ),
    ),

    EvaluationCase(
        number=4,
        category="direct",
        question="What duties apply to company officers?",
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
        required_terms=(
            "officer",
        ),
    ),

    EvaluationCase(
        number=5,
        category="direct",
        question=(
            "What responsibilities do Independent Directors have?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
        required_terms=(
            "independent",
            "director",
        ),
    ),

    # --------------------------------------------------------
    # WHISTLEBLOWER POLICY
    # --------------------------------------------------------

    EvaluationCase(
        number=6,
        category="whistleblower",
        question=(
            "What can a whistleblower do if the protection "
            "provided to them is disregarded?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=7,
        category="whistleblower",
        question=(
            "Can the identity of a whistleblower be disclosed?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=8,
        category="whistleblower",
        question=(
            "What grievance redressal process is available "
            "to whistleblowers?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    # --------------------------------------------------------
    # CSR POLICY
    # --------------------------------------------------------

    EvaluationCase(
        number=9,
        category="csr",
        question=(
            "How much of average net profits must be "
            "earmarked for CSR?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=10,
        category="csr",
        question=(
            "What happens when the CSR budget is not spent?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=11,
        category="csr",
        question=(
            "Can surplus generated from CSR projects become "
            "part of business profit?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=12,
        category="csr",
        question=(
            "What are IndianOil's major CSR thrust areas?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=13,
        category="csr",
        question=(
            "What CSR information must be disclosed "
            "on the company's website?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    # --------------------------------------------------------
    # RELATED PARTY TRANSACTIONS
    # --------------------------------------------------------

    EvaluationCase(
        number=14,
        category="rpt",
        question=(
            "Which authority approves Related Party "
            "Transactions?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=15,
        category="rpt",
        question=(
            "When is Board approval required for a "
            "Related Party Transaction?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=16,
        category="rpt",
        question=(
            "When is shareholder approval required for "
            "Related Party Transactions?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=17,
        category="rpt",
        question=(
            "What is considered a Material Modification "
            "to a Related Party Transaction?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=18,
        category="rpt",
        question=(
            "How often must the Related Party Transaction "
            "policy be reviewed?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    # --------------------------------------------------------
    # CODE OF CONDUCT
    # --------------------------------------------------------

    EvaluationCase(
        number=19,
        category="conduct",
        question=(
            "What conflict-of-interest obligations apply "
            "to officers?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=20,
        category="conduct",
        question=(
            "What restrictions apply to officers serving "
            "in competing businesses?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=21,
        category="conduct",
        question=(
            "What duties apply to officers regarding "
            "company assets?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=22,
        category="conduct",
        question=(
            "What health, safety, and environmental duties "
            "apply to officers?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    # --------------------------------------------------------
    # PARAPHRASE / ROBUSTNESS
    # --------------------------------------------------------

    EvaluationCase(
        number=23,
        category="paraphrase",
        question=(
            "What safeguards exist for someone who reports "
            "misconduct under the whistleblower policy?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=24,
        category="paraphrase",
        question=(
            "How does IndianOil handle money that remains "
            "unspent from its CSR allocation?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    EvaluationCase(
        number=25,
        category="paraphrase",
        question=(
            "Which approvals are needed before related-party "
            "dealings can proceed?"
        ),
        expect_accepted=True,
        expect_insufficient=False,
        expect_relevance="relevant",
    ),

    # --------------------------------------------------------
    # OUT-OF-DOMAIN / RELEVANCE GATE
    # --------------------------------------------------------

    EvaluationCase(
        number=26,
        category="out_of_domain",
        question="Who is M. S. Dhoni?",
        expect_accepted=False,
        expect_insufficient=True,
        expect_relevance="irrelevant",
    ),

    EvaluationCase(
        number=27,
        category="out_of_domain",
        question="What is the capital of France?",
        expect_accepted=False,
        expect_insufficient=True,
        expect_relevance="irrelevant",
    ),

    EvaluationCase(
        number=28,
        category="out_of_domain",
        question="How do I cook pasta?",
        expect_accepted=False,
        expect_insufficient=True,
        expect_relevance="irrelevant",
    ),

    EvaluationCase(
        number=29,
        category="out_of_domain",
        question="Explain Newton's laws of motion.",
        expect_accepted=False,
        expect_insufficient=True,
        expect_relevance="irrelevant",
    ),

    EvaluationCase(
        number=30,
        category="out_of_domain",
        question="Who won the FIFA World Cup in 2022?",
        expect_accepted=False,
        expect_insufficient=True,
        expect_relevance="irrelevant",
    ),

    # --------------------------------------------------------
    # UNSUPPORTED / FABRICATION TRAPS
    # --------------------------------------------------------

    EvaluationCase(
        number=31,
        category="fabrication_trap",
        question=(
            "How much monetary reward does a whistleblower "
            "receive for reporting misconduct?"
        ),
    ),

    EvaluationCase(
        number=32,
        category="fabrication_trap",
        question=(
            "What promotion is automatically given to "
            "whistleblowers after reporting misconduct?"
        ),
    ),

    EvaluationCase(
        number=33,
        category="fabrication_trap",
        question=(
            "What luxury benefits are provided to officers "
            "under the Code of Conduct?"
        ),
    ),

    EvaluationCase(
        number=34,
        category="fabrication_trap",
        question=(
            "What guaranteed financial return does the "
            "company receive from CSR expenditure?"
        ),
    ),

    EvaluationCase(
        number=35,
        category="fabrication_trap",
        question=(
            "Which policy allows officers to use company "
            "assets for personal profit?"
        ),
    ),

    # --------------------------------------------------------
    # CROSS-DOCUMENT / COMPLEX QUESTIONS
    # --------------------------------------------------------

    EvaluationCase(
        number=36,
        category="complex",
        question=(
            "How do the company's governance policies "
            "promote accountability and ethical conduct?"
        ),
    ),

    EvaluationCase(
        number=37,
        category="complex",
        question=(
            "What mechanisms across the available policies "
            "support transparency and oversight?"
        ),
    ),

    EvaluationCase(
        number=38,
        category="complex",
        question=(
            "What responsibilities of the Board appear "
            "across the available governance documents?"
        ),
    ),

    EvaluationCase(
        number=39,
        category="complex",
        question=(
            "How are conflicts of interest controlled across "
            "the company's governance policies?"
        ),
    ),

    EvaluationCase(
        number=40,
        category="complex",
        question=(
            "Summarize the major governance safeguards "
            "described in the available IOCL documents."
        ),
    ),
]


# ============================================================
# SAFE ATTRIBUTE HELPERS
# ============================================================

def as_dict(value):

    if isinstance(value, dict):
        return value

    return {}


def get_grounding(result):

    return as_dict(
        getattr(
            result,
            "grounding_validation",
            {},
        )
    )


def get_citation(result):

    return as_dict(
        getattr(
            result,
            "citation_validation",
            {},
        )
    )


def get_repair(result):

    return as_dict(
        getattr(
            result,
            "repair",
            {},
        )
    )


def get_relevance(result):

    return as_dict(
        getattr(
            result,
            "relevance",
            {},
        )
    )


def get_performance(result):

    return as_dict(
        getattr(
            result,
            "performance",
            {},
        )
    )


# ============================================================
# EVALUATION LOGIC
# ============================================================

def evaluate_case(
    case,
    result,
):

    failures = []
    warnings = []

    answer = (
        getattr(
            result,
            "answer",
            "",
        )
        or ""
    ).strip()

    accepted = bool(
        getattr(
            result,
            "accepted",
            False,
        )
    )

    grounded = bool(
        getattr(
            result,
            "grounded",
            False,
        )
    )

    insufficient = bool(
        getattr(
            result,
            "insufficient_evidence",
            False,
        )
    )

    citation = get_citation(
        result
    )

    grounding = get_grounding(
        result
    )

    repair = get_repair(
        result
    )

    relevance = get_relevance(
        result
    )

    # --------------------------------------------------------
    # BASIC INVARIANTS
    # --------------------------------------------------------

    if not answer:

        failures.append(
            "Final answer is empty."
        )

    if accepted and not insufficient:

        if not grounded:

            failures.append(
                "Accepted factual answer is not marked grounded."
            )

        if not citation.get(
            "validation_passed",
            False,
        ):

            failures.append(
                "Accepted factual answer failed citation validation."
            )

        cited_ids = citation.get(
            "cited_evidence_ids",
            [],
        ) or []

        if not cited_ids:

            failures.append(
                "Accepted factual answer contains no resolved evidence citations."
            )

    # --------------------------------------------------------
    # INVALID CITATIONS MUST NEVER SURVIVE
    # --------------------------------------------------------

    invalid_ids = citation.get(
        "invalid_citation_ids",
        [],
    ) or []

    if invalid_ids:

        failures.append(
            "Final answer contains invalid evidence citations: "
            + str(
                invalid_ids
            )
        )

    # --------------------------------------------------------
    # ACCEPTANCE EXPECTATION
    # --------------------------------------------------------

    if (
        case.expect_accepted
        is not None
        and accepted
        != case.expect_accepted
    ):

        failures.append(
            "Acceptance mismatch: expected "
            f"{case.expect_accepted}, got {accepted}."
        )

    # --------------------------------------------------------
    # INSUFFICIENT EXPECTATION
    # --------------------------------------------------------

    if (
        case.expect_insufficient
        is not None
        and insufficient
        != case.expect_insufficient
    ):

        failures.append(
            "Insufficient-evidence mismatch: expected "
            f"{case.expect_insufficient}, "
            f"got {insufficient}."
        )

    # --------------------------------------------------------
    # RELEVANCE EXPECTATION
    # --------------------------------------------------------

    actual_relevance = relevance.get(
        "status"
    )

    if (
        case.expect_relevance
        is not None
        and actual_relevance
        != case.expect_relevance
    ):

        failures.append(
            "Relevance mismatch: expected "
            f"{case.expect_relevance}, "
            f"got {actual_relevance}."
        )

    # --------------------------------------------------------
    # REQUIRED ANSWER TERMS
    # --------------------------------------------------------

    if (
        accepted
        and case.required_terms
    ):

        normalized = answer.lower()

        for term in case.required_terms:

            if term.lower() not in normalized:

                warnings.append(
                    "Expected semantic term not found "
                    f"in final answer: {term!r}"
                )

    # --------------------------------------------------------
    # GROUNDING / REPAIR CONSISTENCY
    # --------------------------------------------------------

    unsupported = grounding.get(
        "unsupported_claims",
        0,
    ) or 0

    grounding_passed = grounding.get(
        "passed",
        False,
    )

    repair_accepted = repair.get(
        "accepted",
        False,
    )

    if (
        accepted
        and not insufficient
        and not grounding_passed
        and not repair_accepted
    ):

        failures.append(
            "Answer was accepted even though grounding "
            "failed and no repair was accepted."
        )

    if unsupported > 0:

        if repair_accepted:

            warnings.append(
                f"Original generation contained "
                f"{unsupported} unsupported claim(s), "
                "but deterministic repair was accepted."
            )

        elif accepted:

            failures.append(
                "Accepted answer still has unsupported "
                "claims without accepted repair."
            )

    # --------------------------------------------------------
    # OUT-OF-DOMAIN SAFETY
    # --------------------------------------------------------

    if (
        case.category
        == "out_of_domain"
    ):

        if accepted:

            failures.append(
                "Out-of-domain question produced an "
                "accepted document answer."
            )

        if not insufficient:

            failures.append(
                "Out-of-domain question was not marked "
                "insufficient evidence."
            )

    # --------------------------------------------------------
    # FABRICATION TRAPS
    # --------------------------------------------------------

    if (
        case.category
        == "fabrication_trap"
    ):

        if (
            accepted
            and not insufficient
        ):

            # This is not automatically a failure because the
            # documents may explicitly state that something
            # does NOT exist.
            #
            # Grounding and citation validation therefore
            # remain the authority.

            if not grounded:

                failures.append(
                    "Fabrication-trap answer was accepted "
                    "without grounding."
                )

            if not citation.get(
                "validation_passed",
                False,
            ):

                failures.append(
                    "Fabrication-trap answer was accepted "
                    "without valid citations."
                )

    # --------------------------------------------------------
    # CUSTOM CHECK
    # --------------------------------------------------------

    if case.custom_check:

        try:

            ok, message = (
                case.custom_check(
                    result
                )
            )

            if not ok:

                failures.append(
                    message
                )

        except Exception as exc:

            failures.append(
                "Custom evaluation check raised: "
                f"{type(exc).__name__}: {exc}"
            )

    return failures, warnings


# ============================================================
# PRINTING
# ============================================================

def print_result_details(
    case,
    result,
    elapsed,
):

    citation = get_citation(
        result
    )

    grounding = get_grounding(
        result
    )

    repair = get_repair(
        result
    )

    relevance = get_relevance(
        result
    )

    performance = get_performance(
        result
    )

    print(SEPARATOR)

    print(
        f"TEST {case.number}: "
        f"{case.category.upper()}"
    )

    print(SEPARATOR)

    print("\nQUESTION")
    print(SUB_SEPARATOR)
    print(case.question)

    print("\nFINAL ANSWER")
    print(SUB_SEPARATOR)

    answer = (
        getattr(
            result,
            "answer",
            "",
        )
        or ""
    )

    print(
        answer
        if answer
        else "<EMPTY>"
    )

    print("\nPIPELINE DECISION")
    print(SUB_SEPARATOR)

    print(
        "Relevance:            "
        f"{relevance.get('status')}"
    )

    print(
        "Accepted:             "
        f"{getattr(result, 'accepted', None)}"
    )

    print(
        "Grounded:             "
        f"{getattr(result, 'grounded', None)}"
    )

    print(
        "Insufficient:         "
        f"{getattr(result, 'insufficient_evidence', None)}"
    )

    print(
        "Citation valid:       "
        f"{citation.get('citation_valid')}"
    )

    print(
        "Citation passed:      "
        f"{citation.get('validation_passed')}"
    )

    print(
        "Claims checked:       "
        f"{grounding.get('claims_checked', 0)}"
    )

    print(
        "Supported claims:     "
        f"{grounding.get('supported_claims', 0)}"
    )

    print(
        "Unsupported claims:   "
        f"{grounding.get('unsupported_claims', 0)}"
    )

    print(
        "Repair attempted:     "
        f"{repair.get('attempted', False)}"
    )

    print(
        "Repair changed:       "
        f"{repair.get('changed', False)}"
    )

    print(
        "Repair accepted:      "
        f"{repair.get('accepted', False)}"
    )

    print(
        "Cited source count:   "
        f"{len(getattr(result, 'sources', []) or [])}"
    )

    print(
        "Elapsed:              "
        f"{elapsed:.2f}s"
    )

    if performance:

        print("\nPERFORMANCE")
        print(SUB_SEPARATOR)

        for name, seconds in performance.items():

            print(
                f"{name:<40}"
                f"{seconds:>10.4f} sec"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY was not found "
            "in the .env file."
        )

    print(SEPARATOR)
    print("IOCL GRAPHRAG FINAL QA EVALUATION")
    print(SEPARATOR)

    if RUN_FULL_SUITE:

        cases = TEST_CASES

        print(
            "\nMode: FULL EVALUATION"
        )

    else:

        cases = TEST_CASES[
            :QUICK_TEST_COUNT
        ]

        print(
            "\nMode: QUICK EVALUATION"
        )

        print(
            f"Running first "
            f"{len(cases)} test cases."
        )

    print(
        f"Total available cases: "
        f"{len(TEST_CASES)}"
    )

    print(
        f"Cases running now: "
        f"{len(cases)}"
    )

    pipeline = QAPipeline(
        api_key=api_key,
    )

    passed = 0
    failed = 0
    errors = 0

    total_start = (
        time.perf_counter()
    )

    category_stats = {}

    try:

        for case in cases:

            category_stats.setdefault(
                case.category,
                {
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                },
            )

            start = (
                time.perf_counter()
            )

            try:

                result = pipeline.answer(
                    query=case.question,
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                print_result_details(
                    case=case,
                    result=result,
                    elapsed=elapsed,
                )

                failures, warnings = (
                    evaluate_case(
                        case=case,
                        result=result,
                    )
                )

                print("\nRESULT")
                print(SUB_SEPARATOR)

                if failures:

                    failed += 1

                    category_stats[
                        case.category
                    ]["failed"] += 1

                    print("FAIL")

                    for failure in failures:

                        print(
                            f"  - {failure}"
                        )

                else:

                    passed += 1

                    category_stats[
                        case.category
                    ]["passed"] += 1

                    print("PASS")

                if warnings:

                    print("\nWARNINGS")
                    print(SUB_SEPARATOR)

                    for warning in warnings:

                        print(
                            f"  - {warning}"
                        )

            except Exception as exc:

                elapsed = (
                    time.perf_counter()
                    - start
                )

                errors += 1

                category_stats[
                    case.category
                ]["errors"] += 1

                print(SEPARATOR)

                print(
                    f"TEST {case.number}: "
                    f"{case.category.upper()}"
                )

                print(SEPARATOR)

                print("\nQUESTION")
                print(SUB_SEPARATOR)
                print(case.question)

                print("\nRESULT")
                print(SUB_SEPARATOR)

                print("ERROR")

                print(
                    "  - Pipeline raised "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    f"  - Elapsed: "
                    f"{elapsed:.2f}s"
                )

    finally:

        pipeline.close()

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    total_run = (
        passed
        + failed
        + errors
    )

    if total_run:

        pass_rate = (
            passed
            / total_run
            * 100
        )

    else:

        pass_rate = 0.0

    print("\n")
    print(SEPARATOR)
    print("FINAL EVALUATION SUMMARY")
    print(SEPARATOR)

    print(
        f"Tests executed: {total_run}"
    )

    print(
        f"Passed:         {passed}"
    )

    print(
        f"Failed:         {failed}"
    )

    print(
        f"Errors:         {errors}"
    )

    print(
        f"Pass rate:      "
        f"{pass_rate:.2f}%"
    )

    print(
        f"Total time:     "
        f"{total_elapsed:.2f}s"
    )

    print("\nCATEGORY SUMMARY")
    print(SUB_SEPARATOR)

    for category, stats in (
        category_stats.items()
    ):

        category_total = (
            stats["passed"]
            + stats["failed"]
            + stats["errors"]
        )

        print(
            f"{category:<22}"
            f"total={category_total:<3} "
            f"pass={stats['passed']:<3} "
            f"fail={stats['failed']:<3} "
            f"error={stats['errors']:<3}"
        )

    print(SEPARATOR)

    # Important:
    #
    # We deliberately do NOT raise AssertionError here.
    # This lets the entire evaluation suite complete and
    # gives us the full failure pattern instead of stopping
    # at the first bad question.


if __name__ == "__main__":
    main()
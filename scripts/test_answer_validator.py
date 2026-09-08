from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.generation.answer_validator import (
    AnswerValidator,
)


def print_result(
    title: str,
    result: dict,
) -> None:

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        "Validation passed:   ",
        result["validation_passed"],
    )

    print(
        "Citation valid:      ",
        result["citation_valid"],
    )

    print(
        "Insufficient:        ",
        result["insufficient_evidence"],
    )

    print(
        "Cited evidence IDs:  ",
        result["cited_evidence_ids"],
    )

    print(
        "Invalid citations:   ",
        result["invalid_citation_ids"],
    )

    print(
        "Unused evidence IDs: ",
        result["unused_evidence_ids"],
    )

    print("\nCITED SOURCES")

    if not result["cited_sources"]:
        print("  None")

    for source in result[
        "cited_sources"
    ]:

        print(
            f"  [{source['evidence_id']}] "
            f"{source['source']} "
            f"| page={source['page']} "
            f"| chunk={source['chunk_id']}"
        )

    print("\nERRORS")

    if not result["errors"]:
        print("  None")

    for error in result["errors"]:
        print(
            f"  - {error}"
        )

    print("\nWARNINGS")

    if not result["warnings"]:
        print("  None")

    for warning in result["warnings"]:
        print(
            f"  - {warning}"
        )


def main():

    print("=" * 80)
    print("IOCL ANSWER VALIDATOR TEST")
    print("=" * 80)

    evidence = [
        {
            "evidence_id": 1,
            "chunk_id": "chunk-whistle-7",
            "text": (
                "Whistle-blowers shall be protected "
                "against victimization."
            ),
            "metadata": {
                "source": (
                    "Whistle_Blower_policy.pdf"
                ),
                "page": 7,
            },
        },
        {
            "evidence_id": 2,
            "chunk_id": "chunk-whistle-4",
            "text": (
                "The policy describes the procedure "
                "for making disclosures."
            ),
            "metadata": {
                "source": (
                    "Whistle_Blower_policy.pdf"
                ),
                "page": 4,
            },
        },
        {
            "evidence_id": 3,
            "chunk_id": "chunk-code-4",
            "text": (
                "Independent Directors participate "
                "in Board committees."
            ),
            "metadata": {
                "source": (
                    "Code_of_Conduct_for_"
                    "Board_Members_&_SMP.pdf"
                ),
                "page": 4,
            },
        },
    ]

    validator = AnswerValidator()

    # --------------------------------------------------
    # TEST 1
    # Valid citation
    # --------------------------------------------------

    answer_1 = (
        "Whistleblowers are protected against "
        "victimization [1]."
    )

    result_1 = validator.validate(
        answer=answer_1,
        evidence=evidence,
    )

    print_result(
        "TEST 1: VALID CITATION",
        result_1,
    )

    # --------------------------------------------------
    # TEST 2
    # Invalid citation
    # --------------------------------------------------

    answer_2 = (
        "Whistleblowers are protected against "
        "victimization [99]."
    )

    result_2 = validator.validate(
        answer=answer_2,
        evidence=evidence,
    )

    print_result(
        "TEST 2: INVALID CITATION",
        result_2,
    )

    # --------------------------------------------------
    # TEST 3
    # Factual answer with no citation
    # --------------------------------------------------

    answer_3 = (
        "Whistleblowers are protected against "
        "victimization."
    )

    result_3 = validator.validate(
        answer=answer_3,
        evidence=evidence,
    )

    print_result(
        "TEST 3: MISSING CITATION",
        result_3,
    )

    # --------------------------------------------------
    # TEST 4
    # Insufficient evidence
    # --------------------------------------------------

    answer_4 = (
        "There is insufficient evidence in the "
        "provided documents to answer this question."
    )

    result_4 = validator.validate(
        answer=answer_4,
        evidence=evidence,
    )

    print_result(
        "TEST 4: INSUFFICIENT EVIDENCE",
        result_4,
    )

    # --------------------------------------------------
    # TEST 5
    # Multiple valid citations
    # --------------------------------------------------

    answer_5 = (
        "The policy protects whistleblowers "
        "against victimization [1]. "
        "It also describes a disclosure "
        "procedure [2]."
    )

    result_5 = validator.validate(
        answer=answer_5,
        evidence=evidence,
    )

    print_result(
        "TEST 5: MULTIPLE CITATIONS",
        result_5,
    )

    print("\n" + "=" * 80)
    print("ANSWER VALIDATOR TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
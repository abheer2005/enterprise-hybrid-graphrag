from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.generation.grounding_validator import (
    GroundingValidator,
)


def print_result(
    title: str,
    result,
):

    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )

    print(
        f"Grounding passed:   "
        f"{result.passed}"
    )

    print(
        f"Claims checked:     "
        f"{len(result.claims)}"
    )

    print(
        f"Supported claims:   "
        f"{len(result.supported_claims)}"
    )

    print(
        f"Unsupported claims: "
        f"{len(result.unsupported_claims)}"
    )

    print(
        "\nCLAIMS"
    )

    for index, claim in enumerate(
        result.claims,
        start=1,
    ):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"CLAIM #{index}"
        )

        print(
            f"Text: "
            f"{claim.claim}"
        )

        print(
            f"Citations: "
            f"{claim.citations}"
        )

        print(
            f"Evidence IDs: "
            f"{claim.evidence_ids}"
        )

        print(
            f"Label: "
            f"{claim.label}"
        )

        print(
            f"Entailment:    "
            f"{claim.entailment_score:.4f}"
        )

        print(
            f"Contradiction: "
            f"{claim.contradiction_score:.4f}"
        )

        print(
            f"Neutral:       "
            f"{claim.neutral_score:.4f}"
        )

        print(
            f"Supported: "
            f"{claim.supported}"
        )


def main():

    print(
        "=" * 80
    )

    print(
        "IOCL NLI CLAIM GROUNDING TEST"
    )

    print(
        "=" * 80
    )

    validator = GroundingValidator(
        min_entailment_score=0.70,
    )

    evidence = [
        {
            "evidence_id": 1,
            "chunk_id": "test-whistle-1",
            "text": (
                "Whistle-blowers are entitled to "
                "protection against victimization. "
                "The identity of the Whistle-blower "
                "shall be kept confidential during "
                "the investigation."
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
            "chunk_id": "test-rpt-1",
            "text": (
                "All Related Party Transactions "
                "and Material Modifications shall "
                "require prior approval of the "
                "Audit Committee."
            ),
            "metadata": {
                "source": (
                    "RPT_Policy.pdf"
                ),
                "page": 3,
            },
        },
        {
            "evidence_id": 3,
            "chunk_id": "test-csr-1",
            "text": (
                "The company shall earmark at "
                "least two percent of the average "
                "net profits made during the three "
                "immediately preceding financial "
                "years for CSR activities."
            ),
            "metadata": {
                "source": (
                    "IOC_S&CSR_Policy.pdf"
                ),
                "page": 1,
            },
        },
    ]

    tests = [
        (
            "TEST 1: SUPPORTED CLAIM",
            (
                "Whistle-blowers are entitled to "
                "protection against victimization [1]."
            ),
        ),

        (
            "TEST 2: FABRICATED MONETARY REWARD",
            (
                "Whistle-blowers receive a monetary "
                "reward for reporting misconduct [1]."
            ),
        ),

        (
            "TEST 3: SUPPORTED RPT CLAIM",
            (
                "Related Party Transactions require "
                "prior approval of the Audit "
                "Committee [2]."
            ),
        ),

        (
            "TEST 4: WRONG EVIDENCE CITATION",
            (
                "The company earmarks at least two "
                "percent of average net profits for "
                "CSR activities [2]."
            ),
        ),

        (
            "TEST 5: SUPPORTED CSR CLAIM",
            (
                "The company earmarks at least two "
                "percent of average net profits for "
                "CSR activities [3]."
            ),
        ),

        (
            "TEST 6: MIXED CLAIMS",
            (
                "Whistle-blowers are protected "
                "against victimization [1]. "
                "Whistle-blowers automatically receive "
                "a promotion after reporting "
                "misconduct [1]."
            ),
        ),
    ]

    for title, answer in tests:

        result = validator.validate(
            answer=answer,
            evidence=evidence,
        )

        print_result(
            title,
            result,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "NLI CLAIM GROUNDING TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
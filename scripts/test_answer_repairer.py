from pathlib import Path
import sys
from dataclasses import dataclass


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.generation.answer_repairer import (
    AnswerRepairer,
)


@dataclass
class FakeClaim:

    claim: str

    supported: bool


@dataclass
class FakeGroundingResult:

    claims: list[FakeClaim]


def run_test(
    title,
    answer,
    claims,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )

    repairer = AnswerRepairer()

    grounding = (
        FakeGroundingResult(
            claims=claims
        )
    )

    result = repairer.repair(
        answer=answer,
        grounding_result=grounding,
    )

    print(
        "\nORIGINAL ANSWER"
    )

    print(
        "-" * 80
    )

    print(
        answer
    )

    print(
        "\nREPAIRED ANSWER"
    )

    print(
        "-" * 80
    )

    print(
        result[
            "repaired_answer"
        ]
    )

    print(
        "\nSTATS"
    )

    print(
        "-" * 80
    )

    print(
        "Changed:",
        result["changed"],
    )

    print(
        "Supported claims:",
        result[
            "supported_claims"
        ],
    )

    print(
        "Unsupported claims:",
        result[
            "unsupported_claims"
        ],
    )

    print(
        "Usable:",
        result["usable"],
    )

    print(
        "Removed:"
    )

    for fragment in result[
        "removed_fragments"
    ]:

        print(
            "  -",
            fragment,
        )


def main():

    print(
        "=" * 80
    )

    print(
        "IOCL ANSWER REPAIRER TEST"
    )

    print(
        "=" * 80
    )

    # -----------------------------------------------------
    # TEST 1
    # All claims supported.
    # Answer should remain unchanged.
    # -----------------------------------------------------

    answer_1 = (
        "Whistle-blowers are protected against "
        "victimization [1]."
    )

    claims_1 = [
        FakeClaim(
            claim=(
                "Whistle-blowers are protected against "
                "victimization ."
            ),
            supported=True,
        ),
    ]

    run_test(
        "TEST 1: FULLY SUPPORTED ANSWER",
        answer_1,
        claims_1,
    )

    # -----------------------------------------------------
    # TEST 2
    # One supported + one fabricated.
    # Fabricated bullet must disappear.
    # -----------------------------------------------------

    answer_2 = """
### Whistleblower Protection

* Whistle-blowers are protected against victimization [1].
* Whistle-blowers automatically receive a promotion after reporting misconduct [1].
""".strip()

    claims_2 = [
        FakeClaim(
            claim=(
                "Whistle-blowers are protected against "
                "victimization ."
            ),
            supported=True,
        ),

        FakeClaim(
            claim=(
                "Whistle-blowers automatically receive "
                "a promotion after reporting misconduct ."
            ),
            supported=False,
        ),
    ]

    run_test(
        "TEST 2: REMOVE FABRICATED CLAIM",
        answer_2,
        claims_2,
    )

    # -----------------------------------------------------
    # TEST 3
    # Similar to our RPT situation:
    # several claims valid, one unsupported.
    # -----------------------------------------------------

    answer_3 = """
### Related Party Transactions

* The policy must be reviewed by the Board once every three years [4].
* Audit Committee approval is required for all RPTs and Material Modifications [4].
* The company gives every related party a monetary reward [4].
* Omnibus approval may be granted for repetitive transactions subject to quarterly review [4].
""".strip()

    claims_3 = [
        FakeClaim(
            claim=(
                "The policy must be reviewed by the "
                "Board once every three years ."
            ),
            supported=True,
        ),

        FakeClaim(
            claim=(
                "Audit Committee approval is required "
                "for all RPTs and Material Modifications ."
            ),
            supported=True,
        ),

        FakeClaim(
            claim=(
                "The company gives every related party "
                "a monetary reward ."
            ),
            supported=False,
        ),

        FakeClaim(
            claim=(
                "Omnibus approval may be granted for "
                "repetitive transactions subject to "
                "quarterly review ."
            ),
            supported=True,
        ),
    ]

    run_test(
        "TEST 3: PARTIALLY SUPPORTED RPT ANSWER",
        answer_3,
        claims_3,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ANSWER REPAIRER TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv

from src.generation.qa_pipeline import QAPipeline


def main():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )

    print("=" * 80)
    print("IOCL QA PIPELINE PERFORMANCE TEST")
    print("=" * 80)

    pipeline = QAPipeline(
        api_key=api_key,
    )

    try:

        query = "How are whistleblowers protected?"

        print("\nQUESTION")
        print("-" * 80)
        print(query)

        result = pipeline.answer(
            query=query,
        )

        print("\nFINAL ANSWER")
        print("-" * 80)
        print(result.answer)

        print("\nDECISION")
        print("-" * 80)
        print(f"Accepted:     {result.accepted}")
        print(f"Grounded:     {result.grounded}")
        print(
            f"Insufficient: "
            f"{result.insufficient_evidence}"
        )

        print("\nPERFORMANCE")
        print("-" * 80)

        for name, seconds in result.performance.items():
            print(
                f"{name:<40} "
                f"{seconds:>10.4f} sec"
            )

        print("\nBOTTLENECK")
        print("-" * 80)

        ignored = {
            "total_seconds",
            "pipeline_processing_seconds",
        }

        stages = {
            name: seconds
            for name, seconds
            in result.performance.items()
            if name not in ignored
        }

        bottleneck_name = max(
            stages,
            key=stages.get,
        )

        bottleneck_seconds = stages[
            bottleneck_name
        ]

        total_seconds = result.performance.get(
            "total_seconds",
            0.0,
        )

        if total_seconds > 0:

            percentage = (
                bottleneck_seconds
                / total_seconds
                * 100
            )

        else:

            percentage = 0.0

        print(
            f"Slowest stage: {bottleneck_name}"
        )

        print(
            f"Time:          "
            f"{bottleneck_seconds:.4f} sec"
        )

        print(
            f"Share:         "
            f"{percentage:.2f}%"
        )

    finally:

        pipeline.close()

    print("\n" + "=" * 80)
    print("PERFORMANCE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
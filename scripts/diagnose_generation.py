import os
import traceback

from dotenv import load_dotenv

from src.generation.answer_generator import (
    AnswerGenerator,
)


def main():

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found."
        )

    generator = AnswerGenerator(
        api_key=api_key,
        model_name="gemini-3.6-flash",
        max_attempts=1,
    )

    evidence = [
        {
            "chunk_id": "diagnostic_chunk_1",
            "text": (
                "Whistle-blowers are entitled to "
                "protection against victimization."
            ),
            "metadata": {
                "source": "Diagnostic Evidence",
                "page": 1,
            },
        }
    ]

    try:

        result = generator.generate(
            query=(
                "How are whistleblowers protected?"
            ),
            selected_chunks=evidence,
        )

        print("=" * 80)
        print("GENERATION SUCCESS")
        print("=" * 80)

        print(result.answer)

    except Exception as error:

        print("=" * 80)
        print("GENERATION FAILED")
        print("=" * 80)

        print(
            "Wrapper type:",
            type(error).__name__,
        )

        print(
            "Wrapper message:",
            str(error),
        )

        cause = error.__cause__

        if cause is None:

            print(
                "\nNo chained provider "
                "exception was found."
            )

        else:

            print("\nPROVIDER EXCEPTION")
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

        print("\nTRACEBACK")
        print("-" * 80)

        traceback.print_exception(
            type(error),
            error,
            error.__traceback__,
        )


if __name__ == "__main__":
    main()
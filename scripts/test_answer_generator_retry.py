"""
Deterministic tests for AnswerGenerator retry handling.

No real Gemini API request is made.
No retrieval or NLI model is loaded.

Tests:
1. Temporary 503 -> retry -> success.
2. Permanent 401 -> fail immediately.
3. Repeated 503 -> stop after max attempts.
"""

from src.generation.answer_generator import (
    AnswerGenerator,
    AnswerGenerationError,
)


class FakeResponse:

    def __init__(self, text):
        self.text = text


class Temporary503ThenSuccess:
    """
    Fail twice with 503, then succeed.
    """

    def __init__(self):
        self.calls = 0

    def generate_content(
        self,
        model,
        contents,
    ):
        self.calls += 1

        if self.calls <= 2:
            raise RuntimeError(
                "503 UNAVAILABLE: temporary high demand"
            )

        return FakeResponse(
            "Supported answer [1]."
        )


class Permanent401:
    """
    Authentication failure must not be retried.
    """

    def __init__(self):
        self.calls = 0

    def generate_content(
        self,
        model,
        contents,
    ):
        self.calls += 1

        raise RuntimeError(
            "401 UNAUTHENTICATED: API key not valid"
        )


class Always503:
    """
    Temporary failure continues through every attempt.
    """

    def __init__(self):
        self.calls = 0

    def generate_content(
        self,
        model,
        contents,
    ):
        self.calls += 1

        raise RuntimeError(
            "503 UNAVAILABLE: service unavailable"
        )


class FakeClient:

    def __init__(self, models):
        self.models = models


def build_generator(
    fake_models,
    max_attempts=4,
):
    """
    Construct AnswerGenerator without making real API
    requests.

    Retry delays are zero so tests finish immediately.
    """

    generator = AnswerGenerator(
        api_key="test-key",
        max_attempts=max_attempts,
        initial_retry_delay=0.0,
        max_retry_delay=0.0,
        retry_jitter=0.0,
    )

    generator.client = FakeClient(
        fake_models
    )

    return generator


def test_temporary_failure_then_success():

    print("=" * 80)
    print("TEST 1: TEMPORARY 503 -> RETRY -> SUCCESS")
    print("=" * 80)

    models = Temporary503ThenSuccess()

    generator = build_generator(
        models,
        max_attempts=4,
    )

    response = generator._generate_with_retry(
        prompt="test prompt"
    )

    assert response.text == (
        "Supported answer [1]."
    )

    assert models.calls == 3, (
        f"Expected 3 calls, got {models.calls}."
    )

    print(
        f"Calls: {models.calls}"
    )
    print("PASS")


def test_permanent_failure():

    print("\n" + "=" * 80)
    print("TEST 2: PERMANENT 401 -> NO RETRY")
    print("=" * 80)

    models = Permanent401()

    generator = build_generator(
        models,
        max_attempts=4,
    )

    try:

        generator._generate_with_retry(
            prompt="test prompt"
        )

    except AnswerGenerationError:
        pass

    else:
        raise AssertionError(
            "Expected AnswerGenerationError."
        )

    assert models.calls == 1, (
        "401 failure should not be retried. "
        f"Calls: {models.calls}"
    )

    print(
        f"Calls: {models.calls}"
    )
    print("PASS")


def test_retry_exhaustion():

    print("\n" + "=" * 80)
    print("TEST 3: REPEATED 503 -> RETRIES EXHAUSTED")
    print("=" * 80)

    models = Always503()

    generator = build_generator(
        models,
        max_attempts=4,
    )

    try:

        generator._generate_with_retry(
            prompt="test prompt"
        )

    except AnswerGenerationError:
        pass

    else:
        raise AssertionError(
            "Expected AnswerGenerationError."
        )

    assert models.calls == 4, (
        "Expected exactly 4 attempts. "
        f"Calls: {models.calls}"
    )

    print(
        f"Calls: {models.calls}"
    )
    print("PASS")


def main():

    print("=" * 80)
    print("IOCL ANSWER GENERATOR RETRY TEST")
    print("=" * 80)

    test_temporary_failure_then_success()
    test_permanent_failure()
    test_retry_exhaustion()

    print("\n" + "=" * 80)
    print("ALL RETRY TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
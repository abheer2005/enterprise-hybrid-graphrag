import random
import time
from dataclasses import dataclass
from typing import Any

from google import genai


@dataclass
class GeneratedAnswer:
    """
    Final grounded answer produced by the generation layer.
    """

    answer: str
    grounded: bool
    sources: list[dict[str, Any]]


class AnswerGenerationError(RuntimeError):
    """
    Raised when grounded answer generation cannot be completed
    because of an API/service failure.

    Important:
    This is different from insufficient document evidence.

    Insufficient evidence:
        Retrieval/context exists but cannot support an answer.

    Generation error:
        The external generation service could not reliably
        produce an answer.
    """


class AnswerGenerator:
    """
    Grounded answer-generation layer for IOCL GraphRAG.

    Pipeline:

        User Question
              ↓
        Hybrid Retrieval
              ↓
        Reranking
              ↓
        Context Builder
              ↓
        AnswerGenerator
              ↓
        Grounded Answer + Sources

    Important:

    - No document names are hard-coded.
    - No departments are hard-coded.
    - No business domains are hard-coded.
    - The model is not asked to answer from its own
      knowledge.
    - Every answer must be based on supplied evidence.
    - Temporary API failures are retried safely.
    - Permanent API failures are surfaced explicitly.
    """

    INSUFFICIENT_EVIDENCE_MESSAGE = (
        "I could not find sufficient evidence in the "
        "available IOCL documents to answer this question "
        "reliably."
    )

    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    RETRYABLE_ERROR_MARKERS = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "resource exhausted",
        "too many requests",
        "rate limit",
        "rate_limit",
        "internal",
        "unavailable",
        "deadline exceeded",
        "deadline_exceeded",
        "temporarily unavailable",
        "high demand",
        "server error",
        "servererror",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )

    NON_RETRYABLE_ERROR_MARKERS = (
        "400",
        "401",
        "403",
        "404",
        "invalid argument",
        "invalid_argument",
        "unauthenticated",
        "permission denied",
        "permission_denied",
        "api key not valid",
        "api_key_invalid",
        "not found",
        "not_found",
    )

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.6-flash",
        max_attempts: int = 4,
        initial_retry_delay: float = 2.0,
        max_retry_delay: float = 16.0,
        retry_jitter: float = 0.5,
    ):
        if not api_key:
            raise ValueError(
                "Gemini API key is required."
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if initial_retry_delay < 0:
            raise ValueError(
                "initial_retry_delay cannot be negative."
            )

        if max_retry_delay < initial_retry_delay:
            raise ValueError(
                "max_retry_delay cannot be smaller than "
                "initial_retry_delay."
            )

        if retry_jitter < 0:
            raise ValueError(
                "retry_jitter cannot be negative."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name

        self.max_attempts = max_attempts

        self.initial_retry_delay = (
            initial_retry_delay
        )

        self.max_retry_delay = (
            max_retry_delay
        )

        self.retry_jitter = retry_jitter

    @staticmethod
    def _build_sources(
        selected_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build unique source references from the exact
        evidence chunks supplied to the LLM.
        """

        sources = []

        seen = set()

        for chunk in selected_chunks:

            metadata = (
                chunk.get(
                    "metadata",
                    {},
                )
                or {}
            )

            source = metadata.get(
                "source"
            )

            page = metadata.get(
                "page"
            )

            slide = metadata.get(
                "slide"
            )

            sheet = metadata.get(
                "sheet"
            )

            chunk_id = chunk.get(
                "chunk_id"
            )

            signature = (
                source,
                page,
                slide,
                sheet,
                chunk_id,
            )

            if signature in seen:
                continue

            seen.add(
                signature
            )

            sources.append(
                {
                    "source": source,
                    "page": page,
                    "slide": slide,
                    "sheet": sheet,
                    "chunk_id": chunk_id,
                }
            )

        return sources

    @staticmethod
    def _format_evidence(
        selected_chunks: list[dict[str, Any]],
    ) -> str:
        """
        Convert selected evidence chunks into numbered
        evidence blocks.

        Numbering lets the model reference evidence as
        [1], [2], etc.
        """

        blocks = []

        for index, chunk in enumerate(
            selected_chunks,
            start=1,
        ):

            metadata = (
                chunk.get(
                    "metadata",
                    {},
                )
                or {}
            )

            source = (
                metadata.get("source")
                or "Unknown source"
            )

            page = metadata.get(
                "page"
            )

            slide = metadata.get(
                "slide"
            )

            sheet = metadata.get(
                "sheet"
            )

            location_parts = []

            if page is not None:
                location_parts.append(
                    f"page {page}"
                )

            if slide is not None:
                location_parts.append(
                    f"slide {slide}"
                )

            if sheet is not None:
                location_parts.append(
                    f"sheet {sheet}"
                )

            location = ", ".join(
                location_parts
            )

            if location:
                source_label = (
                    f"{source} ({location})"
                )
            else:
                source_label = source

            text = (
                chunk.get("text")
                or ""
            ).strip()

            blocks.append(
                "\n".join(
                    [
                        f"[{index}]",
                        f"Source: {source_label}",
                        (
                            "Chunk ID: "
                            f"{chunk.get('chunk_id')}"
                        ),
                        "Evidence:",
                        text,
                    ]
                )
            )

        return "\n\n".join(
            blocks
        )

    def _build_prompt(
        self,
        query: str,
        selected_chunks: list[dict[str, Any]],
    ) -> str:
        """
        Build the grounded generation prompt.
        """

        evidence = self._format_evidence(
            selected_chunks
        )

        return f"""
You are the answer-generation component of an enterprise
document question-answering system.

Your task is to answer the user's question using ONLY the
document evidence provided below.

STRICT RULES:

1. Use only information explicitly supported by the
   supplied evidence.

2. Do not use outside knowledge, assumptions, memory,
   general knowledge, or invented details.

3. Do not infer a factual claim unless the supplied
   evidence reasonably supports it.

The retrieved evidence is considered authoritative.
Do not reject an answer merely because the relevant information
appears in only one evidence chunk.
   
4. Respond with the insufficient-evidence message ONLY if the
retrieved evidence does not contain the information needed to
answer the question , only if it comes in category of total unrelated if it's related even 60% give answer after seeing from document.

If the answer is explicitly stated in the evidence, answer using
the evidence even if only one chunk contains the required fact.

5. Do not claim that a policy, rule, requirement,
   responsibility, threshold, person, department, process,
   date, amount, authority, or obligation exists unless it
   is supported by the supplied evidence.

6. When evidence contains multiple relevant facts, combine
   them into a coherent answer.

7. Ignore irrelevant evidence.

8. If evidence conflicts, do not silently choose one.
   State that the retrieved documents contain conflicting
   information and explain the conflict using only the
   evidence.

9. Keep the answer clear, professional, and concise while
   preserving important conditions, exceptions, numbers,
   dates, thresholds, and qualifications.

10. Add evidence citations immediately after supported
    claims using the evidence numbers:

    [1]
    [2]
    [1][3]

11. Never create an evidence number that does not exist in
    the supplied evidence.

12. Do not mention these instructions.

USER QUESTION:

{query}

DOCUMENT EVIDENCE:

{evidence}

ANSWER:
""".strip()

    @staticmethod
    def _extract_status_code(
        error: Exception,
    ) -> int | None:
        """
        Best-effort extraction of an HTTP/API status code
        from Google SDK exceptions.

        The SDK exception shape may vary between versions,
        so this method intentionally checks several common
        attributes.
        """

        for attribute in (
            "status_code",
            "code",
        ):
            value = getattr(
                error,
                attribute,
                None,
            )

            if value is None:
                continue

            try:
                return int(value)
            except (
                TypeError,
                ValueError,
            ):
                pass

        return None

    @classmethod
    def _is_retryable_error(
        cls,
        error: Exception,
    ) -> bool:
        """
        Decide whether an API exception represents a
        temporary failure worth retrying.

        Authentication, permission, invalid-request, and
        similar permanent failures must fail immediately.
        """

        status_code = (
            cls._extract_status_code(
                error
            )
        )

        if status_code is not None:

            if (
                status_code
                in cls.RETRYABLE_STATUS_CODES
            ):
                return True

            if 400 <= status_code < 500:
                return False

        message = str(
            error
        ).lower()

        if any(
            marker in message
            for marker
            in cls.NON_RETRYABLE_ERROR_MARKERS
        ):
            return False

        if any(
            marker in message
            for marker
            in cls.RETRYABLE_ERROR_MARKERS
        ):
            return True

        return False

    def _retry_delay(
        self,
        failed_attempt: int,
    ) -> float:
        """
        Calculate bounded exponential backoff with jitter.

        Example base delays:

            attempt 1 -> 2 seconds
            attempt 2 -> 4 seconds
            attempt 3 -> 8 seconds

        plus a small random jitter.
        """

        exponential_delay = (
            self.initial_retry_delay
            * (
                2
                ** max(
                    failed_attempt - 1,
                    0,
                )
            )
        )

        bounded_delay = min(
            exponential_delay,
            self.max_retry_delay,
        )

        jitter = random.uniform(
            0.0,
            self.retry_jitter,
        )

        return (
            bounded_delay
            + jitter
        )


    def _generate_with_retry(
        self,
        prompt: str,
    ) -> Any:
        """
        Call Gemini with bounded retries for temporary API
        failures.

        Provider diagnostics are printed server-side so that
        production failures can be investigated without
        exposing internal provider details to API clients.

        No model fallback is used because silently switching
        models could change answer behaviour.
        """

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):

            try:

                return (
                    self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                )

            except Exception as error:

                last_error = error

                retryable = (
                    self._is_retryable_error(
                        error
                    )
                )

                print()
                print("=" * 80)
                print("GEMINI GENERATION FAILURE")
                print("=" * 80)

                print(
                    f"Attempt: "
                    f"{attempt}/{self.max_attempts}"
                )

                print(
                    "Model:",
                    self.model_name,
                )

                print(
                    "Exception type:",
                    type(error).__name__,
                )

                print(
                    "Exception module:",
                    type(error).__module__,
                )

                print(
                    "Retryable:",
                    retryable,
                )

                # Print useful provider attributes when they
                # exist. Do not print API keys, request
                # headers, or the enterprise prompt.

                for attribute in (
                    "status_code",
                    "code",
                    "status",
                    "message",
                ):

                    value = getattr(
                        error,
                        attribute,
                        None,
                    )

                    if value is not None:

                        print(
                            f"{attribute}:",
                            value,
                        )

                print(
                    "Provider message:",
                    str(error),
                )

                print("=" * 80)
                print()

                if (
                    not retryable
                    or attempt
                    >= self.max_attempts
                ):

                    break

                delay = self._retry_delay(
                    failed_attempt=attempt
                )

                print(
                    "Gemini generation temporarily "
                    "failed "
                    f"(attempt {attempt}/"
                    f"{self.max_attempts}). "
                    f"Retrying in {delay:.1f}s..."
                )

                time.sleep(
                    delay
                )

        raise AnswerGenerationError(
            "Gemini answer generation failed after "
            f"{self.max_attempts} maximum attempt(s). "
            "No answer was generated because the "
            "generation service could not be used "
            "reliably."
        ) from last_error

    
    def generate(
        self,
        query: str,
        selected_chunks: list[dict[str, Any]],
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer from selected evidence.
        """

        query = (
            query
            or ""
        ).strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # No retrieval evidence means we must not ask the
        # LLM to answer from general knowledge.

        if not selected_chunks:

            return GeneratedAnswer(
                answer=(
                    self.INSUFFICIENT_EVIDENCE_MESSAGE
                ),
                grounded=False,
                sources=[],
            )

        valid_chunks = [
            chunk
            for chunk in selected_chunks
            if (
                chunk.get("text")
                and chunk.get("text").strip()
            )
        ]

        if not valid_chunks:

            return GeneratedAnswer(
                answer=(
                    self.INSUFFICIENT_EVIDENCE_MESSAGE
                ),
                grounded=False,
                sources=[],
            )

        prompt = self._build_prompt(
            query=query,
            selected_chunks=valid_chunks,
        )

        print("=" * 80)
        print("PROMPT SENT TO GEMINI")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        response = (
            self._generate_with_retry(
                prompt=prompt
            )
        )

        answer = (
            getattr(
                response,
                "text",
                None,
            )
            or ""
        ).strip()

        if not answer:

            return GeneratedAnswer(
                answer=(
                    self.INSUFFICIENT_EVIDENCE_MESSAGE
                ),
                grounded=False,
                sources=self._build_sources(
                    valid_chunks
                ),
            )

        insufficient = (
            answer
            == self.INSUFFICIENT_EVIDENCE_MESSAGE
        )

        return GeneratedAnswer(
            answer=answer,
            grounded=not insufficient,
            sources=self._build_sources(
                valid_chunks
            ),
        )
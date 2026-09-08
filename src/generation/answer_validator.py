import re
from typing import Any


class AnswerValidator:
    """
    Deterministic validation layer for grounded answers.

    Responsibilities:
        1. Extract citation IDs such as [1], [2], [3].
        2. Verify that every citation refers to supplied evidence.
        3. Detect whether an answer contains factual claims
           without any citations.
        4. Return only evidence that was actually cited.
        5. Handle explicit insufficient-evidence responses.

    Important:
        This validator does NOT attempt semantic claim verification.
        That belongs to the claim-grounding stage.

        This layer is intentionally deterministic and does not
        make another LLM call.

        Insufficient-evidence detection is intentionally tolerant
        of natural wording variation. The generator does not need
        to return one exact sentence for the validator to recognize
        that the available document evidence was insufficient.
    """

    # ---------------------------------------------------------
    # CITATION DETECTION
    # ---------------------------------------------------------

    CITATION_PATTERN = re.compile(
        r"\[(\d+)\]"
    )

    # ---------------------------------------------------------
    # INSUFFICIENT-EVIDENCE DETECTION
    # ---------------------------------------------------------
    #
    # We do NOT rely on one exact generator sentence.
    #
    # Detection has two layers:
    #
    #   A. Strong/direct insufficiency expressions
    #
    #   B. An inability/absence expression together with
    #      evidence/document/context scope.
    #
    # Example:
    #
    #   "I could not verify the specific answer from the
    #    retrieved evidence reliably."
    #
    # contains:
    #
    #   action signal -> "could not verify"
    #   scope signal  -> "retrieved" / "evidence"
    #
    # and is therefore recognized as an explicit
    # insufficient-evidence response.
    #
    # This is safer than matching generic phrases such as
    # "could not find" on their own.
    # ---------------------------------------------------------

    INSUFFICIENT_PATTERNS = (
        "insufficient evidence",
        "insufficient information",
        "not enough evidence",
        "not enough information",
        "lack of evidence",
        "lack of information",
        "lacks sufficient evidence",
        "lacks sufficient information",
        "no sufficient evidence",
        "no sufficient information",
    )

    INSUFFICIENT_ACTION_PATTERNS = (
        # Finding
        "could not find",
        "couldn't find",
        "cannot find",
        "can't find",

        # Verification
        "could not verify",
        "couldn't verify",
        "cannot verify",
        "can't verify",

        # Determination
        "could not determine",
        "couldn't determine",
        "cannot determine",
        "can't determine",

        # Establishment
        "could not establish",
        "couldn't establish",
        "cannot establish",
        "can't establish",

        # Confirmation
        "could not confirm",
        "couldn't confirm",
        "cannot confirm",
        "can't confirm",

        # Answerability
        "could not answer",
        "couldn't answer",
        "cannot answer",
        "can't answer",

        # Missing support
        "does not provide",
        "doesn't provide",
        "do not provide",
        "did not provide",

        "does not contain",
        "doesn't contain",
        "do not contain",

        "not available",
        "not found",

        "no evidence",
        "no information",

        "unable to verify",
        "unable to determine",
        "unable to establish",
        "unable to confirm",
        "unable to answer",
        "unable to find",
    )

    EVIDENCE_SCOPE_PATTERNS = (
        "evidence",
        "information",
        "document",
        "documents",
        "context",
        "source",
        "sources",
        "material",
        "materials",
        "retrieved",
        "provided",
        "available",
        "supplied",
        "corpus",
    )

    def __init__(
        self,
        require_citations: bool = True,
    ):
        self.require_citations = (
            require_citations
        )

    # ---------------------------------------------------------
    # CITATION EXTRACTION
    # ---------------------------------------------------------

    @classmethod
    def extract_citation_ids(
        cls,
        answer: str,
    ) -> list[int]:
        """
        Extract unique citation numbers while preserving
        the order in which they first appear.

        Example:

            "A [2]. B [1]. C [2]."

        becomes:

            [2, 1]
        """

        matches = (
            cls.CITATION_PATTERN.findall(
                answer or ""
            )
        )

        citation_ids = []
        seen = set()

        for match in matches:

            citation_id = int(
                match
            )

            if citation_id in seen:
                continue

            seen.add(
                citation_id
            )

            citation_ids.append(
                citation_id
            )

        return citation_ids

    # ---------------------------------------------------------
    # INSUFFICIENT-EVIDENCE DETECTION
    # ---------------------------------------------------------

    @classmethod
    def is_insufficient_answer(
        cls,
        answer: str,
    ) -> bool:
        """
        Detect explicit insufficient-evidence responses.

        The detector is wording-tolerant but deliberately
        conservative.

        It supports:

            1. Direct insufficiency expressions.

               Example:
                   "There is insufficient evidence..."

            2. Flexible evidence-gap expressions.

               Example:
                   "I could not verify this from the
                    retrieved evidence."

        For flexible detection, BOTH an inability/absence
        signal and an evidence/document/context signal are
        required.

        This prevents ordinary factual statements from being
        classified as insufficient merely because they contain
        words such as "find" or "available".
        """

        normalized = (
            answer
            or ""
        ).strip().lower()

        # An empty answer cannot provide a grounded answer.

        if not normalized:
            return True

        # Normalize whitespace so formatting, newlines,
        # tabs, or repeated spaces do not affect matching.

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        # -------------------------------------------------
        # 1. DIRECT INSUFFICIENCY EXPRESSIONS
        # -------------------------------------------------

        has_direct_insufficiency = any(
            pattern in normalized
            for pattern
            in cls.INSUFFICIENT_PATTERNS
        )

        if has_direct_insufficiency:
            return True

        # -------------------------------------------------
        # 2. FLEXIBLE EVIDENCE-GAP DETECTION
        # -------------------------------------------------

        has_action_signal = any(
            pattern in normalized
            for pattern
            in cls.INSUFFICIENT_ACTION_PATTERNS
        )

        has_evidence_scope = any(
            pattern in normalized
            for pattern
            in cls.EVIDENCE_SCOPE_PATTERNS
        )

        if (
            has_action_signal
            and has_evidence_scope
        ):
            return True

        return False

    # ---------------------------------------------------------
    # EVIDENCE MAPPING
    # ---------------------------------------------------------

    @staticmethod
    def _evidence_map(
        evidence: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """
        Build:

            evidence_id -> evidence object
        """

        mapping = {}

        for item in evidence:

            evidence_id = item.get(
                "evidence_id"
            )

            if evidence_id is None:
                continue

            try:
                evidence_id = int(
                    evidence_id
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            mapping[
                evidence_id
            ] = item

        return mapping

    # ---------------------------------------------------------
    # SOURCE RECORD
    # ---------------------------------------------------------

    @staticmethod
    def _source_record(
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert an evidence item into a compact
        provenance record.
        """

        metadata = (
            evidence.get(
                "metadata",
                {},
            )
            or {}
        )

        return {
            "evidence_id": evidence.get(
                "evidence_id"
            ),
            "chunk_id": evidence.get(
                "chunk_id"
            ),
            "source": metadata.get(
                "source"
            ),
            "page": metadata.get(
                "page"
            ),
            "slide": metadata.get(
                "slide"
            ),
            "sheet": metadata.get(
                "sheet"
            ),
        }

    # ---------------------------------------------------------
    # MAIN VALIDATION
    # ---------------------------------------------------------

    def validate(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Validate citations in a generated answer.

        The validator checks:

            - whether the answer exists,
            - whether citation IDs are valid,
            - whether citations are required,
            - whether the response explicitly indicates
              insufficient evidence,
            - which evidence was actually cited.

        Semantic support of individual claims is handled by
        the separate grounding validator.
        """

        answer = (
            answer
            or ""
        ).strip()

        # -------------------------------------------------
        # BUILD EVIDENCE MAP
        # -------------------------------------------------

        evidence_map = (
            self._evidence_map(
                evidence
            )
        )

        available_ids = set(
            evidence_map.keys()
        )

        # -------------------------------------------------
        # EXTRACT CITATIONS
        # -------------------------------------------------

        cited_ids = (
            self.extract_citation_ids(
                answer
            )
        )

        valid_citation_ids = [
            citation_id
            for citation_id in cited_ids
            if citation_id in available_ids
        ]

        invalid_citation_ids = [
            citation_id
            for citation_id in cited_ids
            if citation_id not in available_ids
        ]

        # -------------------------------------------------
        # DETECT INSUFFICIENT-EVIDENCE RESPONSE
        # -------------------------------------------------

        insufficient = (
            self.is_insufficient_answer(
                answer
            )
        )

        # -------------------------------------------------
        # RESOLVE CITED EVIDENCE
        # -------------------------------------------------

        cited_evidence = [
            evidence_map[
                citation_id
            ]
            for citation_id
            in valid_citation_ids
        ]

        cited_sources = [
            self._source_record(
                item
            )
            for item
            in cited_evidence
        ]

        errors = []
        warnings = []

        # -------------------------------------------------
        # EMPTY ANSWER
        # -------------------------------------------------

        if not answer:

            errors.append(
                "Generated answer is empty."
            )

        # -------------------------------------------------
        # INVALID CITATIONS
        # -------------------------------------------------

        if invalid_citation_ids:

            errors.append(
                "Answer contains citations that do not "
                "exist in the supplied evidence: "
                + ", ".join(
                    f"[{citation_id}]"
                    for citation_id
                    in invalid_citation_ids
                )
            )

        # -------------------------------------------------
        # MISSING CITATIONS
        # -------------------------------------------------
        #
        # Explicit insufficient-evidence responses do not
        # need citations because they are not presenting
        # a factual answer from the documents.
        # -------------------------------------------------

        if (
            self.require_citations
            and not insufficient
            and answer
            and not cited_ids
        ):

            errors.append(
                "Answer contains no evidence citations."
            )

        # -------------------------------------------------
        # CITATIONS EXIST BUT NONE RESOLVE
        # -------------------------------------------------

        if (
            cited_ids
            and not valid_citation_ids
        ):

            errors.append(
                "None of the answer citations resolve "
                "to supplied evidence."
            )

        # -------------------------------------------------
        # UNUSED EVIDENCE
        # -------------------------------------------------

        unused_evidence_ids = sorted(
            available_ids
            - set(
                valid_citation_ids
            )
        )

        if unused_evidence_ids:

            warnings.append(
                "Some retrieved evidence was not cited "
                "by the generated answer."
            )

        # -------------------------------------------------
        # CITATION VALIDITY
        # -------------------------------------------------

        citation_valid = (
            len(
                invalid_citation_ids
            )
            == 0
        )

        # A normal factual answer requires at least one
        # valid citation.
        #
        # An explicit insufficient-evidence response does
        # not.

        if (
            self.require_citations
            and not insufficient
        ):

            citation_valid = (
                citation_valid
                and len(
                    valid_citation_ids
                ) > 0
            )

        # -------------------------------------------------
        # FINAL VALIDATION
        # -------------------------------------------------

        validation_passed = (
            bool(answer)
            and citation_valid
            and len(errors) == 0
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        return {
            "validation_passed": (
                validation_passed
            ),
            "citation_valid": (
                citation_valid
            ),
            "insufficient_evidence": (
                insufficient
            ),
            "cited_evidence_ids": (
                valid_citation_ids
            ),
            "invalid_citation_ids": (
                invalid_citation_ids
            ),
            "unused_evidence_ids": (
                unused_evidence_ids
            ),
            "cited_evidence": (
                cited_evidence
            ),
            "cited_sources": (
                cited_sources
            ),
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "available_evidence": len(
                    evidence_map
                ),
                "citations_found": len(
                    cited_ids
                ),
                "valid_citations": len(
                    valid_citation_ids
                ),
                "invalid_citations": len(
                    invalid_citation_ids
                ),
                "cited_chunks": len(
                    cited_evidence
                ),
            },
        }
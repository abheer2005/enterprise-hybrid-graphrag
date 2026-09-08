import re
from typing import Any


class AnswerRepairer:
    """
    Repairs a generated answer after claim-level grounding.

    Purpose:
        Preserve supported answer content while removing
        factual claims that failed NLI grounding.

    This prevents one unsupported claim from causing an
    otherwise useful grounded answer to be discarded.

    Important:
        - This class does NOT call an LLM.
        - It does NOT invent replacement information.
        - It only keeps text already present in the answer.
        - Citation validity is checked again later.
    """

    CITATION_PATTERN = re.compile(
        r"\[(\d+)\]"
    )

    BULLET_PATTERN = re.compile(
        r"^\s*(?:[-*•]|\d+[.)])\s+"
    )

    HEADING_PATTERN = re.compile(
        r"^\s*#{1,6}\s+"
    )

    def __init__(
        self,
        min_supported_claims: int = 1,
    ):
        self.min_supported_claims = (
            min_supported_claims
        )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize text for conservative comparison.
        """

        text = (
            text
            or ""
        ).strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text.replace(
            "**",
            "",
        )

        return text.strip()

    @classmethod
    def _remove_citations(
        cls,
        text: str,
    ) -> str:
        """
        Remove citation markers from text before matching.
        """

        return cls.CITATION_PATTERN.sub(
            "",
            text or "",
        ).strip()

    @classmethod
    def _claim_text(
        cls,
        claim: Any,
    ) -> str:
        """
        Read claim text from either an object or dict.
        """

        if isinstance(
            claim,
            dict,
        ):
            return str(
                claim.get(
                    "claim",
                    "",
                )
                or ""
            )

        return str(
            getattr(
                claim,
                "claim",
                "",
            )
            or ""
        )

    @staticmethod
    def _claim_supported(
        claim: Any,
    ) -> bool:
        """
        Read support decision from either object or dict.
        """

        if isinstance(
            claim,
            dict,
        ):
            return bool(
                claim.get(
                    "supported",
                    False,
                )
            )

        return bool(
            getattr(
                claim,
                "supported",
                False,
            )
        )

    @classmethod
    def _similar(
        cls,
        first: str,
        second: str,
    ) -> bool:
        """
        Conservative text matching.

        We are not trying to perform semantic inference here.
        GroundingValidator already performed NLI.

        This method only identifies which original answer
        fragment corresponds to a validated claim.
        """

        first = cls._normalize(
            cls._remove_citations(
                first
            )
        ).lower()

        second = cls._normalize(
            cls._remove_citations(
                second
            )
        ).lower()

        if not first or not second:
            return False

        if first == second:
            return True

        if first in second:
            return True

        if second in first:
            return True

        return False

    @classmethod
    def _is_structural_line(
        cls,
        line: str,
    ) -> bool:
        """
        Detect headings or formatting-only lines that can
        safely remain when supported content exists below.
        """

        stripped = (
            line
            or ""
        ).strip()

        if not stripped:
            return True

        if cls.HEADING_PATTERN.match(
            stripped
        ):
            return True

        # Markdown heading-like line:
        #
        # **Budgeting**
        #
        if (
            stripped.startswith("**")
            and stripped.endswith("**")
            and len(
                cls.CITATION_PATTERN.findall(
                    stripped
                )
            )
            == 0
        ):
            return True

        return False

    @classmethod
    def _split_fragments(
        cls,
        answer: str,
    ) -> list[str]:
        """
        Split answer into relatively small fragments.

        We preserve line structure because generated answers
        commonly contain headings and bullet points.
        """

        fragments = []

        for raw_line in (
            answer
            or ""
        ).splitlines():

            line = raw_line.rstrip()

            if not line.strip():
                fragments.append("")
                continue

            # Keep headings intact.
            if cls._is_structural_line(
                line
            ):
                fragments.append(
                    line
                )
                continue

            # Keep bullet/list entries intact because a bullet
            # usually represents one coherent answer unit.
            if cls.BULLET_PATTERN.match(
                line
            ):
                fragments.append(
                    line
                )
                continue

            # For ordinary prose, split on sentence boundaries.
            sentences = re.split(
                r"(?<=[.!?])\s+",
                line.strip(),
            )

            fragments.extend(
                sentence
                for sentence in sentences
                if sentence.strip()
            )

        return fragments

    def repair(
        self,
        answer: str,
        grounding_result: Any,
    ) -> dict[str, Any]:
        """
        Remove fragments corresponding to unsupported claims.

        Returns:

            repaired_answer
            changed
            supported_claims
            unsupported_claims
            removed_fragments
            usable
        """

        answer = (
            answer
            or ""
        ).strip()

        if not answer:

            return {
                "repaired_answer": "",
                "changed": False,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "removed_fragments": [],
                "usable": False,
            }

        if grounding_result is None:

            return {
                "repaired_answer": answer,
                "changed": False,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "removed_fragments": [],
                "usable": True,
            }

        claims = list(
            getattr(
                grounding_result,
                "claims",
                [],
            )
            or []
        )

        supported_claims = [
            claim
            for claim in claims
            if self._claim_supported(
                claim
            )
        ]

        unsupported_claims = [
            claim
            for claim in claims
            if not self._claim_supported(
                claim
            )
        ]

        # Nothing failed, so preserve the answer exactly.
        if not unsupported_claims:

            return {
                "repaired_answer": answer,
                "changed": False,
                "supported_claims": len(
                    supported_claims
                ),
                "unsupported_claims": 0,
                "removed_fragments": [],
                "usable": (
                    len(
                        supported_claims
                    )
                    >= self.min_supported_claims
                ),
            }

        fragments = self._split_fragments(
            answer
        )

        kept = []

        removed = []

        for fragment in fragments:

            if not fragment.strip():

                kept.append(
                    fragment
                )

                continue

            if self._is_structural_line(
                fragment
            ):

                kept.append(
                    fragment
                )

                continue

            matches_unsupported = any(
                self._similar(
                    fragment,
                    self._claim_text(
                        claim
                    ),
                )
                for claim
                in unsupported_claims
            )

            if matches_unsupported:

                removed.append(
                    fragment
                )

                continue

            kept.append(
                fragment
            )

        # Remove excessive blank lines created by filtering.
        repaired_lines = []

        previous_blank = False

        for fragment in kept:

            blank = (
                not fragment.strip()
            )

            if (
                blank
                and previous_blank
            ):
                continue

            repaired_lines.append(
                fragment
            )

            previous_blank = blank

        repaired_answer = "\n".join(
            repaired_lines
        ).strip()

        # Remove headings left behind with no factual content
        # below them.
        repaired_answer = (
            self._remove_empty_sections(
                repaired_answer
            )
        )

        usable = (
            bool(
                repaired_answer
            )
            and len(
                supported_claims
            )
            >= self.min_supported_claims
        )

        return {
            "repaired_answer": (
                repaired_answer
            ),
            "changed": (
                repaired_answer
                != answer
            ),
            "supported_claims": len(
                supported_claims
            ),
            "unsupported_claims": len(
                unsupported_claims
            ),
            "removed_fragments": removed,
            "usable": usable,
        }

    @classmethod
    def _remove_empty_sections(
        cls,
        answer: str,
    ) -> str:
        """
        Remove trailing/empty headings created when all
        content under a section was filtered out.
        """

        lines = (
            answer
            or ""
        ).splitlines()

        cleaned = []

        for index, line in enumerate(
            lines
        ):

            if not cls._is_structural_line(
                line
            ):

                cleaned.append(
                    line
                )

                continue

            if not line.strip():

                cleaned.append(
                    line
                )

                continue

            has_content_after = False

            for later_line in lines[
                index + 1:
            ]:

                if not later_line.strip():
                    continue

                if cls._is_structural_line(
                    later_line
                ):
                    break

                has_content_after = True
                break

            if has_content_after:
                cleaned.append(
                    line
                )

        return "\n".join(
            cleaned
        ).strip()
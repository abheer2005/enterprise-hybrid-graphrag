from typing import Any


class ContextBuilder:
    """
    Builds grounded LLM context from ranked retrieval
    candidates.

    Responsibilities:
        - preserve reranker order
        - remove duplicate chunks
        - ignore empty chunks
        - preserve source/page/chunk provenance
        - enforce a context character budget

    This layer does NOT generate answers.
    """

    def __init__(
        self,
        max_chunks: int = 12,
        max_chars: int = 20000,
    ):
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    @staticmethod
    def _format_chunk(
        index: int,
        candidate: dict[str, Any],
    ) -> str:

        metadata = candidate.get(
            "metadata",
            {},
        ) or {}

        source = (
            metadata.get("source")
            or "Unknown source"
        )

        page = metadata.get("page")

        chunk_id = (
            candidate.get("chunk_id")
            or "unknown"
        )

        text = (
            candidate.get("text")
            or ""
        ).strip()

        header_parts = [
            f"Evidence {index}",
            f"Source: {source}",
        ]

        if page is not None:
            header_parts.append(
                f"Page: {page}"
            )

        header_parts.append(
            f"Chunk ID: {chunk_id}"
        )

        header = "\n".join(
            header_parts
        )

        return (
            f"{header}\n\n"
            f"{text}"
        )

    def build(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Convert ranked candidates into bounded,
        provenance-aware context.
        """

        selected = []

        seen_chunk_ids = set()

        total_chars = 0

        for candidate in candidates:

            if len(selected) >= self.max_chunks:
                break

            chunk_id = candidate.get(
                "chunk_id"
            )

            text = (
                candidate.get("text")
                or ""
            ).strip()

            if not chunk_id:
                continue

            if not text:
                continue

            if chunk_id in seen_chunk_ids:
                continue

            evidence_number = (
                len(selected) + 1
            )

            formatted = self._format_chunk(
                evidence_number,
                candidate,
            )

            # If adding this chunk exceeds the budget,
            # skip it unless no evidence has been selected.
            if (
                total_chars + len(formatted)
                > self.max_chars
            ):

                if selected:
                    continue

                # First chunk alone exceeds the budget.
                # Keep a bounded version instead of
                # returning no context.
                formatted = formatted[
                    :self.max_chars
                ]

            selected.append(
                {
                    "evidence_id": (
                        evidence_number
                    ),
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": candidate.get(
                        "metadata",
                        {},
                    ),
                    "rerank_score": (
                        candidate.get(
                            "rerank_score"
                        )
                    ),
                    "formatted": formatted,
                }
            )

            seen_chunk_ids.add(
                chunk_id
            )

            total_chars += len(
                formatted
            )

        context = "\n\n".join(
            item["formatted"]
            for item in selected
        )

        return {
            "context": context,
            "evidence": selected,
            "stats": {
                "input_candidates": len(
                    candidates
                ),
                "selected_chunks": len(
                    selected
                ),
                "context_chars": len(
                    context
                ),
                "max_chunks": self.max_chunks,
                "max_chars": self.max_chars,
            },
        }
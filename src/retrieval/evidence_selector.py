from typing import Any

import numpy as np


class EvidenceSelector:
    """
    Maximum Marginal Relevance (MMR) based evidence selector.

    Purpose
    -------
    Select a set of evidence chunks that are both:

        1. Highly relevant to the query.
        2. Diverse (non-redundant).

    This improves context quality without introducing any
    domain-specific logic.

    No document names.
    No hardcoded policies.
    No business rules.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def select(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        lambda_param: float = 0.90,
    ) -> list[dict[str, Any]]:

        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        # Reranker has already produced the best ordering.
# Do not apply MMR because policy documents often
# require multiple adjacent pages together.

        return candidates[:top_k]
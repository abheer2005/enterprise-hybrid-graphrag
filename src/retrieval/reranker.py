from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridReranker:
    """
    Generic second-stage cross-encoder reranker.

    Purpose
    -------
    Rank retrieved chunks according to how well each chunk
    answers the user's query.

    Retrieval signals such as vector similarity, graph support,
    and vector/graph agreement are retained for observability,
    but the final relevance ordering is produced by a
    cross-encoder that evaluates the query and chunk jointly.

    Important
    ---------
    - No document names are hard-coded.
    - No policy names are hard-coded.
    - No departments are hard-coded.
    - No business rules are hard-coded.
    - Works with future documents without domain-specific rules.
    """

    def __init__(
        self,
        embedding_model=None,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ):
        # embedding_model is intentionally accepted for backward
        # compatibility with the existing HybridRetriever and
        # QAPipeline constructors.
        #
        # It is NOT used for reranking.
        self.embedding_model = embedding_model

        self.model_name = model_name

        print(
            f"Loading cross-encoder reranker: "
            f"{self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

    @staticmethod
    def _normalize_cross_encoder_scores(
        raw_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raw cross-encoder logits to a stable 0..1
        relevance score using the sigmoid function.

        The ranking itself is unchanged because sigmoid is
        monotonic.
        """

        raw_scores = np.asarray(
            raw_scores,
            dtype="float32",
        ).reshape(-1)

        clipped = np.clip(
            raw_scores,
            -30.0,
            30.0,
        )

        return (
            1.0
            /
            (
                1.0
                + np.exp(-clipped)
            )
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using joint query-document scoring.

        Each pair is evaluated as:

            (user query, evidence chunk)

        This is fundamentally different from bi-encoder vector
        similarity because the cross-encoder can directly model
        interactions between words and concepts in the query and
        the candidate passage.
        """

        if not candidates:
            return []

        query = (
            query
            or ""
        ).strip()

        if not query:
            return candidates[:top_k]

        usable_candidates = []

        for candidate in candidates:

            text = (
                candidate.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not text:
                continue

            usable_candidates.append(
                candidate
            )

        if not usable_candidates:
            return []

        pairs = [
            [
                query,
                (
                    candidate.get(
                        "text",
                        "",
                    )
                    or ""
                ),
            ]
            for candidate in usable_candidates
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        raw_scores = np.asarray(
            raw_scores,
            dtype="float32",
        ).reshape(-1)

        normalized_scores = (
            self._normalize_cross_encoder_scores(
                raw_scores
            )
        )

        reranked = []

        for (
            candidate,
            raw_score,
            normalized_score,
        ) in zip(
            usable_candidates,
            raw_scores,
            normalized_scores,
        ):

            item = dict(candidate)

            vector_score = item.get(
                "vector_score"
            )

            if vector_score is None:
                vector_score = 0.0

            graph_match = bool(
                item.get(
                    "graph_match",
                    False,
                )
            )

            graph_relation_count = int(
                item.get(
                    "graph_relation_count",
                    0,
                )
                or 0
            )

            retrieval_sources = item.get(
                "retrieval_sources",
                [],
            )

            retrieval_agreement = (
                "vector" in retrieval_sources
                and
                "graph" in retrieval_sources
            )

            item[
                "cross_encoder_raw_score"
            ] = float(raw_score)

            item[
                "cross_encoder_score"
            ] = float(normalized_score)

            # Keep this field name because the rest of the
            # existing pipeline expects "rerank_score".
            item[
                "rerank_score"
            ] = float(normalized_score)

            # Preserve useful retrieval signals for debugging
            # and observability. They do NOT override semantic
            # relevance.
            item[
                "vector_score"
            ] = float(vector_score)

            item[
                "graph_match"
            ] = graph_match

            item[
                "graph_relation_count"
            ] = graph_relation_count

            item[
                "retrieval_agreement"
            ] = retrieval_agreement

            reranked.append(
                item
            )

        reranked.sort(
            key=lambda item: item.get(
                "rerank_score",
                0.0,
            ),
            reverse=True,
        )

        print(
            "\nCROSS-ENCODER RERANKING"
        )

        for i, item in enumerate(
            reranked[:top_k],
            start=1,
        ):

            metadata = item.get(
                "metadata",
                {},
            )

            print(
                f"{i}. "
                f"{metadata.get('source')} | "
                f"Page {metadata.get('page')} | "
                f"CrossEncoder="
                f"{item.get('cross_encoder_score', 0.0):.3f} | "
                f"Raw="
                f"{item.get('cross_encoder_raw_score', 0.0):.3f} | "
                f"Vector="
                f"{item.get('vector_score', 0.0):.3f} | "
                f"Graph="
                f"{item.get('graph_relation_count', 0)} | "
                f"Agreement="
                f"{item.get('retrieval_agreement', False)}"
            )

        return reranked[:top_k]
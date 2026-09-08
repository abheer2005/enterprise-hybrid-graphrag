from typing import Any

import numpy as np

from src.graph.neo4j_store import Neo4jStore


class SemanticEntityLinker:
    """
    Semantic entity linker for the IOCL knowledge graph.

    Purpose:
        Map a natural-language user query to relevant
        Entity nodes in Neo4j without relying on exact
        entity-name matching.

    Important:
        - No departments are hard-coded.
        - No document names are hard-coded.
        - No business domains are hard-coded.
        - No LLM call is required.

    The same embedding model already used by the vector
    retrieval layer is reused here.
    """

    def __init__(
        self,
        embedding_model,
        store: Neo4jStore | None = None,
    ):
        self.embedding_model = embedding_model

        self.store = store or Neo4jStore()

        self._owns_store = store is None

        self.entities: list[
            dict[str, Any]
        ] = []

        self.entity_embeddings = None

        self.refresh()

    def _load_entities(
        self,
    ) -> list[dict[str, Any]]:
        """
        Load semantic Entity nodes from Neo4j.
        """

        cypher = """
        MATCH (e:Entity)

        WHERE e.name IS NOT NULL

        RETURN
            e.entity_key AS entity_key,
            e.name AS name,
            e.type AS type,
            e.description AS description

        ORDER BY e.name
        """

        with self.store.driver.session(
            database=self.store.database
        ) as session:

            records = session.run(
                cypher
            )

            return [
                dict(record)
                for record in records
            ]

    @staticmethod
    def _entity_text(
        entity: dict[str, Any],
    ) -> str:
        """
        Build the text representation that will be embedded
        for an entity.

        We include type and description when available so
        matching is not based only on the entity name.
        """

        parts = []

        name = entity.get("name")

        entity_type = entity.get("type")

        description = entity.get(
            "description"
        )

        if name:
            parts.append(
                f"Name: {name}"
            )

        if entity_type:
            parts.append(
                f"Type: {entity_type}"
            )

        if description:
            parts.append(
                f"Description: {description}"
            )

        return "\n".join(parts)

    def refresh(
        self,
    ) -> None:
        """
        Reload entities from Neo4j and rebuild the in-memory
        entity embedding matrix.

        Call this after the knowledge graph has been updated.
        """

        self.entities = (
            self._load_entities()
        )

        if not self.entities:

            self.entity_embeddings = None

            return

        texts = [
            self._entity_text(entity)
            for entity in self.entities
        ]

        embeddings = (
            self.embedding_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        self.entity_embeddings = (
            np.asarray(
                embeddings,
                dtype="float32",
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Find Entity nodes semantically related to the query.

        This returns candidates only.

        The score threshold is deliberately conservative
        because later retrieval/reranking stages will decide
        which evidence is actually useful.
        """

        if not query.strip():
            return []

        if (
            not self.entities
            or self.entity_embeddings is None
        ):
            return []

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )[0]

        scores = (
            self.entity_embeddings
            @ query_embedding
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices:

            score = float(
                scores[index]
            )

            if score < min_score:
                continue

            entity = dict(
                self.entities[index]
            )

            entity[
                "semantic_score"
            ] = score

            results.append(
                entity
            )

            if len(results) >= top_k:
                break

        return results

    def close(
        self,
    ) -> None:

        if self._owns_store:
            self.store.close()
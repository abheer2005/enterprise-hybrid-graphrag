from typing import Any

from src.graph.neo4j_store import Neo4jStore
from src.retrieval.semantic_entity_linker import SemanticEntityLinker


class GraphRetriever:
    """
    Graph Retriever v2 for IOCL GraphRAG.

    Entity discovery uses two independent signals:

        1. Literal entity matching
        2. Semantic entity linking

    The resulting entity candidates are fused and
    deduplicated before graph traversal.

    No document names, departments, domains, policies,
    or business categories are hard-coded.
    """

    def __init__(
        self,
        embedding_model,
        store: Neo4jStore | None = None,
        semantic_linker: SemanticEntityLinker | None = None,
    ):
        self.store = store or Neo4jStore()
        self._owns_store = store is None

        self.semantic_linker = (
            semantic_linker
            or SemanticEntityLinker(
                embedding_model=embedding_model,
                store=self.store,
            )
        )

        self._owns_semantic_linker = (
            semantic_linker is None
        )

    # -----------------------------------------------------
    # 1. LITERAL ENTITY MATCHING
    # -----------------------------------------------------

    def search_entities_literal(
        self,
        user_query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        High-confidence literal entity matching.

        Useful when an entity name itself appears in the
        user question.

        Example:
            "Who do Senior Management Personnel report to?"
        """

        if not user_query.strip():
            return []

        cypher = """
        MATCH (e:Entity)

        WHERE e.name IS NOT NULL

        WITH
            e,
            toLower(e.name) AS entity_name,
            toLower($user_query) AS normalized_query

        WHERE
            normalized_query CONTAINS entity_name
            OR entity_name CONTAINS normalized_query

        RETURN
            e.entity_key AS entity_key,
            e.name AS name,
            e.type AS type,
            e.description AS description

        LIMIT $limit
        """

        with self.store.driver.session(
            database=self.store.database
        ) as session:

            records = session.run(
                cypher,
                user_query=user_query,
                limit=limit,
            )

            results = []

            for record in records:

                entity = dict(record)

                entity["literal_match"] = True

                results.append(entity)

            return results

    # -----------------------------------------------------
    # 2. SEMANTIC ENTITY MATCHING
    # -----------------------------------------------------

    def search_entities_semantic(
        self,
        user_query: str,
        limit: int = 10,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Semantic entity candidate generation.

        Handles paraphrases and wording differences.

        Example:

            Document:
                Senior Management Personnel

            Query:
                senior executives
        """

        if not user_query.strip():
            return []

        results = self.semantic_linker.search(
            query=user_query,
            top_k=limit,
            min_score=min_score,
        )

        for entity in results:
            entity["semantic_match"] = True

        return results

    # -----------------------------------------------------
    # 3. ENTITY FUSION
    # -----------------------------------------------------

    def search_entities(
        self,
        user_query: str,
        limit: int = 10,
        semantic_min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Fuse literal and semantic entity candidates.

        Literal matches receive priority because direct
        entity mentions are strong evidence.

        Semantic matches expand recall for paraphrases,
        synonyms, singular/plural differences, and related
        natural-language wording.

        Candidates are deduplicated using entity_key.
        """

        literal_results = (
            self.search_entities_literal(
                user_query=user_query,
                limit=limit,
            )
        )

        semantic_results = (
            self.search_entities_semantic(
                user_query=user_query,
                limit=limit,
                min_score=semantic_min_score,
            )
        )

        candidates: dict[
            str,
            dict[str, Any],
        ] = {}

        # ---------------------------------------------
        # Add literal matches
        # ---------------------------------------------

        for entity in literal_results:

            entity_key = entity.get(
                "entity_key"
            )

            if not entity_key:
                continue

            candidates[entity_key] = {
                **entity,
                "literal_match": True,
                "semantic_match": False,
                "semantic_score": None,
                "entity_link_sources": [
                    "literal"
                ],
            }

        # ---------------------------------------------
        # Merge semantic matches
        # ---------------------------------------------

        for entity in semantic_results:

            entity_key = entity.get(
                "entity_key"
            )

            if not entity_key:
                continue

            semantic_score = entity.get(
                "semantic_score"
            )

            if entity_key in candidates:

                candidate = (
                    candidates[entity_key]
                )

                candidate[
                    "semantic_match"
                ] = True

                candidate[
                    "semantic_score"
                ] = semantic_score

                if (
                    "semantic"
                    not in candidate[
                        "entity_link_sources"
                    ]
                ):
                    candidate[
                        "entity_link_sources"
                    ].append(
                        "semantic"
                    )

            else:

                candidates[entity_key] = {
                    **entity,
                    "literal_match": False,
                    "semantic_match": True,
                    "semantic_score": (
                        semantic_score
                    ),
                    "entity_link_sources": [
                        "semantic"
                    ],
                }

        # ---------------------------------------------
        # Entity-link score
        # ---------------------------------------------
        #
        # Literal mention = strongest signal.
        #
        # Semantic score remains available for
        # ranking non-literal candidates.
        #
        # A candidate supported by both receives
        # a small overlap bonus.
        # ---------------------------------------------

        fused_results = []

        for candidate in candidates.values():

            semantic_score = (
                candidate.get(
                    "semantic_score"
                )
                or 0.0
            )

            literal_match = (
                candidate.get(
                    "literal_match",
                    False,
                )
            )

            semantic_match = (
                candidate.get(
                    "semantic_match",
                    False,
                )
            )

            if literal_match:

                link_score = 1.0

                if semantic_match:
                    link_score += 0.05

            else:

                link_score = float(
                    semantic_score
                )

            candidate[
                "entity_link_score"
            ] = link_score

            fused_results.append(
                candidate
            )

        fused_results.sort(
            key=lambda item: (
                item[
                    "entity_link_score"
                ],
                item.get(
                    "semantic_score"
                )
                or 0.0,
            ),
            reverse=True,
        )

        return fused_results[:limit]

    # -----------------------------------------------------
    # 4. GRAPH NEIGHBORHOOD
    # -----------------------------------------------------

    def get_neighborhood(
        self,
        entity_key: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Retrieve semantic relationships touching an entity,
        together with documentary provenance.
        """

        cypher = """
        MATCH (e:Entity {
            entity_key: $entity_key
        })

        MATCH (e)-[r]-(other:Entity)

        WHERE type(r) <> 'MENTIONS'

        RETURN
            e.name AS matched_entity,

            startNode(r).name
                AS source_entity,

            type(r)
                AS relationship,

            endNode(r).name
                AS target_entity,

            other.name
                AS connected_entity,

            other.type
                AS connected_type,

            r.evidence
                AS evidence,

            r.chunk_id
                AS chunk_id,

            r.source_document
                AS source,

            r.page
                AS page

        LIMIT $limit
        """

        with self.store.driver.session(
            database=self.store.database
        ) as session:

            records = session.run(
                cypher,
                entity_key=entity_key,
                limit=limit,
            )

            return [
                dict(record)
                for record in records
            ]

    # -----------------------------------------------------
    # 5. PROVENANCE CHUNK
    # -----------------------------------------------------

    def get_chunk(
        self,
        chunk_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve original documentary evidence behind
        a graph fact.
        """

        cypher = """
        MATCH (c:Chunk {
            chunk_id: $chunk_id
        })

        OPTIONAL MATCH (
            d:Document
        )-[:CONTAINS]->(c)

        WITH c, d
        WHERE NOT coalesce(d.status, 'current') IN
            ['obsolete', 'superseded', 'revoked', 'expired']

        RETURN
            c.chunk_id AS chunk_id,
            c.text AS text,
            c.page AS page,
            d.source AS source,
            d.document_id AS document_id,
            d.version AS document_version,
            d.status AS status,
            d.relative_path AS relative_path

        LIMIT 1
        """

        with self.store.driver.session(
            database=self.store.database
        ) as session:

            record = session.run(
                cypher,
                chunk_id=chunk_id,
            ).single()

            if record is None:
                return None

            return dict(record)

    # -----------------------------------------------------
    # 6. FULL GRAPH RETRIEVAL
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        entity_limit: int = 10,
        relation_limit: int = 30,
        semantic_min_score: float = 0.25,
    ) -> dict[str, Any]:
        """
        Graph Retriever v2:

            question
                ↓
        literal entity linking
                +
        semantic entity linking
                ↓
           entity fusion
                ↓
         graph traversal
                ↓
        provenance chunks

        No LLM is used in this retrieval stage.
        """

        entities = self.search_entities(
            user_query=query,
            limit=entity_limit,
            semantic_min_score=semantic_min_score,
        )

        relationships = []

        seen_relationships = set()

        chunk_ids = set()

        for entity in entities:

            neighborhood = (
                self.get_neighborhood(
                    entity_key=entity[
                        "entity_key"
                    ],
                    limit=relation_limit,
                )
            )

            for relation in neighborhood:

                signature = (
                    relation.get(
                        "source_entity"
                    ),
                    relation.get(
                        "relationship"
                    ),
                    relation.get(
                        "target_entity"
                    ),
                    relation.get(
                        "chunk_id"
                    ),
                )

                if (
                    signature
                    in seen_relationships
                ):
                    continue

                seen_relationships.add(
                    signature
                )

                relationships.append(
                    relation
                )

                chunk_id = relation.get(
                    "chunk_id"
                )

                if chunk_id:
                    chunk_ids.add(
                        chunk_id
                    )

        chunks = []

        for chunk_id in chunk_ids:

            chunk = self.get_chunk(
                chunk_id
            )

            if chunk:
                chunks.append(
                    chunk
                )

        return {
            "query": query,
            "entities": entities,
            "relationships": relationships,
            "chunks": chunks,
        }

    # -----------------------------------------------------
    # 7. REFRESH
    # -----------------------------------------------------

    def refresh_entities(
        self,
    ) -> None:
        """
        Refresh semantic entity embeddings after the
        knowledge graph changes.

        Important when new documents/chunks/entities
        have been ingested.
        """

        self.semantic_linker.refresh()

    # -----------------------------------------------------
    # 8. CLOSE
    # -----------------------------------------------------

    def close(
        self,
    ) -> None:

        if (
            self._owns_semantic_linker
            and self.semantic_linker
        ):
            self.semantic_linker.close()

        if self._owns_store:
            self.store.close()

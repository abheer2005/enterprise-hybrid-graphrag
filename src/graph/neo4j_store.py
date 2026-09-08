import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Neo4jStore:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        self.database = os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        )

        if not all([uri, username, password]):
            raise ValueError(
                "Neo4j configuration missing from .env"
            )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def create_constraints(self):
        queries = [
            """
            CREATE CONSTRAINT document_id_unique
            IF NOT EXISTS
            FOR (d:Document)
            REQUIRE d.document_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT chunk_id_unique
            IF NOT EXISTS
            FOR (c:Chunk)
            REQUIRE c.chunk_id IS UNIQUE
            """,
        ]

        with self.driver.session(
            database=self.database
        ) as session:

            for query in queries:
                session.run(query).consume()

    def upsert_document_and_chunk(
        self,
        record: dict,
    ):
        metadata = record["metadata"]

        query = """
        MERGE (d:Document {document_id: $document_id})

        SET d.source = $source,
            d.file_type = $file_type,
            d.file_path = $file_path,
            d.relative_path = $relative_path,
            d.content_hash = $content_hash,
            d.version = $document_version,
            d.status = $status,
            d.effective_from = $effective_from,
            d.effective_to = $effective_to,
            d.supersedes = $supersedes

        MERGE (c:Chunk {chunk_id: $chunk_id})

        SET c.text = $text,
            c.page = $page,
            c.slide = $slide,
            c.sheet = $sheet,
            c.chunk_index = $chunk_index,
            c.character_count = $character_count

        MERGE (d)-[:CONTAINS]->(c)
        """

        parameters = {
            "source": metadata.get("source"),
            "document_id": metadata.get("document_id") or metadata.get("source"),
            "file_type": metadata.get("file_type"),
            "file_path": metadata.get("file_path"),
            "relative_path": metadata.get("relative_path"),
            "content_hash": metadata.get("content_hash"),
            "document_version": metadata.get("document_version"),
            "status": metadata.get("status", "current"),
            "effective_from": metadata.get("effective_from"),
            "effective_to": metadata.get("effective_to"),
            "supersedes": metadata.get("supersedes"),
            "chunk_id": record["chunk_id"],
            "text": record["text"],
            "page": metadata.get("page"),
            "slide": metadata.get("slide"),
            "sheet": metadata.get("sheet"),
            "chunk_index": metadata.get(
                "chunk_index"
            ),
            "character_count": metadata.get(
                "character_count"
            ),
        }

        with self.driver.session(
            database=self.database
        ) as session:

            session.run(
                query,
                **parameters,
            ).consume()

    def get_graph_stats(self) -> dict:
        query = """
        MATCH (n)
        WITH count(n) AS nodes

        MATCH ()-[r]->()
        RETURN
            nodes,
            count(r) AS relationships
        """

        with self.driver.session(
            database=self.database
        ) as session:

            record = session.run(query).single()

            return {
                "nodes": record["nodes"],
                "relationships": record[
                    "relationships"
                ],
            }
    def create_semantic_constraints(self):
        query = """
        CREATE CONSTRAINT entity_key_unique
        IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.entity_key IS UNIQUE
        """

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(query).consume()

    @staticmethod
    def make_entity_key(
        name: str,
        entity_type: str,
    ) -> str:
        """
        Conservative entity identity.

        We keep type in the key so entities with the same
        text but genuinely different semantic roles are not
        automatically collapsed.
        """

        normalized_name = " ".join(
            name.lower().split()
        )

        return (
            f"{entity_type.lower()}"
            f"::{normalized_name}"
        )

    def upsert_extraction(
        self,
        chunk: dict,
        extraction,
    ):
        """
        Persist validated entities and relationships while
        preserving source provenance.
        """

        metadata = chunk["metadata"]

        chunk_id = chunk["chunk_id"]
        source = metadata.get("source")
        page = metadata.get("page")

        entity_types = {
            entity.name.lower(): entity.type
            for entity in extraction.entities
        }

        with self.driver.session(
            database=self.database
        ) as session:

            # -----------------------------------------
            # ENTITIES + CHUNK MENTIONS
            # -----------------------------------------

            for entity in extraction.entities:

                entity_key = self.make_entity_key(
                    entity.name,
                    entity.type,
                )

                session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id})

                    MERGE (e:Entity {
                        entity_key: $entity_key
                    })

                    ON CREATE SET
                        e.name = $name,
                        e.type = $entity_type,
                        e.description = $description

                    ON MATCH SET
                        e.name = $name

                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk_id,
                    entity_key=entity_key,
                    name=entity.name,
                    entity_type=entity.type,
                    description=entity.description,
                ).consume()

            # -----------------------------------------
            # SEMANTIC RELATIONSHIPS
            # -----------------------------------------

            for relationship in extraction.relationships:

                source_type = entity_types.get(
                    relationship.source.lower()
                )

                target_type = entity_types.get(
                    relationship.target.lower()
                )

                if not source_type or not target_type:
                    continue

                source_key = self.make_entity_key(
                    relationship.source,
                    source_type,
                )

                target_key = self.make_entity_key(
                    relationship.target,
                    target_type,
                )

                relation = relationship.relation

                # Predicate has already been normalized by
                # KnowledgeExtractor to uppercase A-Z0-9_.
                if not relation:
                    continue

                query = f"""
                MATCH (source_entity:Entity {{
                    entity_key: $source_key
                }})

                MATCH (target_entity:Entity {{
                    entity_key: $target_key
                }})

                MERGE (
                    source_entity
                )-[r:{relation} {{
                    chunk_id: $chunk_id
                }}]->(
                    target_entity
                )

                SET
                    r.evidence = $evidence,
                    r.source_document = $source_document,
                    r.page = $page
                """

                session.run(
                    query,
                    source_key=source_key,
                    target_key=target_key,
                    chunk_id=chunk_id,
                    evidence=relationship.evidence,
                    source_document=source,
                    page=page,
                ).consume()

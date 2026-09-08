from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph.neo4j_store import Neo4jStore


def main():

    print("=" * 80)
    print("IOCL KNOWLEDGE GRAPH PROVENANCE TEST")
    print("=" * 80)

    store = Neo4jStore()

    try:

        # ----------------------------------------------------------
        # TEST 1
        # Entity -> Chunk -> Document provenance
        # ----------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("TEST 1: ENTITY PROVENANCE")
        print("=" * 80)

        entity_query = """
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)

        RETURN
            e.name AS entity,
            e.type AS entity_type,
            c.chunk_id AS chunk_id,
            c.page AS chunk_page,
            d.source AS document_source,
            d.name AS document_name

        LIMIT 20
        """

        with store.driver.session(
            database=store.database
        ) as session:

            results = list(
                session.run(entity_query)
            )

        if not results:

            print(
                "No entity provenance records found."
            )

        else:

            for index, record in enumerate(
                results,
                start=1,
            ):

                print("\n" + "-" * 80)

                print(
                    f"RESULT #{index}"
                )

                print(
                    f"Entity:        "
                    f"{record.get('entity')}"
                )

                print(
                    f"Entity type:   "
                    f"{record.get('entity_type')}"
                )

                print(
                    f"Chunk ID:      "
                    f"{record.get('chunk_id')}"
                )

                print(
                    f"Chunk page:    "
                    f"{record.get('chunk_page')}"
                )

                print(
                    f"Document src:  "
                    f"{record.get('document_source')}"
                )

                print(
                    f"Document name: "
                    f"{record.get('document_name')}"
                )

        # ----------------------------------------------------------
        # TEST 2
        # Semantic relationship provenance
        # ----------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("TEST 2: RELATIONSHIP PROVENANCE")
        print("=" * 80)

        relationship_query = """
        MATCH (a:Entity)-[r]->(b:Entity)

        WHERE type(r) <> 'MENTIONS'

        RETURN
            a.name AS source_entity,
            type(r) AS relationship,
            b.name AS target_entity,
            r.evidence AS evidence,
            r.chunk_id AS chunk_id,
            r.source_document AS source_document,
            r.page AS page

        LIMIT 30
        """

        with store.driver.session(
            database=store.database
        ) as session:

            results = list(
                session.run(
                    relationship_query
                )
            )

        if not results:

            print(
                "No semantic relationships found."
            )

        else:

            for index, record in enumerate(
                results,
                start=1,
            ):

                print("\n" + "-" * 80)

                print(
                    f"RELATIONSHIP #{index}"
                )

                print(
                    f"{record.get('source_entity')} "
                    f"--{record.get('relationship')}--> "
                    f"{record.get('target_entity')}"
                )

                print(
                    f"\nEvidence:"
                )

                print(
                    record.get('evidence')
                )

                print(
                    f"\nChunk ID: "
                    f"{record.get('chunk_id')}"
                )

                print(
                    f"Source:   "
                    f"{record.get('source_document')}"
                )

                print(
                    f"Page:     "
                    f"{record.get('page')}"
                )

        # ----------------------------------------------------------
        # TEST 3
        # Verify relationship provenance points to real Chunk nodes
        # ----------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("TEST 3: PROVENANCE INTEGRITY")
        print("=" * 80)

        integrity_query = """
        MATCH (a:Entity)-[r]->(b:Entity)

        WHERE
            type(r) <> 'MENTIONS'
            AND r.chunk_id IS NOT NULL

        OPTIONAL MATCH (c:Chunk {
            chunk_id: r.chunk_id
        })

        RETURN
            count(r) AS total_relationships,
            count(c) AS relationships_with_chunk,
            count(r) - count(c) AS missing_chunk_links
        """

        with store.driver.session(
            database=store.database
        ) as session:

            record = session.run(
                integrity_query
            ).single()

        if record:

            total = record.get(
                "total_relationships"
            )

            linked = record.get(
                "relationships_with_chunk"
            )

            missing = record.get(
                "missing_chunk_links"
            )

            print(
                f"\nSemantic relationships: "
                f"{total}"
            )

            print(
                f"Valid chunk provenance:  "
                f"{linked}"
            )

            print(
                f"Missing chunk provenance:"
                f" {missing}"
            )

            if missing == 0:

                print(
                    "\nPROVENANCE INTEGRITY: PASS"
                )

            else:

                print(
                    "\nPROVENANCE INTEGRITY: FAIL"
                )

                print(
                    "Some semantic relationships "
                    "cannot be traced to a Chunk."
                )

        print("\n")
        print("=" * 80)
        print("PROVENANCE TEST COMPLETE")
        print("=" * 80)

    finally:

        store.close()


if __name__ == "__main__":
    main()
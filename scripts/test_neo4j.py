from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


load_dotenv(PROJECT_ROOT / ".env")


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def main():
    print("=" * 70)
    print("IOCL NEO4J CONNECTION TEST")
    print("=" * 70)

    if not all(
        [
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
        ]
    ):
        raise ValueError(
            "Neo4j configuration is missing from .env"
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
        ),
    )

    try:
        driver.verify_connectivity()

        print("\nNeo4j connectivity: SUCCESS")

        with driver.session(
            database=NEO4J_DATABASE
        ) as session:

            result = session.run(
                """
                RETURN
                    'IOCL Knowledge Graph' AS project,
                    datetime() AS database_time
                """
            )

            record = result.single()

            print(f"Project:       {record['project']}")
            print(f"Database time: {record['database_time']}")

        print("\nPython is successfully connected to Neo4j.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
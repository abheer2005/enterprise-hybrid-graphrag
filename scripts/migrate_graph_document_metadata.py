import json
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.graph.neo4j_store import Neo4jStore


CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_documents() -> dict[str, dict]:
    documents: dict[str, dict] = {}

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            metadata = json.loads(line)["metadata"]
            source = metadata["source"]
            current = documents.get(source)

            if current and current["document_id"] != metadata["document_id"]:
                raise ValueError(f"Conflicting document IDs for source: {source}")

            documents[source] = metadata

    return documents


def main():
    documents = load_documents()
    store = Neo4jStore()

    try:
        with store.driver.session(database=store.database) as session:
            previous = session.run(
                "MATCH (d:Document) RETURN properties(d) AS document"
            ).data()
            backup_dir = PROJECT_ROOT / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_file = backup_dir / f"neo4j_documents_{timestamp}.json"
            backup_file.write_text(
                json.dumps(previous, indent=2, default=str),
                encoding="utf-8",
            )

            result = session.run(
                """
                UNWIND $documents AS item
                MATCH (d:Document {source: item.source})
                SET d.document_id = item.document_id,
                    d.file_type = item.file_type,
                    d.file_path = item.file_path,
                    d.relative_path = item.relative_path,
                    d.content_hash = item.content_hash,
                    d.version = item.document_version,
                    d.status = item.status,
                    d.effective_from = item.effective_from,
                    d.effective_to = item.effective_to,
                    d.supersedes = item.supersedes
                RETURN count(d) AS updated
                """,
                documents=[
                    {
                        "source": metadata.get("source"),
                        "document_id": metadata.get("document_id"),
                        "file_type": metadata.get("file_type"),
                        "file_path": metadata.get("file_path"),
                        "relative_path": metadata.get("relative_path"),
                        "content_hash": metadata.get("content_hash"),
                        "document_version": metadata.get("document_version"),
                        "status": metadata.get("status", "current"),
                        "effective_from": metadata.get("effective_from"),
                        "effective_to": metadata.get("effective_to"),
                        "supersedes": metadata.get("supersedes"),
                    }
                    for metadata in documents.values()
                ],
            ).single()

        updated = result["updated"]
        store.create_constraints()
        print(f"Backup: {backup_file}")
        print(f"Updated {updated}/{len(documents)} graph documents.")

        if updated != len(documents):
            raise RuntimeError(
                "Graph document count did not match processed corpus; "
                "run scripts/build_base_graph.py before serving queries."
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()

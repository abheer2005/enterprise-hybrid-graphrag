from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.document_loader import build_document_metadata
from src.ingestion.processor import merge_incremental_chunks
from src.retrieval.vector_store import VectorStore


def test_provenance_and_versioning() -> None:
    metadata = build_document_metadata(PROJECT_ROOT / "README.md", PROJECT_ROOT)
    assert metadata["relative_path"] == "README.md"
    assert metadata["document_id"]
    assert metadata["content_hash"]
    assert metadata["document_version"] == metadata["content_hash"][:12]
    assert metadata["status"] == "current"


def test_incremental_replacement() -> None:
    old = [{"chunk_id": "old", "text": "old", "metadata": {"document_id": "doc"}}]
    other = [{"chunk_id": "keep", "text": "keep", "metadata": {"document_id": "other"}}]
    new = [{"chunk_id": "new", "text": "new", "metadata": {"document_id": "doc"}}]
    merged = merge_incremental_chunks(old + other, new)
    assert {item["chunk_id"] for item in merged} == {"keep", "new"}


def test_inactive_vector_records_are_filtered() -> None:
    class Model:
        def encode(self, *_args, **_kwargs):
            import numpy as np
            return np.asarray([[1.0]], dtype="float32")

    class Index:
        ntotal = 2
        def search(self, *_args):
            import numpy as np
            return np.asarray([[0.9, 0.8]]), np.asarray([[0, 1]])

    store = VectorStore.__new__(VectorStore)
    store.model = Model()
    store.index = Index()
    store.records = [
        {"chunk_id": "revoked", "text": "x", "metadata": {"status": "revoked"}},
        {"chunk_id": "current", "text": "y", "metadata": {"status": "current"}},
    ]
    results = store.search("query", top_k=1)
    assert [item["chunk_id"] for item in results] == ["current"]


if __name__ == "__main__":
    test_provenance_and_versioning()
    test_incremental_replacement()
    test_inactive_vector_records_are_filtered()
    print("Enterprise requirement regressions passed.")

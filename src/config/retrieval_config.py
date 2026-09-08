from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievalConfig:
        VECTOR_TOP_K = 75
        GRAPH_ENTITY_LIMIT = 20
        GRAPH_RELATION_LIMIT = 75
        RERANK_TOP_K = 30
        FINAL_CONTEXT_K = 20
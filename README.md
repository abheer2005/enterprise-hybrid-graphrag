# IOCL Enterprise GraphRAG

An enterprise document question-answering system designed to retrieve evidence from organizational documents, generate evidence-grounded responses, validate citations, and verify factual claims before accepting an answer.

## Overview

The system combines semantic vector retrieval and knowledge-graph retrieval to provide grounded question answering over enterprise documents.

The pipeline follows:

```text
Enterprise Documents
        |
        v
Document Ingestion
        |
        v
Chunking + Metadata
        |
        +--------------------+
        |                    |
        v                    v
Vector Index          Knowledge Graph
   FAISS                  Neo4j
        |                    |
        +---------+----------+
                  |
                  v
           Hybrid Retrieval
                  |
                  v
              Reranking
                  |
                  v
            Context Builder
                  |
                  v
          Gemini Generation
                  |
                  v
        Citation Validation
                  |
                  v
        NLI Claim Grounding
                  |
             +----+----+
             |         |
           PASS       FAIL
             |         |
             |      Repair
             |         |
             +----+----+
                  |
                  v
          Verified Response
```

## Core Features

- Multi-format enterprise document ingestion
- PDF, DOCX, PPTX and XLSX support
- Metadata-aware document chunking
- FAISS semantic vector retrieval
- Neo4j knowledge graph
- Semantic entity linking
- Hybrid graph + vector retrieval
- Candidate reranking
- Context construction
- Evidence-grounded Gemini answer generation
- Inline evidence citations
- Citation validation
- NLI-based claim grounding
- Answer repair
- Insufficient-evidence handling
- Provider/API failure handling with bounded retries
- FastAPI REST API
- Responsive browser interface
- Pipeline performance metrics

## Technology Stack

### AI and Retrieval

- Google Gemini
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- FAISS

### Knowledge Graph

- Neo4j

### API

- FastAPI
- Uvicorn
- Pydantic

### Document Processing

- PyMuPDF
- python-docx
- python-pptx
- openpyxl

## Project Structure

```text
Knowledge Graph/
|
|-- data/
|
|-- scripts/
|   |-- build_base_graph.py
|   |-- build_knowledge_graph.py
|   |-- build_vector_index.py
|   |-- process_corpus.py
|   `-- test_*.py
|
|-- src/
|   |
|   |-- api/
|   |   |-- app.py
|   |   `-- index.html
|   |
|   |-- generation/
|   |   |-- answer_generator.py
|   |   |-- answer_repairer.py
|   |   |-- answer_validator.py
|   |   |-- context_builder.py
|   |   |-- grounding_validator.py
|   |   `-- qa_pipeline.py
|   |
|   |-- graph/
|   |   |-- entity_extractor.py
|   |   |-- extraction_cache.py
|   |   `-- neo4j_store.py
|   |
|   |-- ingestion/
|   |   |-- chunker.py
|   |   |-- document_loader.py
|   |   `-- processor.py
|   |
|   `-- retrieval/
|       |-- graph_retriever.py
|       |-- hybrid_retriever.py
|       |-- relevance_gate.py
|       |-- reranker.py
|       |-- semantic_entity_linker.py
|       `-- vector_store.py
|
|-- requirements.txt
|-- .env
|-- .gitignore
`-- README.md
```

## Environment Setup

Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root.

Required configuration includes the Gemini API key and Neo4j connection information used by the application.

Example:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.6-flash

NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

Never commit `.env` or production credentials to source control.

## Running the API

From the project root with the virtual environment activated:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

The application initializes the retrieval and grounding models during startup.

Open the application at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
GET /health
```

Question-answering endpoint:

```text
POST /ask
```

## Verification Philosophy

The system does not treat successful LLM generation as sufficient for answer acceptance.

Generated answers pass through additional verification stages including citation validation and NLI-based claim grounding.

A response may therefore be generated but rejected if its factual claims cannot be sufficiently supported by the retrieved evidence.

The system also distinguishes between:

- insufficient document evidence,
- an unverified/generated answer,
- provider or generation failure,
- and a successfully verified response.

## Reliability

Temporary generation-provider failures use bounded retry handling.

Non-retryable failures are not repeatedly sent to the provider.

If generation cannot be completed reliably, the QA pipeline returns a safe unaccepted result rather than allowing the complete application request to crash.

## Performance

Pipeline timing information is recorded for major stages including:

- retrieval
- reranking
- context building
- generation
- citation validation
- NLI grounding
- answer repair

Development testing identified local CPU NLI inference as the primary latency bottleneck.

This is a deployment-performance consideration rather than a reason to remove claim verification. Production environments may use accelerated inference, suitable hardware, caching, batching, or other deployment optimizations while retaining the verification architecture.

## Security Notes

- API keys and database credentials are loaded from environment variables.
- `.env` is excluded through `.gitignore`.
- Credentials should never be embedded in application source code.
- Internal provider errors should be logged server-side rather than exposed to normal end users.
- Production deployment should add organization-appropriate authentication, authorization, network controls and secret management.

## Validation

The project contains tests covering major components of the system, including:

- ingestion
- chunking
- vector search
- graph retrieval
- semantic entity linking
- hybrid retrieval
- relevance gating
- context building
- answer generation
- citation validation
- claim grounding
- answer repair
- QA failure paths
- end-to-end QA evaluation

## Deployment Considerations

Before organization-wide production deployment, infrastructure-specific decisions should be made for:

- authentication and authorization
- TLS/HTTPS
- enterprise secret management
- request rate limiting
- audit logging
- monitoring and observability
- model hosting/inference hardware
- backup and recovery
- Neo4j production configuration
- document-access permissions
- horizontal scaling
- CI/CD

These controls depend on the target enterprise infrastructure and are intentionally separate from the core GraphRAG implementation.

## Status

**Version:** 1.0.0

The v1 system implements the complete enterprise GraphRAG question-answering workflow from document ingestion and hybrid retrieval through grounded generation, citation validation and claim-level verification.
# IOCL Enterprise Hybrid GraphRAG

This backend exposes one enterprise assistant over a persistent document
knowledge base. Routing is implicit: semantic vector retrieval and knowledge-
graph entity linking infer the relevant documents from each question. No
department names or assistant selection rules are hard-coded.

## Production data paths

- `data/raw/`: permanent managed corpus. `scripts/process_corpus.py` performs a
  snapshot update, skips unchanged files, replaces changed versions, and removes
  documents no longer present.
- `data/processed/`: chunk records and the persistent vector index. The vector
  builder reuses embeddings for unchanged chunk IDs.
- Neo4j: document, chunk, entity, relationship, and provenance graph.
- `POST /v1/temporary-file`: isolated ad-hoc analysis. Its bytes are deleted at
  request completion and never enter the permanent indexes or graph.

Each permanent source receives a stable `document_id`, relative source URI,
SHA-256 content fingerprint, version, lifecycle status, timestamps, and exact
page/slide/sheet/chunk provenance. Optional governance metadata belongs in a
sidecar named `document.ext.metadata.json`, for example:

```json
{
  "document_version": "2026.2",
  "status": "current",
  "effective_from": "2026-04-01",
  "effective_to": null,
  "supersedes": "previous-document-id",
  "owner": "document owner",
  "classification": "internal"
}
```

Inactive statuses (`obsolete`, `superseded`, `revoked`, `expired`) are excluded
from normal vector and graph retrieval. This is metadata-driven and remains
generic across future domains.

## API integration

Use `POST /v1/ask` from IOCL's existing UI:

```json
{"query": "What procedure applies when ...?"}
```

Retrieval limits and acceptance policy are server-controlled on this endpoint.
The legacy `/ask` endpoint remains available for backward compatibility and
testing. Answers are accepted only after citation validation and claim-level NLI
grounding (or safe removal of unsupported claims); otherwise the service returns
an evidence-insufficiency or non-acceptance response.

For a temporary file, send the raw file bytes to `POST /v1/temporary-file` with
`X-Filename` and optional `X-Instruction` headers. The default size limit is
25 MiB and can be changed with `TEMP_UPLOAD_MAX_BYTES`.

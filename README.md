Enterprise Hybrid GraphRAG

An enterprise-grade Hybrid GraphRAG question-answering system that combines semantic vector search, knowledge-graph retrieval, cross-encoder reranking, grounded LLM generation, citation validation, and NLI-based factual verification.

Built as part of an enterprise AI/ML internship project to improve document question answering beyond naïve RAG.

Overview

Traditional RAG systems often retrieve semantically similar chunks but can miss relationships between entities, policies, roles, and processes spread across multiple documents.

This project addresses that limitation with a Hybrid GraphRAG pipeline that combines:

FAISS semantic retrieval

Neo4j Knowledge Graph retrieval

Cross-encoder reranking

Relevance gating

Evidence selection and context building

Grounded LLM generation

Source/page citations

NLI-based claim verification

Deterministic answer repair and acceptance checks

The goal is a single enterprise assistant that can automatically identify relevant evidence across heterogeneous documents without requiring users to manually select a department, assistant, or filename.

Key Features

Hybrid Retrieval

Uses both vector and graph retrieval to capture:

semantic similarity

entity relationships

cross-document connections

policy/process dependencies

Knowledge Graph

Stores extracted entities and relationships in Neo4j, enabling graph-aware retrieval alongside semantic search.

Semantic Search

Uses multilingual Sentence Transformer embeddings with FAISS for efficient document retrieval.

Cross-Encoder Reranking

Retrieved candidates are reranked using a cross-encoder to improve query-document relevance.

Relevance Gate

Rejects weak or unsupported retrieval results before answer generation.

Grounded Generation

The LLM is instructed to answer only from retrieved enterprise evidence.

Citation Validation

Generated citations are checked against retrieved evidence and document provenance.

NLI Grounding

A DeBERTa NLI model validates whether generated factual claims are:

entailed

neutral

contradicted

Unsupported content can be removed before the final response is accepted.

Abstention

If sufficient evidence is not available, the system can return an insufficient evidence response instead of hallucinating.

OpenAI-Compatible API

The backend exposes OpenAI-compatible endpoints so it can integrate with interfaces such as Open WebUI.

System Architecture

User / Open WebUI
        |
        v
FastAPI Backend
        |
        v
Query
        |
        +-------------------+
        |                   |
        v                   v
FAISS Vector Search     Neo4j Graph Search
        |                   |
        +---------+---------+
                  |
                  v
          Hybrid Candidate Set
                  |
                  v
        Cross-Encoder Reranking
                  |
                  v
            Relevance Gate
                  |
                  v
          Evidence Selection
                  |
                  v
           Context Builder
                  |
                  v
         LLM Generation Layer
                  |
                  v
         Citation Validation
                  |
                  v
       DeBERTa NLI Verification
                  |
                  v
      Repair / Acceptance Gate
                  |
                  v
          Final Grounded Answer

Tech Stack

Layer

Technology

Language

Python

API

FastAPI, Uvicorn

Vector Search

FAISS

Knowledge Graph

Neo4j

Embeddings

Sentence Transformers

Reranking

Cross-Encoder

LLM Integration

Ollama-compatible / configurable provider

Grounding

DeBERTa NLI

Inference Runtime

ONNX Runtime

Validation

Deterministic citation + grounding checks

Repository Structure

enterprise-hybrid-graphrag/
├── src/
│   ├── api/
│   ├── config/
│   ├── generation/
│   ├── graph/
│   ├── ingestion/
│   └── retrieval/
├── scripts/
├── tests/
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

End-to-End Pipeline

1. Document Ingestion

Documents
   ↓
Parsing / Cleaning
   ↓
Chunking
   ↓
Metadata + Provenance
   ↓
Embeddings → FAISS
   ↓
Entity / Relationship Extraction → Neo4j

2. Query Processing

Question
   ↓
Vector Retrieval + Graph Retrieval
   ↓
Hybrid Fusion
   ↓
Cross-Encoder Reranking
   ↓
Relevance Filtering

3. Answer Generation

Selected Evidence
   ↓
Context Builder
   ↓
LLM
   ↓
Grounded Answer + Citations

4. Verification

Generated Answer
   ↓
Citation Validation
   ↓
Claim Extraction
   ↓
NLI Grounding
   ↓
Repair if Required
   ↓
Final Acceptance / Abstention

Example Capabilities

The system is designed to handle:

direct factual questions

paraphrased questions

policy and compliance queries

cross-document reasoning

process and approval-chain questions

enterprise document summaries

unsupported/out-of-domain questions with abstention

Example:

Question:
How are whistleblowers protected?

System:
→ retrieves relevant policy evidence
→ reranks the strongest passages
→ generates an evidence-constrained answer
→ attaches source/page citations
→ verifies factual claims using NLI
→ returns the final grounded response

Open WebUI Integration

The GraphRAG backend can be exposed as an OpenAI-compatible provider.

Main endpoints:

GET  /health
GET  /v1/models
POST /v1/chat/completions

Typical deployment flow:

Open WebUI
    ↓
GraphRAG FastAPI
    ↓
FAISS + Neo4j
    ↓
LLM via Ollama
    ↓
Citation + NLI Validation
    ↓
Open WebUI

This allows the retrieval and grounding layer to remain independent from the user interface.

Environment Configuration

Create a .env file from .env.example.

Example:

OLLAMA_BASE_URL=http://<ollama-server>:11434
OLLAMA_MODEL=<model-name>

NEO4J_URI=bolt://<neo4j-server>:7687
NEO4J_USER=<username>
NEO4J_PASSWORD=<password>

DOCUMENT_ROOT=/path/to/documents

Never commit real credentials, internal server addresses, or confidential enterprise documents.

Installation

git clone https://github.com/abheer2005/enterprise-hybrid-graphrag.git
cd enterprise-hybrid-graphrag

python -m venv .venv

Windows:

.venv\Scripts\activate

Linux/macOS:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the API

python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001

Health check:

GET http://localhost:8001/health

Engineering Highlights

Hybrid vector + graph retrieval instead of naïve RAG

Cross-encoder reranking for stronger relevance

Provenance-aware evidence handling

Source/page-level citation support

NLI-based hallucination detection

Deterministic answer repair

Unsupported-query abstention

Persistent vector and graph indexes

API-first architecture

OpenAI-compatible integration layer

Designed for heterogeneous enterprise documents

Why Hybrid GraphRAG?

A pure vector RAG system is effective for semantic similarity but may struggle when answers depend on relationships such as:

Policy → Approval Authority
Role → Responsibility
Process → Requirement
Document → Regulation
Entity → Related Entity

A knowledge graph provides relationship-aware retrieval, while vector search provides semantic flexibility.

Combining both helps create a stronger enterprise retrieval system.

Future Improvements

incremental document ingestion

document version/conflict handling

graph entity normalization

improved graph candidate filtering

asynchronous ingestion jobs

GPU-accelerated NLI

streaming chat responses

authentication and role-based access control

monitoring and observability

retrieval/evaluation dashboards

Security & Confidentiality

This repository contains the application architecture and implementation only.

It should not contain:

confidential enterprise documents

real API keys

passwords

internal server credentials

proprietary production data

private vector indexes built from confidential documents

Use .env.example for configuration placeholders and keep the actual .env private.

Author

Abheer Agarwal

B.Tech Computer Science & Engineering
Focused on AI/ML, LLMs, RAG, GraphRAG, Knowledge Graphs and enterprise AI systems.

GitHub: abheer2005

Disclaimer

This repository represents an engineering project/prototype and should not be interpreted as an official public deployment or product of any organization. Enterprise deployment details, credentials, and proprietary documents are intentionally excluded
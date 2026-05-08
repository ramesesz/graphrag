# GraphRAG — Knowledge Graph Extraction & Chat Pipeline

A config-driven pipeline that ingests documents (PDF/HTML), extracts entities and relationships using GPT-4o, stores them in a Neo4j knowledge graph, and answers natural-language questions via a Streamlit chat interface backed by a three-tier retrieval strategy.

## System Architecture

Three services orchestrated via Docker Compose:

- **Neo4j** — Graph database storing nodes, relationships, and vector embeddings
- **Processor** — FastAPI service that handles document ingestion, LLM extraction, and embedding
- **Frontend** — Streamlit chat app with domain selection, graph visualization, and job monitoring

Domains are defined as YAML config files — each domain specifies its own node types, relationship types, chunking strategy, and LLM prompts.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI API key (GPT-4o for extraction, GPT-4o-mini for chat)

### Setup

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
# Set OPENAI_API_KEY and NEO4J_PASSWORD in .env
```

Start all services:

```bash
docker-compose up -d
```

| Service | URL |
|---------|-----|
| Chat frontend | http://localhost:8501 |
| Processor API | http://localhost:8000 |
| Neo4j Browser | http://localhost:7474 |

Neo4j credentials: `neo4j` / `password123`

## Domains

Domains are configured in `configs/domains/`. Two domains are included:

| Domain | Config file | Description |
|--------|-------------|-------------|
| Deutsches Verkehrsrecht (StVO/StVZO) | `stvo_stvozo.yaml` | German traffic law — paragraphs, rules, violations, vehicle categories |
| Fantasy LitRPG (Azarinth Healer) | `fantasy_litrpg.yaml` | Characters, skills, monsters, locations from a LitRPG book series |

To add a new domain, create a YAML file in `configs/domains/` following the existing structure.

## Processing Documents

### Step 1 — Upload

Upload PDFs or HTML files via the sidebar in the frontend, or place them directly in `data/<domain_id>/documents/`.

### Step 2 — Extract

Trigger extraction from the frontend sidebar. Four modes are available:

| Mode | What it does |
|------|-------------|
| `full` | File → chunks → LLM extraction → graph JSON → Neo4j + embeddings |
| `chunks` | File → text chunks JSON only (no LLM cost) |
| `extract` | Existing chunks JSON → LLM extraction → graph JSON |
| `neo4j` | Existing graph JSON → Neo4j + embeddings only |

Monitor job progress in the sidebar's **Jobs** panel.

### Step 3 — Chat

Select a domain in the sidebar and ask questions in the chat interface. Each answer includes:

- The LLM response grounded strictly in graph data
- An interactive graph visualization showing which nodes and edges were used

## Retrieval Strategy

Every question goes through a three-tier fallback:

1. **Vector search** — embeds the question and finds semantically similar nodes via `node_embeddings` index
2. **Fulltext BM25** — extracts named entities from the question and searches the `node_search` Lucene index
3. **CONTAINS** — substring match on `node.id` as a last resort (no index required)

Tier 2 and 3 only run if the previous tier returns no results.

## Project Structure

```
.
├── docker-compose.yml
├── configs/
│   └── domains/                    # Domain YAML configs
│       ├── stvo_stvozo.yaml
│       └── fantasy_litrpg.yaml
├── data/                           # Document input and extraction output
│   ├── <domain_id>/documents/      # [INPUT] Drop PDFs/HTML here
│   ├── output_json/                # Extracted graph JSON
│   └── chunks/                     # Intermediate chunk JSON
└── services/
    ├── neo4j/
    │   └── init/                   # Cypher constraints and indexes
    ├── processor/                  # FastAPI extraction service
    │   ├── api.py
    │   ├── pipeline/
    │   │   ├── config_loader.py
    │   │   └── ...
    │   └── Dockerfile
    └── frontend/                   # Streamlit chat app
        ├── app.py
        └── Dockerfile
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Vector search returns nothing | The `node_embeddings` index is created after the first `full` or `neo4j` extraction — run at least one document through first |
| Fulltext search fails | The `node_search` index is created by `services/neo4j/init/constraints.cypher` on startup — check Neo4j logs |
| Processor API unreachable | Run `docker-compose logs processor` — likely a missing `.env` variable |
| OpenAI rate limit errors | Reduce `batch_size` in the domain YAML config |

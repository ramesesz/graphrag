# Local GraphRAG Extraction Pipeline

This project implements a fully local Graph Retrieval-Augmented Generation (GraphRAG) pipeline. It ingests unstructured documents (PDFs), extracts entities and relationships using a local LLM (Llama 3.1 via Ollama), and constructs a Knowledge Graph in Neo4j.

## 🏗 System Architecture

The system runs entirely in Docker and consists of four main services:

- **Neo4j**: Graph Database to store nodes and relationships
- **Ollama**: Local Inference Server hosting the Llama 3.1 model
- **Processor**: A batch worker that converts PDFs → Text Chunks → Graph Data → Neo4j
- **Jupyter**: An interactive notebook environment for testing and prototyping (The Playground)

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- Hardware: At least 16GB RAM recommended (running an 8B model)

### Setup (First Run Only)

Start the stack in the background:

```bash
docker-compose up -d
```

> **⚠️ IMPORTANT**: You must download the model into the Ollama container before the processor can use it. Run this command and wait for the download (~4.7GB) to finish:
>
> ```bash
> docker exec -it ollama ollama pull llama3.1
> ```

### Usage: How to Process Documents

#### Step A: Add Documents

Place your PDF files into the local data folder: `./data/documents/` 

*(Example: Copy GameOfThrones.pdf into this folder)*

#### Step B: Trigger Extraction

The processor service runs as a batch script. To trigger the ingestion of all files in the input folder:

```bash
docker-compose restart processor
```

Alternatively, if it is stopped:

```bash
docker-compose up processor
```

You can follow the progress (chunking, extraction, loading) by watching the logs:

```bash
docker-compose logs -f processor
```

### Visualization

Once the processor finishes:

1. Open the Neo4j Browser at http://localhost:7474
2. Login with:
   - **Username**: `neo4j`
   - **Password**: `password123` (or as defined in your compose file)
3. Run a query:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```
## 📂 Project Structure

```
.
├── docker-compose.yml              # Orchestration
├── .gitignore                       # Ignores large data files
├── data/
│   ├── documents/                  # [INPUT] Drop PDFs here
│   └── extracted_json/             # [OUTPUT] Intermediate JSON triples saved here
├── notebooks/                       # Jupyter notebooks for testing
└── services/
    ├── processor/                  # Python logic for extraction
    │   ├── main.py
    │   ├── requirements.txt
    │   └── Dockerfile
    └── neo4j/                       # Database configuration
```
## 🔬 Development & Testing

A Jupyter Lab environment is included to test code snippets interactively.

1. Ensure the container is running:

```bash
docker-compose up -d jupyter
```

2. Get the login token from the logs:

```bash
docker logs rag-notebook
```

3. Open the URL (e.g., `http://127.0.0.1:8888/?token=...`) in your browser.

## 🛠 Applications (Coming Soon)

The current pipeline handles the ETL (Extract, Transform, Load) phase. The following applications will be built on top of the populated graph:

### 🤖 Chatbot (Planned)

A conversational interface (RAG) that allows users to query the documents using:

- **Vector Search**: To find relevant text chunks
- **Graph Traversal**: To find hidden connections between entities (2-3 hops)
- **Context Synthesis**: Generating answers using Llama 3.1 based on retrieved graph context

### 📊 Graph Analysis (Planned)

Tools for advanced network analysis, including:

- **Centrality Algorithms**: Identifying key characters or entities (PageRank, Betweenness)
- **Community Detection**: Finding clusters or factions within the data (Louvain)
- **Pathfinding**: Analyzing shortest paths between two disconnected nodes

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" error | If the processor fails immediately, you likely skipped the model pull step. Run: `docker exec -it ollama ollama pull llama3.1` |
| Performance is slow | The extraction process runs on your CPU/iGPU. Large books can take several minutes. |
| Check CPU usage | Run: `docker stats` |
| Slow performance | Ensure Docker has access to sufficient resources (especially on Mac/Windows). |
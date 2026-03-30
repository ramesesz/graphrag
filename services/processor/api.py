"""
GraphRAG Processor REST API

Replaces the one-shot container startup with a persistent FastAPI service.
Documents can be uploaded and processing triggered dynamically from the UI.

Endpoints:
  POST /process?domain=&mode=    Trigger pipeline as background task
  POST /upload/{domain}          Upload a PDF to a domain's input directory
  GET  /jobs/{job_id}            Check job status
  GET  /jobs                     List all jobs
  DELETE /graph/{domain}         Clear all graph data for a domain from Neo4j
  GET  /domains                  List available domain configs
  GET  /health                   Health check
"""
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from pipeline.config_loader import list_domains, load_domain_config
from pipeline.embedder import LocalEmbedder, ensure_vector_index

# Module-level embedder — loaded once at startup, shared across requests
_embedder: LocalEmbedder | None = None

APP_DIR = Path("/app") if Path("/app").exists() else Path.cwd()
DATA_DIR = APP_DIR / "data"
INPUT_DIR = DATA_DIR / "input_docs"

app = FastAPI(
    title="GraphRAG Processor API",
    description="Upload documents and trigger knowledge graph extraction",
    version="1.0.0",
)


@app.on_event("startup")
def create_schema():
    """Load the embedding model and create Neo4j indexes on startup."""
    global _embedder
    _embedder = LocalEmbedder()

    graph = _connect_neo4j()
    if graph is None:
        logger.warning("Could not create schema — Neo4j not available at startup")
        return

    index_cypher = """
    CREATE FULLTEXT INDEX node_search IF NOT EXISTS
    FOR (n:Person|Character|House|Location|Skill|Class|Monster|Item|Organization|
         Paragraph|Rule|Violation|Prohibition|Requirement|Exception|Fine|Penalty|
         VehicleCategory|PersonRole|RoadType|Permit|Authority|Definition|
         Battle|Alliance|Faction|Title|Weapon|Event)
    ON EACH [n.id, n.description, n.text_excerpt, n.paragraph_number, n.paragraph_title]
    """
    try:
        graph.query(index_cypher)
        logger.info("Fulltext index 'node_search' ensured")
    except Exception as e:
        logger.warning("Could not create fulltext index (may already exist): %s", e)

    ensure_vector_index(graph)

# In-memory job registry (replace with Redis for multi-instance deployments)
jobs: Dict[str, dict] = {}


def _connect_neo4j():
    """Connect to Neo4j. Returns Neo4jGraph or None."""
    from langchain_neo4j import Neo4jGraph
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    max_retries = 30
    for i in range(max_retries):
        try:
            g = Neo4jGraph(url=neo4j_uri, username="neo4j", password=neo4j_password)
            g.refresh_schema()
            return g
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2)
            else:
                logger.error("Could not connect to Neo4j: %s", e)
                return None


def _run_pipeline(job_id: str, domain: str, mode: str) -> None:
    """Background task: run the extraction pipeline for a domain."""
    from main import process_document
    embedder = _embedder

    jobs[job_id]["status"] = "running"
    logger.info("Job %s starting: domain=%s mode=%s", job_id, domain, mode)

    try:
        config = load_domain_config(domain)

        graph = None
        if mode in ("full", "graph"):
            graph = _connect_neo4j()
            if graph is None:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Could not connect to Neo4j"
                return

        import glob
        domain_input_dir = INPUT_DIR / domain
        search_dir = domain_input_dir if domain_input_dir.exists() else INPUT_DIR
        input_files = (
            glob.glob(str(search_dir / "*.pdf")) +
            glob.glob(str(search_dir / "*.html"))
        )

        if not input_files:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = f"No PDF or HTML files found in data/input_docs/{domain}/"
            return

        jobs[job_id]["total_files"] = len(input_files)
        jobs[job_id]["processed_files"] = 0

        for input_file in input_files:
            process_document(input_file, mode=mode, config=config, embedder=embedder)
            jobs[job_id]["processed_files"] += 1

        jobs[job_id]["status"] = "completed"
        logger.info("Job %s completed", job_id)

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed_text(body: dict):
    """Return the embedding vector for a given text string."""
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required")
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not yet loaded")
    vector = _embedder.embed_query(text)
    return {"embedding": vector}


@app.get("/domains")
def get_domains():
    """List all available domain configs."""
    available = list_domains()
    result = []
    for d in available:
        try:
            cfg = load_domain_config(d)
            result.append({"id": d, "display_name": cfg.display_name})
        except Exception:
            result.append({"id": d, "display_name": d})
    return result


@app.post("/process")
def trigger_processing(
    domain: str,
    mode: Literal["full", "chunks", "graph"] = "full",
    background_tasks: BackgroundTasks = None,
):
    """Trigger the extraction pipeline for a domain as a background job."""
    available = list_domains()
    if domain not in available:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'. Available: {available}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "domain": domain,
        "mode": mode,
    }
    background_tasks.add_task(_run_pipeline, job_id, domain, mode)
    logger.info("Queued job %s: domain=%s mode=%s", job_id, domain, mode)
    return {"job_id": job_id, "status": "queued", "domain": domain, "mode": mode}


@app.post("/upload/{domain}")
async def upload_document(domain: str, file: UploadFile = File(...)):
    """Upload a PDF to a domain's input directory."""
    available = list_domains()
    if domain not in available:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'. Available: {available}")

    if not file.filename.lower().endswith((".pdf", ".html")):
        raise HTTPException(status_code=400, detail="Only PDF and HTML files are supported.")

    dest_dir = INPUT_DIR / domain
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    logger.info("Uploaded %s to %s", file.filename, dest)
    return {"filename": file.filename, "domain": domain, "path": str(dest), "size_bytes": len(content)}


@app.get("/jobs")
def list_jobs():
    """List all jobs."""
    return list(jobs.values())


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return jobs[job_id]


@app.delete("/graph/{domain}")
def clear_domain_graph(domain: str):
    """Remove all nodes tagged with a domain label from Neo4j."""
    graph = _connect_neo4j()
    if graph is None:
        raise HTTPException(status_code=503, detail="Could not connect to Neo4j")

    available = list_domains()
    if domain not in available:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{domain}'.")

    try:
        config = load_domain_config(domain)
        node_labels = config.allowed_nodes
        deleted_counts = {}
        for label in node_labels:
            result = graph.query(f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS deleted")
            deleted_counts[label] = result[0]["deleted"] if result else 0
        logger.info("Cleared graph for domain %s: %s", domain, deleted_counts)
        return {"domain": domain, "deleted_by_label": deleted_counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

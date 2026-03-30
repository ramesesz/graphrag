"""
One-time backfill script — embeds all existing Neo4j nodes so the vector
similarity index is populated for graphs extracted before embedding support
was added to the pipeline.

Usage (inside the processor container):
    python embed_nodes.py

Or via docker exec:
    docker exec graph-processor python embed_nodes.py
"""
import logging
import os
import time

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

from langchain_neo4j import Neo4jGraph

from pipeline.embedder import (
    EMBEDDING_MODEL,
    LocalEmbedder,
    embed_all_nodes,
    ensure_vector_index,
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def connect_neo4j(max_retries: int = 10) -> Neo4jGraph:
    for i in range(max_retries):
        try:
            g = Neo4jGraph(url=NEO4J_URI, username="neo4j", password=NEO4J_PASSWORD)
            g.refresh_schema()
            logger.info("Connected to Neo4j")
            return g
        except Exception as e:
            if i < max_retries - 1:
                logger.warning("Neo4j not ready, retrying in 2s... (%s)", e)
                time.sleep(2)
            else:
                raise RuntimeError(f"Could not connect to Neo4j after {max_retries} attempts") from e


if __name__ == "__main__":
    logger.info("Connecting to Neo4j at %s", NEO4J_URI)
    graph = connect_neo4j()

    embedder = LocalEmbedder()

    ensure_vector_index(graph)
    embed_all_nodes(graph, embedder)

    logger.info("Done.")

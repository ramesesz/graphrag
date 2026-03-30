"""
Node embedding utilities.

Builds rich text representations of Neo4j nodes (id + type + properties +
relationship context) and stores OpenAI embeddings back as `n.embedding` for
vector similarity search.

All nodes are also tagged with :__KGNode__ so a single vector index covers the
entire graph regardless of domain-specific labels.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384
BASE_LABEL = "__KGNode__"

# Cypher to fetch nodes with their relationship context
_FETCH_BY_IDS = """
UNWIND $ids AS nid
MATCH (n {id: nid})
OPTIONAL MATCH (n)-[r]-(neighbor)
WITH n,
     labels(n) AS lbls,
     collect(DISTINCT type(r) + ' ' + coalesce(neighbor.id, '')) AS rels
RETURN n.id AS id, lbls[0] AS label, properties(n) AS props, rels
"""

_FETCH_ALL_PAGINATED = """
MATCH (n)
WHERE n.id IS NOT NULL
OPTIONAL MATCH (n)-[r]-(neighbor)
WITH n,
     labels(n) AS lbls,
     collect(DISTINCT type(r) + ' ' + coalesce(neighbor.id, '')) AS rels
RETURN n.id AS id, lbls[0] AS label, properties(n) AS props, rels
ORDER BY n.id
SKIP $skip LIMIT $limit
"""

_BATCH_WRITE = f"""
UNWIND $updates AS upd
MATCH (n {{id: upd.id}})
SET n.embedding = upd.vec, n:{BASE_LABEL}
"""


def build_embedding_text(node_id: str, node_type: str, props: dict, rels: list) -> str:
    """
    Build a rich text string that represents a node for embedding.

    Format:
        <Type>: <id>
        <description / text_excerpt if present>
        <other key properties>
        Relationships: REL_TYPE neighbor_id; ...
    """
    parts = [f"{node_type}: {node_id}"]

    # Primary descriptive text
    for key in ("description", "text_excerpt", "text", "paragraph_title"):
        val = props.get(key)
        if val:
            parts.append(str(val))
            break  # one descriptive field is enough

    # Structured properties that add retrieval signal
    for key in ("paragraph_number", "source", "fine_amount", "penalty"):
        val = props.get(key)
        if val is not None:
            parts.append(f"{key}: {val}")

    # Relationship context (capped to avoid exceeding token limits)
    meaningful_rels = [r for r in (rels or []) if r.strip() and not r.endswith(" ")]
    if meaningful_rels:
        parts.append("Relationships: " + "; ".join(meaningful_rels[:25]))

    return "\n".join(parts)


def embed_graph_documents(graph, graph_documents, embedder) -> None:
    """
    Embed nodes from a freshly added batch of graph documents and write
    embeddings back to Neo4j.  Called after every `add_graph_documents` call
    during extraction so embeddings stay in sync with the graph.
    """
    node_ids = list({node.id for doc in graph_documents for node in doc.nodes})
    if not node_ids:
        return

    results = graph.query(_FETCH_BY_IDS, {"ids": node_ids})
    _embed_and_store(graph, results, embedder)
    logger.info("Embedded %d nodes from batch", len(results))


def embed_all_nodes(graph, embedder, batch_size: int = 100) -> None:
    """
    One-time backfill: iterate over every node in the graph, embed it, and
    write the vector + :__KGNode__ label back.  Safe to re-run — existing
    embeddings are simply overwritten.
    """
    count_result = graph.query(
        "MATCH (n) WHERE n.id IS NOT NULL RETURN count(n) AS cnt"
    )
    total = count_result[0]["cnt"] if count_result else 0
    if total == 0:
        logger.info("No nodes found — nothing to embed.")
        return

    logger.info("Embedding %d nodes in batches of %d...", total, batch_size)
    offset = 0
    embedded = 0
    while True:
        results = graph.query(
            _FETCH_ALL_PAGINATED, {"skip": offset, "limit": batch_size}
        )
        if not results:
            break
        _embed_and_store(graph, results, embedder)
        embedded += len(results)
        logger.info("  %d / %d nodes embedded", embedded, total)
        offset += batch_size

    logger.info("Backfill complete — %d nodes embedded.", embedded)


def ensure_vector_index(graph) -> None:
    """Create the vector index on :__KGNode__ if it doesn't already exist."""
    cypher = f"""
    CREATE VECTOR INDEX node_embeddings IF NOT EXISTS
    FOR (n:{BASE_LABEL})
    ON n.embedding
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIMENSIONS},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    try:
        graph.query(cypher)
        logger.info("Vector index 'node_embeddings' ensured on :%s", BASE_LABEL)
    except Exception as e:
        logger.warning("Could not create vector index: %s", e)


# ---------------------------------------------------------------------------
# Local embedder
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    Thin wrapper around a sentence-transformers model.
    Exposes embed_documents / embed_query to match the LangChain embedder interface
    so it's a drop-in replacement for OpenAIEmbeddings.

    The model is downloaded on first instantiation (~80 MB) and cached in
    ~/.cache/huggingface/hub/ inside the container.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready")

    def embed_documents(self, texts: list) -> list:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_and_store(graph, results: list, embedder) -> None:
    """Embed a list of node records and batch-write vectors to Neo4j."""
    if not results:
        return

    texts = [
        build_embedding_text(
            r["id"],
            r["label"] or "Unknown",
            r["props"] or {},
            r["rels"] or [],
        )
        for r in results
    ]

    vectors = embedder.embed_documents(texts)

    updates = [
        {"id": r["id"], "vec": vec}
        for r, vec in zip(results, vectors)
    ]
    graph.query(_BATCH_WRITE, {"updates": updates})

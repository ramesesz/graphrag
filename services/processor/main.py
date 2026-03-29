import os
import json
import glob
import time
import random
import argparse
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

from pipeline.config_loader import DomainConfig, load_domain_config, list_domains

load_dotenv()

# --- Configuration ---
APP_DIR = Path("/app") if Path("/app").exists() else Path.cwd()
DATA_DIR = APP_DIR / "data"
INPUT_DIR = DATA_DIR / "input_docs"
OUTPUT_DIR = DATA_DIR / "output_json"
CHUNKS_DIR = DATA_DIR / "output_chunks"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger.info("App directory: %s", APP_DIR)
logger.info("Data directory: %s", DATA_DIR)

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Connect to Neo4j
# (Retry loop because Neo4j takes time to initialize the bolt protocol)
graph = None
max_retries = 30
retry_delay = 2
for i in range(max_retries):
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username="neo4j",
            password=os.environ["NEO4J_PASSWORD"]
        )
        graph.refresh_schema()
        logger.info("Connected to Neo4j")
        break
    except Exception as e:
        remaining_attempts = max_retries - i - 1
        if remaining_attempts > 0:
            logger.warning("Connection attempt %d/%d failed. Retrying in %ds...", i + 1, max_retries, retry_delay)
            time.sleep(retry_delay)
        else:
            logger.error("Failed to connect to Neo4j after %d attempts: %s", max_retries, e)
            logger.error("Make sure Neo4j container is running: docker-compose logs neo4j")


def build_transformer(config: DomainConfig) -> LLMGraphTransformer:
    """Build an LLMGraphTransformer from a domain config."""
    llm = ChatOpenAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        api_key=OPENAI_API_KEY,
    )
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", config.system_prompt),
        ("human", "{input}"),
    ])
    return LLMGraphTransformer(
        prompt=custom_prompt,
        llm=llm,
        allowed_nodes=config.allowed_nodes,
        allowed_relationships=config.allowed_relationships,
        node_properties=config.node_properties,
        relationship_properties=config.relationship_properties,
    )


def save_graph_to_json(graph_documents, filename: str) -> None:
    data_export = []
    for doc in graph_documents:
        nodes = [{"id": n.id, "type": n.type, "properties": n.properties} for n in doc.nodes]
        rels = [{
            "source": r.source.id,
            "target": r.target.id,
            "type": r.type,
            "properties": r.properties
        } for r in doc.relationships]
        data_export.append({
            "source_text_chunk": doc.source.page_content,
            "nodes": nodes,
            "relationships": rels
        })

    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_export, f, indent=2)
    logger.info("Saved JSON to: %s", filename)


def save_chunks_to_json(chunks, filename: str) -> None:
    """Save text chunks as intermediate JSON."""
    data_export = [
        {"chunk_id": i, "content": chunk.page_content, "metadata": chunk.metadata}
        for i, chunk in enumerate(chunks)
    ]
    path = CHUNKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_export, f, indent=2)
    logger.info("Saved %d chunks to: %s", len(chunks), filename)


def load_chunks_from_json(filename: str):
    """Load text chunks from a saved JSON file."""
    path = CHUNKS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = [Document(page_content=item["content"], metadata=item["metadata"]) for item in data]
    logger.info("Loaded %d chunks from %s", len(chunks), filename)
    return chunks


def _load_file(file_path: str) -> Document:
    """Load a PDF or HTML file and return a single merged Document."""
    ext = Path(file_path).suffix.lower()

    if ext == ".html":
        from bs4 import BeautifulSoup
        raw_bytes = Path(file_path).read_bytes()
        for encoding in ("utf-8", "iso-8859-1", "windows-1252"):
            try:
                raw_html = raw_bytes.decode(encoding)
                soup = BeautifulSoup(raw_html, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
                content = re.sub(r"\n{3,}", "\n\n", text.strip())
                logger.info("Loaded HTML with encoding: %s", encoding)
                return Document(page_content=content, metadata={"source": file_path})
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {file_path} with utf-8, iso-8859-1, or windows-1252")

    # PDF: merge pages with page markers
    loader = PyPDFLoader(file_path)
    raw_pages = loader.load()
    merged_content = ""
    base_metadata = raw_pages[0].metadata if raw_pages else {}
    for i, page in enumerate(raw_pages):
        label = page.metadata.get("page_label", str(i + 1))
        content_cleaned = page.page_content.replace("OceanofPDF.com", "")
        content_cleaned = re.sub(r"\n\s*\n", "\n\n", content_cleaned)
        merged_content += f"\n\n--- Page {label} ---\n" + content_cleaned
    return Document(page_content=merged_content, metadata=base_metadata)


def _add_chunk_metadata(chunks, file_path: str, is_pdf: bool) -> None:
    """Inject source/page metadata into each chunk in-place."""
    source_doc_name = os.path.basename(file_path)
    current_page_label = "1"
    for chunk in chunks:
        if is_pdf:
            match = re.search(r"--- Page (\d+) ---", chunk.page_content)
            if match:
                current_page_label = match.group(1)
            chunk.metadata["page_label"] = current_page_label
            chunk.metadata["page_number"] = int(current_page_label)
            chunk.page_content += f"\n\n[Source: {source_doc_name} | Page: {current_page_label}]"
        else:
            chunk.metadata["source"] = file_path
            chunk.page_content += f"\n\n[Source: {source_doc_name}]"


def _convert_with_retry(llm_transformer, batch, max_retries: int = 8):
    """Convert a batch to graph documents, retrying on rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            return llm_transformer.convert_to_graph_documents(batch)
        except Exception as e:
            is_rate_limit = (
                "429" in str(e)
                or "rate_limit" in str(e).lower()
                or type(e).__name__ == "RateLimitError"
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = min(120, (2 ** attempt) + random.uniform(0, 2))
            logger.warning(
                "Rate limit hit (attempt %d/%d). Waiting %.1fs before retry...",
                attempt + 1, max_retries, wait,
            )
            time.sleep(wait)


def process_document(file_path: str, mode: str, config: DomainConfig) -> None:
    """
    Process a document based on the specified mode.

    Modes:
    - "full":   file → chunks → graph extraction → JSON → Neo4j
    - "chunks": file → chunks JSON only (fast, for testing/reviewing)
    - "graph":  Load existing chunks JSON → graph extraction → JSON → Neo4j
    """
    base_filename = Path(file_path).stem
    chunks_filename = f"{base_filename}_chunks.json"

    if mode == "graph":
        logger.info("Processing graph extraction from chunks: %s", base_filename)
        try:
            chunks = load_chunks_from_json(chunks_filename)
        except FileNotFoundError as e:
            logger.error("%s", e)
            logger.warning("Please run in 'chunks' or 'full' mode first to generate %s", chunks_filename)
            return
    else:
        logger.info("Processing: %s", os.path.basename(file_path))

        is_pdf = Path(file_path).suffix.lower() == ".pdf"
        single_doc = _load_file(file_path)

        cfg = config.chunking
        separators = list(cfg.separators)
        if is_pdf and "--- Page" not in separators:
            separators = ["--- Page"] + separators

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=separators,
        )
        chunks = text_splitter.split_documents([single_doc])
        _add_chunk_metadata(chunks, file_path, is_pdf)

        logger.info("Split into %d chunks.", len(chunks))
        save_chunks_to_json(chunks, chunks_filename)

        if mode == "chunks":
            return

    # Build transformer from domain config (loaded fresh per document to use correct schema)
    llm_transformer = build_transformer(config)

    batch_size = config.llm.batch_size
    total_chunks = len(chunks)
    logger.info("Extracting graph from %d chunks in batches of %d...", total_chunks, batch_size)

    json_filename = f"{base_filename}_graph.json"
    all_graph_documents = []
    for batch_start in range(0, total_chunks, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total_chunks)
        logger.info("  Batch %d-%d / %d", batch_start + 1, batch_end, total_chunks)
        batch_docs = _convert_with_retry(llm_transformer, batch)
        all_graph_documents.extend(batch_docs)

        if graph:
            graph.add_graph_documents(batch_docs)

        # Save incrementally so progress survives a later failure
        save_graph_to_json(all_graph_documents, json_filename)


def main():
    available_domains = list_domains()

    parser = argparse.ArgumentParser(
        description="GraphRAG extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modes:
  full    - Complete pipeline: PDF -> chunks JSON -> graph extraction -> JSON -> Neo4j (default)
  chunks  - Extract text chunks only: PDF -> chunks JSON (fast, no LLM cost)
  graph   - Extract graph from saved chunks: chunks JSON -> graph extraction -> JSON -> Neo4j

Available domains: {available_domains}

Examples:
  python main.py --domain game_of_thrones --mode full
  python main.py --domain stvo_stvozo --mode chunks
  python main.py --domain fantasy_litrpg --mode graph
        """
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=available_domains,
        help=f"Domain config to use. Available: {available_domains}",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "chunks", "graph"],
        default="full",
        help="Processing mode (default: full)",
    )
    args = parser.parse_args()

    config = load_domain_config(args.domain)
    logger.info("Domain: %s | Mode: %s", config.display_name, args.mode)

    if args.mode in ["full", "graph"] and not graph:
        logger.error("Could not connect to Neo4j. Cannot run in 'full' or 'graph' mode.")
        return

    # Look for documents in the domain-specific subdirectory, falling back to the root input dir
    domain_input_dir = INPUT_DIR / args.domain
    search_dir = domain_input_dir if domain_input_dir.exists() else INPUT_DIR
    input_files = (
        glob.glob(str(search_dir / "*.pdf")) +
        glob.glob(str(search_dir / "*.html"))
    )

    if not input_files:
        logger.warning(
            "No files found. Place PDFs or HTML files in data/input_docs/%s/ and re-run.", args.domain
        )
        return

    logger.info("Found %d file(s) to process", len(input_files))

    for input_file in input_files:
        try:
            process_document(input_file, mode=args.mode, config=config)
        except Exception as e:
            logger.error("Error processing %s: %s", input_file, e)


if __name__ == "__main__":
    main()

import os
import json
import glob
import time
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# --- Configuration ---
INPUT_DIR = "/app/input_docs"
OUTPUT_DIR = "/app/output_json"
CHUNKS_DIR = "/app/output_chunks"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Connect to Neo4j
# (Retry loop because Neo4j takes time to initialize the bolt protocol)
graph = None
max_retries = 30
retry_delay = 2
for i in range(max_retries):
    try:
        graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username="neo4j",
            password=os.environ["NEO4J_PASSWORD"]
        )
        graph.refresh_schema()
        print("   [✓] Connected to Neo4j")
        break
    except Exception as e:
        remaining_attempts = max_retries - i - 1
        if remaining_attempts > 0:
            print(f"   [!] Connection attempt {i+1}/{max_retries} failed. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            print(f"   [✗] Failed to connect to Neo4j after {max_retries} attempts: {e}")
            print("   Make sure Neo4j container is running: docker-compose logs neo4j")

# Setup Local LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url=OLLAMA_URL
)

# The "Extractor" 
# Note: LLMGraphTransformer works best with models that follow instructions well (like Llama 3.1)
llm_transformer = LLMGraphTransformer(llm=llm)

def save_graph_to_json(graph_documents, filename):
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
            "source_text_chunk": doc.source.page_content[:100] + "...",
            "nodes": nodes,
            "relationships": rels
        })
        
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_export, f, indent=2)
    print(f"   [✓] Saved JSON to: {filename}")

def load_chunks_from_json(filename):
    """Load text chunks from a saved JSON file."""
    path = os.path.join(CHUNKS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct LangChain Document objects
    from langchain_core.documents import Document
    chunks = [
        Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        for item in data
    ]
    print(f"   [✓] Loaded {len(chunks)} chunks from {filename}")
    return chunks

def process_document(file_path, mode="full"):
    """
    Process a document based on the specified mode.
    
    Modes:
    - "full": PDF → chunks → graph extraction → JSON → Neo4j
    - "chunks": PDF → chunks (save JSON only)
    - "graph": Load chunks JSON → graph extraction → JSON → Neo4j
    """
    base_filename = os.path.basename(file_path).replace(".pdf", "")
    chunks_filename = f"{base_filename}_chunks.json"
    
    if mode == "graph":
        # Graph mode: load chunks from JSON
        print(f"\nProcessing graph extraction from chunks: {base_filename}...")
        try:
            chunks = load_chunks_from_json(chunks_filename)
        except FileNotFoundError as e:
            print(f"   [✗] {e}")
            print(f"   [!] Please run in 'chunks' or 'full' mode first to generate {chunks_filename}")
            return
    else:
        # Full/chunks mode: load PDF and extract chunks
        print(f"\nProcessing: {os.path.basename(file_path)}...")
        
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(raw_docs)
        print(f"   [i] Split into {len(chunks)} chunks.")
        
        # Save chunks to JSON
        save_chunks_to_json(chunks, chunks_filename)
        
        # If mode is chunks-only, stop here
        if mode == "chunks":
            return
    
    print("   [i] Extracting graph (this may take time on CPU/iGPU)...")
    graph_documents = llm_transformer.convert_to_graph_documents(chunks)
    
    json_filename = f"{base_filename}_graph.json"
    save_graph_to_json(graph_documents, json_filename)

    if graph:
        print("   [i] Writing to Neo4j...")
        graph.add_graph_documents(graph_documents)

def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full    - Complete pipeline: PDF → chunks JSON → graph extraction → JSON → Neo4j (default)
  chunks  - Extract text chunks only: PDF → chunks JSON (fast, for testing/reviewing)
  graph   - Extract graph from chunks: chunks JSON → graph extraction → JSON → Neo4j (requires chunks file)
        """
    )
    parser.add_argument(
        "--mode",
        choices=["full", "chunks", "graph"],
        default="full",
        help="Processing mode (default: full)"
    )
    args = parser.parse_args()
    
    # For graph-only mode, ensure Neo4j is connected
    if args.mode in ["full", "graph"] and not graph:
        print("[✗] ERROR: Could not connect to Neo4j. Cannot run in 'full' or 'graph' mode.")
        return
    
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    if not pdf_files:
        print("No PDFs found in input folder.")
        return
    
    print(f"[i] Running in '{args.mode}' mode")
    print(f"[i] Found {len(pdf_files)} PDF(s) to process\n")

    for pdf_file in pdf_files:
        try:
            process_document(pdf_file, mode=args.mode)
        except Exception as e:
            print(f"[✗] Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    main()
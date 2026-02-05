import os
import json
import glob
import time
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Configuration ---
# Use /app as base directory if it exists (Docker), otherwise use current directory
APP_DIR = Path("/app") if Path("/app").exists() else Path.cwd()
DATA_DIR = APP_DIR / "data"
INPUT_DIR = DATA_DIR / "input_docs"
OUTPUT_DIR = DATA_DIR / "output_json"
CHUNKS_DIR = DATA_DIR / "output_chunks"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"[i] App directory: {APP_DIR}")
print(f"[i] Data directory: {DATA_DIR}")
print(f"[i] Input directory: {INPUT_DIR}")
print(f"[i] Output directory: {OUTPUT_DIR}")

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

# Setup LLM
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0, 
    api_key=OPENAI_API_KEY
)

# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0,
#     base_url=OLLAMA_URL
# )

allowed_nodes = [
    "Person",    # Sentient individuals like Ilea [cite: 2]
    "Race",      # Biological category like Human [cite: 73]
    "Class",     # System paths or academic majors [cite: 82, 149]
    "Skill",     # System abilities like Identify [cite: 212]
    "Monster",  # Individual monsters like the Drake [cite: 211]
    "Location",  # Geographical areas like the Forest [cite: 160]
    "Item",       # Objects of interest like the Blue Flower [cite: 169]
    "Organization" # Factions or groups (optional, for future expansion)
]

allowed_relationships = [
    "HAS_CLASS",
    "HAS_SKILL",
    "ENCOUNTERED",
    "KILLED",
    "BELONGS_TO_RACE",
    "VISITED",
    "LOCATED_IN",
    "IS_HOSTILE",
    "IS_ALLY",
    "HAS_ALIAS"
]

node_properties = [
    "description",      # Visual details or lore (e.g., "A dense forest with mana-infused trees")
    "level",            # Crucial for LitRPG scaling (e.g., "Level 300")
    "status",           # Current state (e.g., "Alive", "Dead", "Evolved", "Injured")
    "attributes",       # Specific traits (e.g., "Fire resistance", "High regeneration")
]

relationship_properties = [
    "source_document",      
    "source_page",       
    "source_chunk"
]

system_prompt = (
    "# Fantasy Realm Knowledge Graph Instructions\n"
    "## 1. Overview\n"
    "You are an advanced Chronicler AI designed to extract lore and system data into "
    "structured formats for a comprehensive fantasy knowledge graph.\n"
    "Capture every entity, attribute, and bond mentioned in the text with absolute precision. "
    "Do not invent lore or assume stats not explicitly stated in the provided scrolls.\n"
    "- **Nodes** represent the inhabitants, locations, and system-constructs of the realm.\n"
    "- The goal is a high-density, interconnected map of the world's narrative state.\n\n"
    
    "## 2. Entity Labeling (Nodes)\n"
    "- **Label Consistency**: Use the provided allowed labels. Map specific roles to "
    "the base categories. \n"
    "- **Example**: If the text mentions a 'Necromancer', label it as **'Person'**. "
    "If it mentions a 'High Elf', label it as **'Race'**. If it mentions a 'Fireball', "
    "label it as **'Skill'**.\n"
    "- **Node IDs**: Use the entity's primary name found in the text (e.g., 'Ilea', "
    "'Blackwood Forest'). Never use generic integers like 'Node_1'.\n\n"
    
    "## 3. Relationship Logic\n"
    "- **Timelessness**: Focus on the state of being rather than a fleeting action. "
    "Instead of 'LEARNED_BLINK', use **'HAS_SKILL'**. Instead of 'FOUGHT_THE_DRAKE', "
    "use **'ENCOUNTERED'** or **'IS_HOSTILE'** depending on the outcome.\n"
    "- **Directionality**: Ensure the relationship flows logically (e.g., [Person]->[HAS_CLASS]->[Class]).\n\n"
    
    "## 4. Coreference & Aliases\n"
    "- **Entity Resolution**: Fantasy characters often have many titles. If 'The Unbound', "
    "'Ilea', and 'she' are used, map all data to the most complete name: **'Ilea'**.\n"
    "- Use the **'HAS_ALIAS'** relationship to link titles (e.g., [Ilea]->HAS_ALIAS->[The Unbound]).\n\n"
    
    "## 5. Property Extraction\n"
    "- For every node, aggressively extract **'level'**, **'status'**, and **'attributes'**. "
    "Example: If a character is 'Level 45 and poisoned', set level: '45' and status: 'Poisoned'.\n\n"
    
    "## 6. Strict Compliance\n"
    "Follow these protocols precisely. Inaccuracy or hallucination will compromise the "
    "integrity of the realm's history."
)

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"), # LLMGraphTransformer injects the chunk here
])

# The "Extractor" 
# Note: LLMGraphTransformer works best with models that follow instructions well (like Llama 3.1)
llm_transformer = LLMGraphTransformer(
    prompt=custom_prompt,
    llm=llm, 
    allowed_nodes=allowed_nodes, 
    allowed_relationships=allowed_relationships,
    node_properties=node_properties,
    relationship_properties=relationship_properties,
)


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
            "source_text_chunk": doc.source.page_content,
            "nodes": nodes,
            "relationships": rels
        })
        
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_export, f, indent=2)
    print(f"   [✓] Saved JSON to: {filename}")

def save_chunks_to_json(chunks, filename):
    """Save text chunks as intermediate JSON."""
    data_export = [
        {
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for i, chunk in enumerate(chunks)
    ]
    
    path = CHUNKS_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_export, f, indent=2)
    print(f"   [✓] Saved chunks to: {filename}")

def load_chunks_from_json(filename):
    """Load text chunks from a saved JSON file."""
    path = CHUNKS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
        print(f"\nProcessing: {os.path.basename(file_path)}...")
        
        loader = PyPDFLoader(file_path)
        raw_pages = loader.load()
                
        # Inject page markers
        merged_content = ""
        base_metadata = raw_pages[0].metadata if raw_pages else {}
        
        # Inject markers into the text stream
        for i, page in enumerate(raw_pages):
            # Using a distinct format: "--- Page 5 ---"
            # The 'page_label' is often cleaner than 'page' index if available
            label = page.metadata.get("page_label", str(i + 1))
            content_cleaned = page.page_content.replace("OceanofPDF.com", "")
            
            # Optional: Remove double newlines created by the removal if needed
            content_cleaned = re.sub(r'\n\s*\n', '\n\n', content_cleaned)
            
            merged_content += f"\n\n--- Page {label} ---\n" + content_cleaned

        # Combine documents into a single one
        single_doc = Document(page_content=merged_content, metadata=base_metadata)
        
        # Split documents with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
            # We add our custom page marker to separators to help the splitter break cleanly there
            separators=["--- Page", "\n\n", "\n", ".", " "] 
        )
        chunks = text_splitter.split_documents([single_doc])
        
        # Re-extract injected metadata for each chunk
        current_page_label = raw_pages[0].metadata.get("page_label", "1")
        
        for chunk in chunks:
            # Look for the page marker in this specific chunk
            # Regex explanation: Looks for "--- Page " followed by digits
            match = re.search(r"--- Page (\d+) ---", chunk.page_content)
            
            if match:
                # If we find a new marker, update our tracker
                current_page_label = match.group(1)
            
            # Update the chunk's metadata with the tracked page number
            chunk.metadata["page_label"] = current_page_label
            chunk.metadata["page_number"] = int(current_page_label)
            chunk.metadata["source"] = str(file_path)
            
            source_doc_name = os.path.basename(chunk.metadata["source"])
            source_page_num = chunk.metadata["page_number"]
            
            # Append to the actual text content so the LLM/Embedder "sees" it
            chunk.page_content += f"\n\n[Source: {source_doc_name} | Page: {source_page_num}]"

        print(f"   [i] Split into {len(chunks)} chunks. Metadata restored.")
        
        # Save chunks to JSON
        save_chunks_to_json(chunks, chunks_filename)
        
        # If mode is chunks-only, stop here
        if mode == "chunks":
            return
    
    num_chunks = 10
    print(f"   [i] Extracting graph from {num_chunks} chunks (this may take time on CPU/iGPU)...")
    graph_documents = llm_transformer.convert_to_graph_documents(chunks[:num_chunks])
    
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
    
    pdf_files = glob.glob(str(INPUT_DIR / "*.pdf"))
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
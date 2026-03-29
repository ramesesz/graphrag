import streamlit as st
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from streamlit_agraph import agraph, Node, Edge, Config

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Allow importing config_loader from the processor pipeline package
# (both share the configs/ volume at /app/configs in Docker)
_PROCESSOR_DIR = Path("/app/processor") if Path("/app/processor").exists() else Path(__file__).parents[1] / "processor"
if str(_PROCESSOR_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESSOR_DIR))

try:
    from pipeline.config_loader import load_domain_config, list_domains, DomainConfig
    _CONFIG_LOADER_AVAILABLE = True
except ImportError:
    _CONFIG_LOADER_AVAILABLE = False
    logger.warning("config_loader not available — domain config features disabled")

    class DomainConfig:  # stub so type annotations don't fail at runtime
        pass

    def load_domain_config(domain_id):
        return None

    def list_domains():
        return []

load_dotenv()

# --- 1. Page config ---
st.set_page_config(layout="wide", page_title="GraphRAG Knowledge Chat")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROCESSOR_API_URL = os.getenv("PROCESSOR_API_URL", "http://processor:8000")

# --- 2. Initialize connections ---
if "graph" not in st.session_state:
    try:
        st.session_state.graph = Neo4jGraph(
            url=NEO4J_URI,
            username="neo4j",
            password=NEO4J_PASSWORD
        )
        st.toast("Connected to Neo4j", icon="✅")
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")

if "llm" not in st.session_state:
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Please check your .env file.")
        st.stop()
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_domain" not in st.session_state:
    st.session_state.active_domain = None

if "domain_config" not in st.session_state:
    st.session_state.domain_config = None


# --- 3. Node color registry ---

DEFAULT_NODE_COLORS = {
    # Fantasy / LitRPG
    "Person": "#FF6B6B",
    "Monster": "#FF8C42",
    "Skill": "#4ECDC4",
    "Class": "#A78BFA",
    "Race": "#F59E0B",
    "Item": "#34D399",
    # Game of Thrones
    "Character": "#FF6B6B",
    "House": "#F59E0B",
    "Battle": "#FF8C42",
    "Alliance": "#60A5FA",
    "Faction": "#A78BFA",
    "Title": "#D1FAE5",
    "Weapon": "#9CA3AF",
    "Event": "#FDE68A",
    # Legal (StVO/StVZO)
    "Paragraph": "#3B82F6",
    "Rule": "#60A5FA",
    "Violation": "#EF4444",
    "Fine": "#F97316",
    "Penalty": "#DC2626",
    "Prohibition": "#FCA5A5",
    "Requirement": "#BFDBFE",
    "Exception": "#9CA3AF",
    "VehicleCategory": "#34D399",
    "PersonRole": "#6EE7B7",
    "RoadType": "#A7F3D0",
    "Permit": "#FDE68A",
    "Authority": "#C4B5FD",
    "Definition": "#E5E7EB",
    # Shared
    "Location": "#86EFAC",
    "Organization": "#FCD34D",
}


def get_node_colors() -> dict:
    """Return node color map, preferring domain config values over defaults."""
    cfg: DomainConfig | None = st.session_state.get("domain_config")
    if cfg and cfg.visualization.get("node_colors"):
        return {**DEFAULT_NODE_COLORS, **cfg.visualization["node_colors"]}
    return DEFAULT_NODE_COLORS


# --- 4. Domain selector sidebar ---

def render_processor_panel():
    """Sidebar panel for uploading PDFs and triggering processing."""
    if not _REQUESTS_AVAILABLE:
        return

    st.sidebar.divider()
    st.sidebar.subheader("Process Documents")

    domain = st.session_state.get("active_domain")
    if not domain:
        st.sidebar.caption("Select a domain first.")
        return

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF", type=["pdf"], key="pdf_uploader"
    )
    if uploaded_file and st.sidebar.button("Upload to graph"):
        try:
            resp = _requests.post(
                f"{PROCESSOR_API_URL}/upload/{domain}",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                timeout=30,
            )
            if resp.ok:
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
            else:
                st.sidebar.error(f"Upload failed: {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Upload error: {e}")

    mode = st.sidebar.selectbox(
        "Extraction mode",
        ["full", "chunks", "graph"],
        help="full=PDF→graph, chunks=PDF only, graph=chunks→graph",
        key="extraction_mode",
    )
    if st.sidebar.button("▶ Start extraction", type="primary"):
        try:
            resp = _requests.post(
                f"{PROCESSOR_API_URL}/process",
                params={"domain": domain, "mode": mode},
                timeout=10,
            )
            if resp.ok:
                job = resp.json()
                st.sidebar.success(f"Job started: `{job['job_id'][:8]}...`")
            else:
                st.sidebar.error(f"Failed to start job: {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Could not reach processor API: {e}")


def render_domain_selector():
    st.sidebar.title("Knowledge Domain")

    if not _CONFIG_LOADER_AVAILABLE:
        st.sidebar.warning("Domain configs not available.")
        return

    domains = list_domains()
    if not domains:
        st.sidebar.warning("No domain configs found in configs/domains/.")
        return

    # Build display names
    domain_labels = {}
    for d in domains:
        try:
            cfg = load_domain_config(d)
            domain_labels[d] = cfg.display_name
        except Exception:
            domain_labels[d] = d

    selected = st.sidebar.selectbox(
        "Select domain",
        options=domains,
        format_func=lambda x: domain_labels.get(x, x),
        key="domain_selector_widget",
    )

    if selected != st.session_state.active_domain:
        st.session_state.active_domain = selected
        try:
            st.session_state.domain_config = load_domain_config(selected)
        except Exception as e:
            st.sidebar.error(f"Failed to load domain config: {e}")
            st.session_state.domain_config = None
        # Clear chat history when switching domains
        st.session_state.messages = []
        st.rerun()


# --- 5. Logic functions ---

def extract_entities(question: str) -> list[str]:
    domain_cfg: DomainConfig | None = st.session_state.get("domain_config")
    is_legal = domain_cfg and domain_cfg.domain_id == "stvo_stvozo"

    if is_legal:
        prompt = ChatPromptTemplate.from_template(
            """Du bist ein Experte für Named Entity Recognition (NER) im deutschen Verkehrsrecht.
Extrahiere ALLE relevanten Entitäten aus der Frage: Paragraphen (z.B. "§ 3"), Fahrzeugkategorien,
Personenrollen, Straßentypen, Verstöße, Bußgelder.
Gib NUR eine kommagetrennte Liste der Entitäten zurück.
Frage: {question}
Antwort:"""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """You are a Named Entity Recognition (NER) expert.
Extract ALL meaningful entities (Persons, Characters, Houses, Locations, Monsters, Skills,
Paragraphs, Vehicles, Rules) from the question.
Return ONLY a comma-separated list of names.
Question: {question}
Answer:"""
        )

    chain = prompt | st.session_state.llm
    response = chain.invoke({"question": question})
    return [e.strip() for e in response.content.split(",") if e.strip()]


# Fulltext index query — much faster and more accurate than CONTAINS
_CYPHER_FULLTEXT = """
CALL db.index.fulltext.queryNodes('node_search', $search_term)
YIELD node AS center, score
WITH center ORDER BY score DESC LIMIT 5
MATCH (center)-[rel]-(neighbor)
WITH center, rel, neighbor, count(rel) AS degree
ORDER BY degree DESC
RETURN
    center.id AS center_id,
    labels(center)[0] AS center_label,
    properties(center) AS center_props,
    type(rel) AS rel_type,
    properties(rel) AS rel_props,
    neighbor.id AS neighbor_id,
    labels(neighbor)[0] AS neighbor_label,
    properties(neighbor) AS neighbor_props
LIMIT 50
"""

# Fallback when the fulltext index doesn't exist yet
_CYPHER_CONTAINS = """
UNWIND $entities AS entity_name
MATCH (center)
WHERE toLower(center.id) CONTAINS toLower(entity_name)
MATCH (center)-[rel]-(neighbor)
WITH center, rel, neighbor, count(rel) AS degree
ORDER BY degree DESC
RETURN
    center.id AS center_id,
    labels(center)[0] AS center_label,
    properties(center) AS center_props,
    type(rel) AS rel_type,
    properties(rel) AS rel_props,
    neighbor.id AS neighbor_id,
    labels(neighbor)[0] AS neighbor_label,
    properties(neighbor) AS neighbor_props
LIMIT 50
"""


def _collect_graph_results(results) -> dict:
    graph_data: dict = {"nodes": {}, "edges": [], "context_text": []}
    for record in results:
        center_id = record["center_id"]
        center_label = record["center_label"] or "Unknown"
        center_props = record["center_props"] or {}
        rel_type = record["rel_type"]
        rel_props = record["rel_props"] or {}
        neighbor_id = record["neighbor_id"]
        neighbor_label = record["neighbor_label"] or "Unknown"
        neighbor_props = record["neighbor_props"] or {}

        if center_id not in graph_data["nodes"]:
            graph_data["nodes"][center_id] = {"id": center_id, "type": center_label, **center_props}
        if neighbor_id not in graph_data["nodes"]:
            graph_data["nodes"][neighbor_id] = {"id": neighbor_id, "type": neighbor_label, **neighbor_props}

        edge_tooltip = f"Type: {rel_type}\n" + "\n".join([f"{k}: {v}" for k, v in rel_props.items()])
        graph_data["edges"].append({
            "source": center_id,
            "target": neighbor_id,
            "label": rel_type,
            "title": edge_tooltip,
        })
        graph_data["context_text"].append(
            f"{center_id} ({center_label}) -[{rel_type}]-> {neighbor_id} ({neighbor_label})"
        )
    return graph_data


def get_graph_context(entities: list[str]) -> dict:
    if not entities:
        return {"nodes": {}, "edges": [], "context_text": []}

    # Try fulltext index first; fall back to CONTAINS if index not yet created
    search_term = " OR ".join(entities)
    try:
        results = st.session_state.graph.query(_CYPHER_FULLTEXT, {"search_term": search_term})
        return _collect_graph_results(results)
    except Exception as fulltext_err:
        if "node_search" in str(fulltext_err).lower() or "no such" in str(fulltext_err).lower():
            logger.warning("Fulltext index not available, falling back to CONTAINS search")
        else:
            logger.error("Graph query error: %s", fulltext_err)
            st.error(f"Graph query error: {fulltext_err}")
            return {"nodes": {}, "edges": [], "context_text": []}

    # Fallback
    try:
        results = st.session_state.graph.query(_CYPHER_CONTAINS, {"entities": entities})
        return _collect_graph_results(results)
    except Exception as e:
        logger.error("Fallback graph query error: %s", e)
        st.error(f"Graph query error: {e}")
        return {"nodes": {}, "edges": [], "context_text": []}


def generate_answer(question: str, context_text: list[str]) -> str:
    context_block = "\n".join(context_text) if context_text else "No specific graph data found."
    domain_cfg: DomainConfig | None = st.session_state.get("domain_config")
    is_legal = domain_cfg and domain_cfg.domain_id == "stvo_stvozo"

    if is_legal:
        prompt = ChatPromptTemplate.from_template(
            """Du bist ein Rechtsassistent für deutsches Verkehrsrecht (StVO/StVZO).

Nutze die folgenden Graphdaten als Grundlage deiner Antwort:
{context}

Beantworte die Frage präzise. Strukturiere deine Antwort wie folgt:
1. **Direkte Antwort** (1-2 Sätze)
2. **Rechtsgrundlage**: Zitiere die genaue Paragraphennummer und den Gesetzestext
3. **Ausnahmen oder Sonderfälle** (falls vorhanden)
4. **Bußgeld / Strafe** (falls relevant)

Wenn die Graphdaten keine ausreichenden Informationen enthalten, weise darauf hin.

Frage: {question}"""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            "Context from Knowledge Graph:\n{context}\n\nUser Question: {question}"
        )

    chain = prompt | st.session_state.llm
    response = chain.invoke({"context": context_block, "question": question})
    return response.content


# --- 6. Graph visualization ---

def render_graph_viz(data: dict):
    """Render the knowledge graph subnetwork using streamlit-agraph."""
    if not data or not data["nodes"]:
        return

    node_colors = get_node_colors()

    # Compute degree for proportional node sizing
    degree: dict[str, int] = {}
    for edge in data["edges"]:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    nodes = []
    for node_id, props in data["nodes"].items():
        node_type = props.get("type", "Unknown")
        color = node_colors.get(node_type, "#FFD700")
        size = 15 + min(degree.get(node_id, 0) * 3, 30)  # scale 15-45
        tooltip = f"Type: {node_type}\n" + "\n".join(
            [f"{k}: {v}" for k, v in props.items() if k not in ("id", "type")]
        )
        nodes.append(Node(id=node_id, label=node_id, size=size, color=color, title=tooltip))

    edges = [
        Edge(source=e["source"], target=e["target"], label=e["label"], title=e["title"])
        for e in data["edges"]
    ]
    config = Config(width=800, height=450, directed=True, physics=True)
    return agraph(nodes=nodes, edges=edges, config=config)


# --- 7. Main UI ---

render_domain_selector()
render_processor_panel()

domain_cfg: DomainConfig | None = st.session_state.get("domain_config")
title = domain_cfg.display_name if domain_cfg else "GraphRAG Knowledge Chat"
is_legal = domain_cfg and domain_cfg.domain_id == "stvo_stvozo"

st.title(f"{'⚖️' if is_legal else '🧙'} {title}")

if is_legal:
    st.caption("Fragen Sie nach Paragraphen, Geschwindigkeiten, Bußgeldern und Fahrzeugvorschriften.")
elif domain_cfg and domain_cfg.domain_id == "game_of_thrones":
    st.caption("Ask about characters, houses, battles, and alliances in Westeros.")
else:
    st.caption("Ask about entities, skills, and relationships in the knowledge graph.")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "graph_data" in message and message["graph_data"]["nodes"]:
            with st.expander("Explore Graph Context"):
                render_graph_viz(message["graph_data"])

# New input
placeholder = (
    "Fragen Sie z.B.: Wie schnell darf ich innerorts fahren?"
    if is_legal
    else "Ask something..."
)
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("🛠️ Analyzing Knowledge Graph...", expanded=True)
        with status_box:
            st.write("Extracting entities...")
            entities = extract_entities(prompt)
            st.write(f"Detected: `{entities}`")

            st.write("Searching Knowledge Graph...")
            graph_data = get_graph_context(entities)

            if not graph_data["nodes"]:
                st.warning("No matching nodes found in the graph.")
            else:
                st.success(f"Found {len(graph_data['nodes'])} related nodes.")

            status_box.update(label="Analysis Complete", state="complete", expanded=False)

        with st.spinner("Generating response..."):
            response_text = generate_answer(prompt, graph_data["context_text"])

        st.markdown(response_text)

        if graph_data["nodes"]:
            with st.expander("Graph Subnetwork", expanded=True):
                render_graph_viz(graph_data)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "graph_data": graph_data,
        })

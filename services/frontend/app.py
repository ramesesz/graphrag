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


# --- 4. Document browser sidebar ---

def _set_active_domain(domain: str) -> None:
    """Switch active domain and clear chat history."""
    if domain == st.session_state.active_domain:
        return
    st.session_state.active_domain = domain
    try:
        st.session_state.domain_config = load_domain_config(domain)
    except Exception:
        st.session_state.domain_config = None
    st.session_state.messages = []


def _start_extraction(domain: str, mode: str, files_filter: list[str] | None = None) -> None:
    """POST /process, store job in session state, and toast the result."""
    params = {"domain": domain, "mode": mode}
    if files_filter:
        params["files"] = files_filter
    try:
        resp = _requests.post(
            f"{PROCESSOR_API_URL}/process",
            params=params,
            timeout=10,
        )
        if resp.ok:
            job = resp.json()
            if "recent_jobs" not in st.session_state:
                st.session_state.recent_jobs = []
            st.session_state.recent_jobs.append({
                "job_id": job["job_id"],
                "domain": domain,
                "mode": mode,
                "status": "queued",
            })
            st.toast(f"Job started: {job['job_id'][:8]}...", icon="▶")
        else:
            st.error(f"Failed to start job: {resp.text}")
    except Exception as e:
        st.error(f"Could not reach processor API: {e}")


def render_jobs_panel() -> None:
    """Sidebar panel showing recent job statuses with a refresh button."""
    if "recent_jobs" not in st.session_state or not st.session_state.recent_jobs:
        return

    st.sidebar.divider()
    st.sidebar.subheader("Jobs")

    updated = []
    for job in st.session_state.recent_jobs:
        try:
            r = _requests.get(f"{PROCESSOR_API_URL}/jobs/{job['job_id']}", timeout=5)
            if r.ok:
                job = {**job, **r.json()}
        except Exception:
            pass

        status = job.get("status", "?")
        icon = {"queued": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")
        total = job.get("total_files")
        done = job.get("processed_files")
        progress = f" ({done}/{total} files)" if total else ""
        st.sidebar.markdown(
            f"{icon} **{job['domain']}** — `{status}`{progress}  \n"
            f"<small>`{job['job_id'][:8]}...` · {job['mode']}</small>",
            unsafe_allow_html=True,
        )
        if job.get("error"):
            st.sidebar.caption(f"Error: {job['error']}")

        # Keep only non-completed jobs + last 3 completed/failed
        if status not in ("completed", "failed"):
            updated.append(job)
        else:
            updated.append(job)

    # Trim: keep all running/queued + last 3 terminal ones
    running = [j for j in updated if j.get("status") not in ("completed", "failed")]
    terminal = [j for j in updated if j.get("status") in ("completed", "failed")][-3:]
    st.session_state.recent_jobs = running + terminal

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Refresh", key="jobs_refresh"):
            st.rerun()
    with col2:
        if st.button("Clear", key="jobs_clear"):
            st.session_state.recent_jobs = []
            st.rerun()


def render_document_browser():
    """Sidebar document browser: domains as expanders, files with extraction badges."""
    st.sidebar.title("Knowledge Domains")

    if not _REQUESTS_AVAILABLE or not _CONFIG_LOADER_AVAILABLE:
        st.sidebar.warning("Processor API or config loader not available.")
        return

    # Fetch document state from processor API
    try:
        resp = _requests.get(f"{PROCESSOR_API_URL}/documents", timeout=5)
        doc_state = resp.json() if resp.ok else {}
    except Exception:
        doc_state = {}

    # Fall back to just listing domains without file state
    domains = list_domains()
    if not domains:
        st.sidebar.warning("No domain configs found in configs/domains/.")
        return

    # Auto-select first domain if none active
    if not st.session_state.active_domain and domains:
        _set_active_domain(domains[0])

    # Pending extraction confirmation stored in session state
    if "pending_extraction" not in st.session_state:
        st.session_state.pending_extraction = None

    for domain in domains:
        domain_info = doc_state.get(domain, {})
        display_name = domain_info.get("display_name", domain)
        files = domain_info.get("files", [])

        is_active = st.session_state.active_domain == domain

        with st.sidebar.expander(display_name, expanded=is_active):
            # Switch active domain when expander is opened
            if not is_active:
                if st.button("Switch to this domain", key=f"switch_{domain}"):
                    _set_active_domain(domain)
                    st.rerun()

            # File list with checkboxes and 4-state status badges
            selected_files = []
            if files:
                st.caption("Select files to extract:")
                for f in files:
                    if f.get("has_neo4j"):
                        badge, status = "🟢", "in Neo4j"
                    elif f.get("has_graph"):
                        badge, status = "🟠", "graph JSON only"
                    elif f.get("has_chunks"):
                        badge, status = "🟡", "chunks only"
                    else:
                        badge, status = "🔴", "not extracted"
                    checked = st.checkbox(
                        f"{badge} `{f['name']}` — *{status}*",
                        key=f"chk_{domain}_{f['name']}",
                    )
                    if checked:
                        selected_files.append(f["name"])
            else:
                st.caption("No documents yet.")

            st.divider()

            # Upload
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "html"],
                accept_multiple_files=True,
                key=f"uploader_{domain}",
            )
            if uploaded_files and st.button("Upload", key=f"upload_btn_{domain}"):
                for uf in uploaded_files:
                    mime = "application/pdf" if uf.name.lower().endswith(".pdf") else "text/html"
                    try:
                        r = _requests.post(
                            f"{PROCESSOR_API_URL}/upload/{domain}",
                            files={"file": (uf.name, uf.getvalue(), mime)},
                            timeout=30,
                        )
                        if r.ok:
                            st.success(f"Uploaded {uf.name}")
                        else:
                            st.error(f"Upload failed: {r.text}")
                    except Exception as e:
                        st.error(f"Upload error: {e}")
                st.rerun()

            # Extraction mode + trigger
            mode = st.selectbox(
                "Extraction mode",
                ["full", "chunks", "extract", "neo4j"],
                help=(
                    "full = file→chunks→LLM→graph JSON→Neo4j  |  "
                    "chunks = file→chunks JSON only  |  "
                    "extract = chunks JSON→LLM→graph JSON (no Neo4j)  |  "
                    "neo4j = graph JSON→Neo4j only"
                ),
                key=f"mode_{domain}",
            )

            btn_label = f"▶ Run on {len(selected_files)} selected" if selected_files else "▶ Extract all"
            if st.button(btn_label, type="primary", key=f"extract_{domain}"):
                target_files = selected_files if selected_files else None
                already_done = [f["name"] for f in files if f.get("has_neo4j")]
                unextracted = [
                    name for name in (target_files or [f["name"] for f in files])
                    if not next((f for f in files if f["name"] == name), {}).get("has_neo4j")
                ]

                if not selected_files and already_done and unextracted:
                    # Full-domain extract with existing Neo4j data — ask for confirmation
                    st.session_state.pending_extraction = {
                        "domain": domain,
                        "mode": mode,
                        "unextracted": unextracted,
                        "already_done": already_done,
                    }
                    st.rerun()
                else:
                    _start_extraction(domain, mode, target_files)

            # Confirmation dialog for diff case (all-domain extract only)
            pending = st.session_state.pending_extraction
            if pending and pending["domain"] == domain:
                st.warning(
                    f"**{len(pending['already_done'])} file(s) already in Neo4j.**\n\n"
                    f"Unextracted file(s):\n"
                    + "\n".join(f"- `{n}`" for n in pending["unextracted"])
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("New only", key=f"new_only_{domain}"):
                        _start_extraction(domain, pending["mode"], pending["unextracted"])
                        st.session_state.pending_extraction = None
                        st.rerun()
                with col2:
                    if st.button("Extract all", key=f"all_{domain}"):
                        _start_extraction(domain, pending["mode"])
                        st.session_state.pending_extraction = None
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


# 1. Vector similarity — semantically closest nodes to the query embedding
_CYPHER_VECTOR = """
CALL db.index.vector.queryNodes('node_embeddings', 5, $query_vec)
YIELD node AS center, score
MATCH (center)-[rel]-(neighbor)
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

# 2. Fulltext index — Lucene BM25 keyword match
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

# 3. Last-resort string match — no index required
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


def _get_query_embedding(question: str) -> list | None:
    """Call the processor /embed endpoint to get the query vector."""
    if not _REQUESTS_AVAILABLE:
        return None
    try:
        resp = _requests.post(
            f"{PROCESSOR_API_URL}/embed",
            json={"text": question},
            timeout=10,
        )
        if resp.ok:
            return resp.json()["embedding"]
        logger.warning("Embed endpoint returned %s: %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Could not reach processor /embed: %s", e)
    return None


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


def get_graph_context(question: str, entities: list[str]) -> dict:
    """
    Retrieve graph context using a three-tier fallback:
      1. Vector similarity  — embed the full question, find semantically closest nodes
      2. Fulltext (BM25)    — Lucene match on extracted entity names
      3. CONTAINS           — simple substring match, no index required
    """
    empty = {"nodes": {}, "edges": [], "context_text": []}

    # --- Tier 1: vector search ---
    query_vec = _get_query_embedding(question)
    if query_vec:
        try:
            results = st.session_state.graph.query(_CYPHER_VECTOR, {"query_vec": query_vec})
            if results:
                logger.info("Vector search returned %d rows", len(results))
                return _collect_graph_results(results)
            logger.info("Vector search returned no results, falling back to fulltext")
        except Exception as e:
            if "node_embeddings" in str(e).lower() or "no such" in str(e).lower():
                logger.warning("Vector index not available yet, falling back to fulltext")
            else:
                logger.error("Vector search error: %s", e)

    # --- Tier 2: fulltext index ---
    if not entities:
        return empty
    search_term = " OR ".join(entities)
    try:
        results = st.session_state.graph.query(_CYPHER_FULLTEXT, {"search_term": search_term})
        if results:
            logger.info("Fulltext search returned %d rows", len(results))
            return _collect_graph_results(results)
        logger.info("Fulltext search returned no results, falling back to CONTAINS")
    except Exception as e:
        if "node_search" in str(e).lower() or "no such" in str(e).lower():
            logger.warning("Fulltext index not available, falling back to CONTAINS")
        else:
            logger.error("Fulltext search error: %s", e)
            st.error(f"Graph query error: {e}")
            return empty

    # --- Tier 3: CONTAINS fallback ---
    try:
        results = st.session_state.graph.query(_CYPHER_CONTAINS, {"entities": entities})
        return _collect_graph_results(results)
    except Exception as e:
        logger.error("CONTAINS fallback error: %s", e)
        st.error(f"Graph query error: {e}")
        return empty


def generate_answer(question: str, context_text: list[str]) -> str:
    context_block = "\n".join(context_text) if context_text else "No specific graph data found."
    domain_cfg: DomainConfig | None = st.session_state.get("domain_config")
    is_legal = domain_cfg and domain_cfg.domain_id == "stvo_stvozo"

    if is_legal:
        prompt = ChatPromptTemplate.from_template(
            """Du bist ein Rechtsassistent für deutsches Verkehrsrecht (StVO/StVZO).

WICHTIGE ANWEISUNGEN:
- Antworte AUSSCHLIESSLICH auf Basis der unten stehenden Graphdaten.
- Wenn die Graphdaten die Frage nicht beantworten, sage klar: "Diese Information ist in den vorliegenden Dokumenten nicht enthalten."
- Erfinde KEINE Gesetzesinhalte, Paragraphennummern oder Definitionen, die nicht in den Graphdaten stehen.
- Spekuliere NICHT und ergänze KEIN Vorwissen aus deinem Training.
- Zitiere nur Paragraphen und Texte, die direkt in den Graphdaten erscheinen.

Graphdaten:
{context}

Beantworte die Frage präzise. Strukturiere deine Antwort wie folgt:
1. **Direkte Antwort** (1-2 Sätze, nur wenn die Information in den Graphdaten vorhanden ist)
2. **Rechtsgrundlage**: Zitiere nur Paragraphennummern und Texte, die in den Graphdaten stehen
3. **Ausnahmen oder Sonderfälle** (nur wenn in den Graphdaten erwähnt)
4. **Bußgeld / Strafe** (nur wenn in den Graphdaten enthalten)

Falls die Graphdaten keine ausreichenden Informationen enthalten, antworte: "Diese Information ist in den vorliegenden Dokumenten nicht enthalten. Ich kann dazu keine verlässliche Aussage machen."

Frage: {question}"""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """Answer the user's question based ONLY on the context from the knowledge graph below.
Do NOT use any knowledge from your training. If the context does not contain the answer, say clearly: "This information is not available in the current documents."
Do not speculate, assume, or invent facts.

Context from Knowledge Graph:
{context}

User Question: {question}"""
        )

    chain = prompt | st.session_state.llm
    response = chain.invoke({"context": context_block, "question": question})
    return response.content


# --- 6. Graph visualization ---

def render_graph_viz(data: dict, key: str = "graph"):
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
    with st.container(key=key):
        return agraph(nodes=nodes, edges=edges, config=config)


# --- 7. Main UI ---

render_document_browser()
render_jobs_panel()

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
            is_latest = i == len(st.session_state.messages) - 1
            with st.expander("Explore Graph Context", expanded=is_latest):
                render_graph_viz(message["graph_data"], key=f"graph_{i}")

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
            graph_data = get_graph_context(prompt, entities)

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
                render_graph_viz(graph_data, key=f"graph_{len(st.session_state.messages)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "graph_data": graph_data,
        })

import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from streamlit_agraph import agraph, Node, Edge, Config

# --- 1. Configuration ---
st.set_page_config(layout="wide", page_title="Azarinth Healer Graph Chat")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- 2. Initialize Connections ---
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
    st.session_state.llm = ChatOllama(
        model="llama3.1", 
        temperature=0, 
        base_url=OLLAMA_URL
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Logic Functions ---

def extract_entities(question):
    """Uses LLM to extract names of characters, places, etc."""
    prompt = ChatPromptTemplate.from_template(
        """You are a Named Entity Recognition (NER) expert. 
        Extract ALL meaningful entities (Persons, Locations, Monsters, Skills) from the question.
        Return ONLY a comma-separated list of names.
        
        Question: {question}
        Answer:"""
    )
    chain = prompt | st.session_state.llm
    response = chain.invoke({"question": question})
    entities = [e.strip() for e in response.content.split(',') if e.strip()]
    return entities

def get_graph_context(entities):
    """Queries Neo4j for entities and their 1-hop neighbors."""
    graph_data = {"nodes": {}, "edges": [], "context_text": []}
    if not entities:
        return graph_data

    # Note: Using 'id' here. If your nodes use 'name', change center.id to center.name
    cypher_query = """
    UNWIND $entities AS entity_name
    MATCH (center) 
    WHERE toLower(center.id) CONTAINS toLower(entity_name)
    MATCH path = (center)-[rel]-(neighbor)
    RETURN center, rel, neighbor
    LIMIT 50
    """
    
    try:
        results = st.session_state.graph.query(cypher_query, {"entities": entities})
        for record in results:
            source = record['center']
            target = record['neighbor']
            rel = record['rel'] # Format: (start_node, type, end_node, props)
            
            # 1. Store Unique Nodes & their full properties for hover
            for n in [source, target]:
                if n['id'] not in graph_data["nodes"]:
                    graph_data["nodes"][n['id']] = n 

            # 2. Extract Relationship info
            rel_type = rel[1]
            rel_props = rel[3] if len(rel) > 3 else {}
            
            # Create hover tooltip for edges
            edge_tooltip = f"Type: {rel_type}\n" + "\n".join([f"{k}: {v}" for k, v in rel_props.items()])
            
            graph_data["edges"].append({
                "source": source['id'],
                "target": target['id'],
                "label": rel_type,
                "title": edge_tooltip
            })
            
            # 3. Create Text Context for LLM
            context_line = f"{source['id']} ({source.get('type','')}) -[{rel_type}]-> {target['id']} ({target.get('type','')})"
            graph_data["context_text"].append(context_line)
            
    except Exception as e:
        st.error(f"Graph query error: {e}")
        
    return graph_data

def generate_answer(question, context_text):
    """Final LLM call to answer the question using graph data."""
    if not context_text:
        context_block = "No specific graph data found."
    else:
        context_block = "\n".join(context_text) if isinstance(context_text, list) else context_text
        
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant with access to a Knowledge Graph about Azarinth Healer.
        Use the following context from the graph to answer the user's question.
        
        Graph Context:
        {context}
        
        User Question: {question}
        """
    )
    chain = prompt | st.session_state.llm
    response = chain.invoke({"context": context_block, "question": question})
    return response.content

# --- 4. UI Helper Functions ---

def render_graph_viz(data, unique_id):
    """Renders the interactive AGraph."""
    if not data or not data["nodes"]:
        st.warning("No graph nodes to display.")
        return
    
    nodes = []
    for node_id, props in data["nodes"].items():
        node_type = props.get('type', 'Unknown')
        
        # Color Logic
        color = "#FFD700" # Gold
        if node_type == "Person": color = "#FF6B6B"
        elif node_type == "Skill": color = "#4ECDC4"
        elif node_type == "Creature": color = "#C7F464"
        elif node_type == "Location": color = "#556270"
        
        # Hover Tooltip (Title)
        tooltip = "Properties:\n" + "\n".join([f"{k}: {v}" for k, v in props.items()])
        
        nodes.append(Node(id=node_id, label=node_id, size=20, color=color, title=tooltip))
            
    edges = [
        Edge(source=e["source"], target=e["target"], label=e["label"], title=e["title"]) 
        for e in data["edges"]
    ]
    
    config = Config(
        width=800, 
        height=450, 
        directed=True, 
        physics=True, 
        hierarchical=False
    )
    
    return agraph(nodes=nodes, edges=edges, config=config, key=f"graph_{unique_id}")

# --- 5. Main UI Layout ---

st.title("🧙 Azarinth Healer Graph Chat")

# Display Chat History from session state
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this is an assistant message and has associated graph data, show it
        if "graph_data" in message and message["graph_data"]["nodes"]:
            with st.expander("Explore Graph Context", expanded=False):
                render_graph_viz(message["graph_data"], i)

# Handle New User Input
if prompt := st.chat_input("Ask about Ilea, classes, or monsters..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        # Processing Status
        with st.status("🛠️ System Analysis", expanded=True) as status:
            st.write("Extracting Entities...")
            entities = extract_entities(prompt)
            st.write(f"Detected: `{entities}`")
            
            st.write("Searching Knowledge Graph...")
            graph_data = get_graph_context(entities)
            
            # Debug view within the status
            if not graph_data["nodes"]:
                st.warning("No matching entities found in Neo4j.")
            else:
                st.success(f"Found {len(graph_data['nodes'])} related nodes.")
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

        # Generate LLM Answer
        response_text = generate_answer(prompt, graph_data["context_text"])
        st.markdown(response_text)
        
        # Show Graph Visualizer for THIS specific question
        if graph_data["nodes"]:
            with st.expander("Graph Subnetwork", expanded=True):
                render_graph_viz(graph_data, f"new_{len(st.session_state.messages)}")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "graph_data": graph_data
        })
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from streamlit_agraph import agraph, Node, Edge, Config

load_dotenv()

# --- 1. Configuration ---
st.set_page_config(layout="wide", page_title="Azarinth Healer Graph Chat")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Please check your .env file.")
        st.stop()
        
    st.session_state.llm = ChatOpenAI(
        model="gpt-3.5-turbo", # "gpt-3.5-turbo, gpt-4o"
        temperature=0,
        api_key=OPENAI_API_KEY
    )

# if "llm" not in st.session_state:
#     st.session_state.llm = ChatOllama(
#         model="llama3.1", 
#         temperature=0, 
#         base_url=OLLAMA_URL
#     )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Logic Functions ---

def extract_entities(question):
    prompt = ChatPromptTemplate.from_template(
        """You are a Named Entity Recognition (NER) expert. 
        Extract ALL meaningful entities (Persons, Locations, Monsters, Skills) from the question.
        Return ONLY a comma-separated list of names.
        Question: {question}
        Answer:"""
    )
    chain = prompt | st.session_state.llm
    response = chain.invoke({"question": question})
    return [e.strip() for e in response.content.split(',') if e.strip()]

def get_graph_context(entities):
    graph_data = {"nodes": {}, "edges": [], "context_text": []}
    if not entities:
        return graph_data

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
            source, target, rel = record['center'], record['neighbor'], record['rel']
            
            for n in [source, target]:
                if n['id'] not in graph_data["nodes"]:
                    graph_data["nodes"][n['id']] = n 

            rel_type = rel[1]
            rel_props = rel[3] if len(rel) > 3 else {}
            edge_tooltip = f"Type: {rel_type}\n" + "\n".join([f"{k}: {v}" for k, v in rel_props.items()])
            
            graph_data["edges"].append({
                "source": source['id'],
                "target": target['id'],
                "label": rel_type,
                "title": edge_tooltip
            })
            
            graph_data["context_text"].append(f"{source['id']} -[{rel_type}]-> {target['id']}")
            
    except Exception as e:
        st.error(f"Graph query error: {e}")
        
    return graph_data

def generate_answer(question, context_text):
    context_block = "\n".join(context_text) if context_text else "No specific graph data found."
    prompt = ChatPromptTemplate.from_template(
        "Context from Graph:\n{context}\n\nUser Question: {question}"
    )
    chain = prompt | st.session_state.llm
    response = chain.invoke({"context": context_block, "question": question})
    return response.content

# --- 4. UI Helper Functions ---

def render_graph_viz(data):
    """Renders the agraph without the 'key' argument."""
    if not data or not data["nodes"]:
        return
    
    nodes = []
    for node_id, props in data["nodes"].items():
        node_type = props.get('type', 'Unknown')
        color = "#FFD700" 
        if node_type == "Person": color = "#FF6B6B"
        elif node_type == "Skill": color = "#4ECDC4"
        
        tooltip = "Properties:\n" + "\n".join([f"{k}: {v}" for k, v in props.items()])
        nodes.append(Node(id=node_id, label=node_id, size=20, color=color, title=tooltip))
            
    edges = [Edge(source=e["source"], target=e["target"], label=e["label"], title=e["title"]) for e in data["edges"]]
    config = Config(width=800, height=450, directed=True, physics=True)
    
    # Removed key parameter to avoid TypeError
    return agraph(nodes=nodes, edges=edges, config=config)

# --- 5. Main UI Layout ---

st.title("🧙 Fantasy Graph Chat")

# Display History
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "graph_data" in message and message["graph_data"]["nodes"]:
            with st.expander("Explore Graph Context"):
                render_graph_viz(message["graph_data"])

# New Input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Graph Analysis Status
        status_box = st.status("🛠️ Analyzing Graph Data...", expanded=True)
        with status_box:
            st.write("Extracting entities...")
            entities = extract_entities(prompt)
            st.write(f"Detected: `{entities}`")
            
            st.write("Searching Knowledge Graph...")
            graph_data = get_graph_context(entities)
            
            if not graph_data["nodes"]:
                st.warning("No matching nodes found.")
            else:
                st.success(f"Found {len(graph_data['nodes'])} related nodes.")
            
            status_box.update(label="Analysis Complete", state="complete", expanded=False)

        # 2. Loading Animation for LLM Answer
        # This keeps the user informed while the LLM is thinking
        with st.spinner("Writing response based on graph context..."):
            response_text = generate_answer(prompt, graph_data["context_text"])
        
        # 3. Display Final Answer
        st.markdown(response_text)
        
        # 4. Display Graph
        if graph_data["nodes"]:
            with st.expander("Graph Subnetwork", expanded=True):
                render_graph_viz(graph_data)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text, 
            "graph_data": graph_data
        })
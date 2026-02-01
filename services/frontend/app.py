import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from streamlit_agraph import agraph, Node, Edge, Config

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Azarinth Graph Chat")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Initialize Connections ---
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

# --- Logic Functions ---

def extract_entities(question):
    """Step 2: Use LLM to extract potential entity names from the question."""
    prompt = ChatPromptTemplate.from_template(
        """Extract the key entities (people, places, creatures, skills) from the following question. 
        Return ONLY a comma-separated list of names. Do not add any other text.
        
        Question: {question}"""
    )
    chain = prompt | st.session_state.llm
    response = chain.invoke({"question": question})
    
    # Clean up response
    entities = [e.strip() for e in response.content.split(',') if e.strip()]
    return entities

def get_graph_context(entities):
    """Step 3 & 4: Fuzzy match entities and get one-hop subgraph."""
    graph_data = {"nodes": set(), "edges": [], "context_text": []}
    
    if not entities:
        return graph_data

    # This Cypher query does:
    # 1. Finds nodes where the ID contains the search term (case-insensitive fuzzy-ish match)
    # 2. Expands 1 hop (neighbors)
    # 3. Returns the paths
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
            rel = record['rel']
            
            # Format text context for LLM
            # "Ilea (Person) -[ENCOUNTERED]-> Drake (Creature)"
            context_line = f"{source['id']} ({source.get('type','Node')}) -[{rel['type']}]-> {target['id']} ({target.get('type','Node')})"
            graph_data["context_text"].append(context_line)
            
            # Prepare Visual Nodes (Streamlit AGraph)
            # We use the ID as the unique key
            graph_data["nodes"].add((source['id'], source.get('type', 'Unknown')))
            graph_data["nodes"].add((target['id'], target.get('type', 'Unknown')))
            
            # Prepare Visual Edges
            graph_data["edges"].append({
                "source": source['id'],
                "target": target['id'],
                "label": rel['type']
            })
            
    except Exception as e:
        st.error(f"Graph query error: {e}")
        
    return graph_data

def generate_answer(question, context_text):
    """Step 5: Answer question using the retrieved context."""
    if not context_text:
        context_block = "No specific graph data found."
    else:
        context_block = "\n".join(context_text)
        
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant with access to a Knowledge Graph about Azarinth Healer.
        Use the following context from the graph to answer the user's question.
        If the context doesn't have the answer, say so, but try to infer from what is there.
        
        Graph Context:
        {context}
        
        User Question: {question}
        """
    )
    chain = prompt | st.session_state.llm
    response = chain.invoke({"context": context_block, "question": question})
    return response.content

# --- UI Layout ---

st.title("🧙 Azarinth Healer Graph Chat")
col1, col2 = st.columns([1, 1])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_graph_data" not in st.session_state:
    st.session_state.last_graph_data = None

# Left Column: Chat Interface
with col1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Step 1: User Input
    if prompt := st.chat_input("Ask about Ilea, classes, or monsters..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.status("Thinking...", expanded=True)
            
            # Step 2: Extract Entities
            status_container.write("🔍 Extracting entities...")
            entities = extract_entities(prompt)
            status_container.write(f"identified: {entities}")
            
            # Step 3 & 4: Query Graph
            status_container.write("🕸️ Querying Knowledge Graph...")
            graph_data = get_graph_context(entities)
            st.session_state.last_graph_data = graph_data # Save for visualizer
            
            # Step 5: Generate Answer
            status_container.write("💡 Generating answer...")
            response_text = generate_answer(prompt, graph_data["context_text"])
            
            status_container.update(label="Complete!", state="complete", expanded=False)
            st.markdown(response_text)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Right Column: Graph Visualization (Step 6)
with col2:
    st.subheader("Graph Context Visualizer")
    
    if st.session_state.last_graph_data and st.session_state.last_graph_data["nodes"]:
        data = st.session_state.last_graph_data
        
        # Convert to agraph objects
        nodes = []
        for node_id, node_type in data["nodes"]:
            # Color code based on type for better visuals
            color = "#FFD700" # Default Gold
            if node_type == "Person": color = "#FF6B6B" # Red
            elif node_type == "Skill": color = "#4ECDC4" # Teal
            elif node_type == "Creature": color = "#C7F464" # Lime
            elif node_type == "Location": color = "#556270" # Grey
            
            nodes.append(Node(
                id=node_id, 
                label=node_id, 
                size=25, 
                color=color,
                title=f"Type: {node_type}" # Tooltip
            ))
            
        edges = []
        for edge in data["edges"]:
            edges.append(Edge(
                source=edge["source"], 
                target=edge["target"], 
                label=edge["label"]
            ))
            
        config = Config(
            width=600,
            height=600,
            directed=True, 
            physics=True, 
            hierarchical=False,
        )
        
        # Render the graph
        return_value = agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.info("Ask a question to see the relevant graph network here.")
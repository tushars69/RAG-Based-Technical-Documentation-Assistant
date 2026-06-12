import streamlit as st
import requests

# ── Configuration ────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Docs Assistant", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (Neon/Electric Blue Aesthetic) ────────────────────────────
st.markdown("""
<style>
    /* Neon accents for inputs and buttons */
    .stTextInput>div>div>input {
        border: 1px solid #005f73;
    }
    .stTextInput>div>div>input:focus {
        border-color: #00e5ff;
        box-shadow: 0 0 8px #00e5ff;
    }
    .stButton>button {
        background-color: transparent;
        border: 1px solid #00e5ff;
        color: #00e5ff;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00e5ff;
        color: #121212;
        box-shadow: 0 0 10px #00e5ff;
    }
    /* Customizing the chat bubbles */
    .stChatMessage {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #00e5ff;
        background-color: rgba(0, 229, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ── State Initialization ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None


# ── Sidebar: Control Panel & Ingestion ───────────────────────────────────
with st.sidebar:
    st.title("⚡ System Control")
    
    # Ping backend for status
    try:
        res = requests.get(f"{API_URL}/")
        if res.status_code == 200:
            data = res.json()
            st.success("Backend Online")
        else:
            st.error("Backend Offline")
    except requests.exceptions.ConnectionError:
        st.error("Backend Unreachable. Is FastAPI running?")

    st.divider()

    # Document Ingestion UI
    st.subheader("Ingest New Data")
    ingest_url = st.text_input("Enter Documentation URL:")
    if st.button("Index URL") and ingest_url:
        with st.spinner("Scraping and Chunking..."):
            try:
                res = requests.post(f"{API_URL}/ingest", json={"urls": [ingest_url]})
                if res.status_code == 200:
                    st.success("Successfully ingested!")
                    st.rerun()
                else:
                    st.error(f"Failed: {res.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── Main Chat Interface ──────────────────────────────────────────────────
st.title("Technical Docs Assistant")
st.caption("Powered by LangGraph, ChromaDB, and FastAPI")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 **{s.get('title', 'Doc')}** (Score: {s.get('similarity_score', 'N/A')})")
                    st.caption(f"🔗 {s.get('source', '')}")

# Chat Input
if prompt := st.chat_input("Ask a technical question..."):
    
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Analyzing query and searching vectors..."):
            payload = {"question": prompt, "max_retries": 2}
            if st.session_state.session_id:
                payload["session_id"] = st.session_state.session_id

            try:
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update session state with returned ID
                    st.session_state.session_id = data["session_id"]
                    
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    # Display the final answer
                    message_placeholder.markdown(answer)
                    

                    # Save assistant response to memory
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })

                else:
                    st.error(f"Pipeline Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                
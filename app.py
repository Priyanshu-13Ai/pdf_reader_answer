# app.py

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Import RAG processing helpers
from rag import (
    extract_text_from_pdf,
    split_documents,
    get_embedding_model,
    create_vector_store,
    get_groq_llm,
    answer_question
)

# Page configuration
st.set_page_config(
    page_title="InsightPDF - Intelligent PDF Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium custom styling block to satisfy design guidelines (Glassmorphism, custom typography, dark mode enforcement)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Apply typography and light mode styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.8);
        backdrop-filter: blur(8px);
    }
    
    /* Sidebar Styling - Light soft gray */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Header Gradient Title */
    .title-gradient {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Chat Message bubble custom styling - Clean white cards */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        color: #0f172a !important;
    }
    
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #f8fafc !important;
        border-color: #e2e8f0;
    }
    
    /* Visual step progress boxes - Light theme */
    .step-box {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        font-size: 0.85rem;
        color: #475569;
    }
    
    .step-box.completed {
        border-color: #a7f3d0;
        background: #ecfdf5;
        color: #047857;
    }
    
    .step-icon {
        margin-right: 10px;
        font-size: 1.1rem;
    }
    
    /* Badge styling */
    .badge {
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }
    
    /* Sidebar stats styling */
    .sidebar-stats {
        padding: 12px;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 10px;
        font-size: 0.85rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# 1. State Management Init
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_pdf_name" not in st.session_state:
    st.session_state.processed_pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stats" not in st.session_state:
    st.session_state.stats = {}

# Fetch Groq API Key automatically from environment
groq_api_key = os.getenv("GROQ_API_KEY", "")

# 2. Sidebar Layout
with st.sidebar:
    st.markdown("## ⚙️ InsightPDF Console")
    
    # PDF File Uploader (User simply uploads PDF)
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    # Clear conversation button
    if st.button("🔄 New Chat / Reset", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")
    
    # Expandable Advanced RAG Configuration Parameters
    with st.expander("🛠️ Advanced Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)
        k_val = st.slider("Retrieve Chunks (k)", min_value=1, max_value=10, value=3)

# 3. Main Area Header Setup
st.markdown('<div class="title-gradient">InsightPDF 💬</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interact with your PDF documents via a modern, semantic RAG chatbot</div>', unsafe_allow_html=True)

# API key warning message if key is not configured in .env
if not groq_api_key.strip():
    st.warning("⚠️ **Groq API Key Missing:** Please add `GROQ_API_KEY` in your `.env` file to enable AI answers.")

# 4. Pipeline Execution & Status Panel
if uploaded_file is not None:
    # Compound key to detect document name or parameter updates
    settings_key = f"{uploaded_file.name}_{chunk_size}_{chunk_overlap}"
    
    if st.session_state.processed_pdf_name != settings_key:
        with st.sidebar:
            st.markdown("### ⚙️ Pipeline Process Status")
            
            # Step 1: Text extraction
            step1 = st.empty()
            step1.markdown('<div class="step-box"><span class="step-icon">⏳</span>Reading PDF File...</div>', unsafe_allow_html=True)
            try:
                documents = extract_text_from_pdf(uploaded_file, uploaded_file.name)
                step1.markdown('<div class="step-box completed"><span class="step-icon">✅</span>PDF Text Extracted</div>', unsafe_allow_html=True)
            except Exception as e:
                step1.markdown(f'<div class="step-box"><span class="step-icon">❌</span>Extraction Error: {str(e)}</div>', unsafe_allow_html=True)
                st.stop()
            
            # Step 2: Split text
            step2 = st.empty()
            step2.markdown('<div class="step-box"><span class="step-icon">⏳</span>Splitting Text into Chunks...</div>', unsafe_allow_html=True)
            try:
                chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                step2.markdown(f'<div class="step-box completed"><span class="step-icon">✅</span>Split into {len(chunks)} Chunks</div>', unsafe_allow_html=True)
            except Exception as e:
                step2.markdown(f'<div class="step-box"><span class="step-icon">❌</span>Chunking Error: {str(e)}</div>', unsafe_allow_html=True)
                st.stop()
            
            # Step 3: Embeddings generation & Chroma load
            step3 = st.empty()
            step3.markdown('<div class="step-box"><span class="step-icon">⏳</span>Generating Embeddings & Storing in ChromaDB...</div>', unsafe_allow_html=True)
            try:
                embedding_model = get_embedding_model()
                vector_store = create_vector_store(chunks, embedding_model)
                step3.markdown('<div class="step-box completed"><span class="step-icon">✅</span>ChromaDB Indexing Complete</div>', unsafe_allow_html=True)
                
                # Save settings/state
                st.session_state.vector_store = vector_store
                st.session_state.processed_pdf_name = settings_key
                st.session_state.chat_history = []  # Reset chat history for new uploads
                st.session_state.stats = {
                    "filename": uploaded_file.name,
                    "pages": len(documents),
                    "chunks": len(chunks)
                }
                
                st.sidebar.success("🎉 RAG pipeline ready!")
            except Exception as e:
                step3.markdown(f'<div class="step-box"><span class="step-icon">❌</span>Vector Store Error: {str(e)}</div>', unsafe_allow_html=True)
                st.stop()
                
    # Display Document stats in sidebar once processed
    if st.session_state.stats:
        with st.sidebar:
            st.markdown("### 📊 Document Metadata")
            st.markdown(f"""
            <div class="sidebar-stats">
                <strong>📄 Name:</strong> {st.session_state.stats['filename']}<br>
                <strong>📖 Pages:</strong> {st.session_state.stats['pages']}<br>
                <strong>🧩 Chunks:</strong> {st.session_state.stats['chunks']}
            </div>
            """, unsafe_allow_html=True)
else:
    # Clean workspace session states when document is removed
    st.session_state.vector_store = None
    st.session_state.processed_pdf_name = None
    st.session_state.chat_history = []
    st.session_state.stats = {}
    
    st.markdown(
        """
        <div style="background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 16px; padding: 40px; text-align: center; margin-top: 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h3 style="color: #0f172a; margin-bottom: 8px;">👋 Welcome to InsightPDF!</h3>
            <p style="color: #475569; font-size: 0.95rem; margin-bottom: 20px;">Upload a PDF document in the sidebar to automatically index it, then start chatting with your document instantly.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 5. Question & Answer Interface
if st.session_state.vector_store is not None:
    # Render existing conversation history (ChatGPT style)
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            
            # Citation badge and context expander
            if chat["sources"]:
                pages = sorted(list(set([src['page'] for src in chat['sources']])))
                pages_str = ", ".join([f"Page {p}" for p in pages])
                st.markdown(f'<span class="badge">📍 Source Citation(s): {pages_str}</span>', unsafe_allow_html=True)
                
                with st.expander("🔍 Show Source Passages", expanded=False):
                    for i, src in enumerate(chat["sources"]):
                        st.markdown(f"**Chunk {i+1} | Page {src['page']} | Document: {src['source']}**")
                        st.info(src["content"])

    # Chat input box pinned at bottom
    if prompt := st.chat_input("Ask a question about your PDF..."):
        if not groq_api_key.strip():
            st.error("🔑 Groq API Key is not set. Please add it to your `.env` file.")
        else:
            # Display user query instantly
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document content..."):
                    try:
                        # Setup Groq LLM
                        llm = get_groq_llm(api_key=groq_api_key)
                        # Query RAG
                        result = answer_question(prompt, st.session_state.vector_store, llm, k=k_val)
                        
                        st.markdown(result["answer"])
                        
                        # Display citation source badge
                        if result["sources"]:
                            pages = sorted(list(set([src['page'] for src in result['sources']])))
                            pages_str = ", ".join([f"Page {p}" for p in pages])
                            st.markdown(f'<span class="badge">📍 Source Citation(s): {pages_str}</span>', unsafe_allow_html=True)
                            
                            with st.expander("🔍 Show Source Passages", expanded=False):
                                for i, src in enumerate(result["sources"]):
                                    st.markdown(f"**Chunk {i+1} | Page {src['page']} | Document: {src['source']}**")
                                    st.info(src["content"])
                                    
                        # Store in chat history
                        st.session_state.chat_history.append({
                            "question": prompt,
                            "answer": result["answer"],
                            "sources": result["sources"]
                        })
                        
                        # Rerun to sync chat history layout cleanly
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating response: {str(e)}")

# rag.py

import fitz  # PyMuPDF
from langchain_core.documents import Document

# Safe imports with fallback for older/newer langchain package versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from prompts import SYSTEM_PROMPT


def extract_text_from_pdf(pdf_source, file_name: str = None) -> list[Document]:
    """
    Extracts text page-by-page from a PDF file (path or file-like object)
    and wraps it in LangChain Document objects with page metadata.
    
    This keeps page numbers (1-indexed) intact for accurate citations.
    """
    if isinstance(pdf_source, str):
        # The input is a file path
        doc = fitz.open(pdf_source)
        source_name = file_name or pdf_source
    else:
        # The input is a file-like bytes object (e.g. from streamlit file_uploader)
        if hasattr(pdf_source, 'seek'):
            pdf_source.seek(0)
        pdf_bytes = pdf_source.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        source_name = file_name or getattr(pdf_source, 'name', 'Uploaded_PDF')

    documents = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        cleaned_text = text.strip()
        # Only add pages that contain text
        if cleaned_text:
            metadata = {
                "source": source_name,
                "page": page_num + 1  # Store 1-indexed page number
            }
            documents.append(Document(page_content=cleaned_text, metadata=metadata))
    
    return documents


def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits list of Documents into smaller chunks using RecursiveCharacterTextSplitter.
    Chunk size and overlap are configurable parameters.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads HuggingFaceEmbeddings model for text embedding generation.
    Default model is 'all-MiniLM-L6-v2', which is highly performant and lightweight.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )


def create_vector_store(chunks: list[Document], embedding_model):
    """
    Creates an ephemeral (in-memory) Chroma vector database from chunks and embedding model.
    Using an in-memory client ensures clean database states per upload.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )


def get_groq_llm(api_key: str, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.0):
    """
    Initializes the Groq Chat model via langchain-groq.
    Model name defaults to llama-3.1-8b-instant for fast, cost-efficient, high-quality responses.
    """
    return ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )


def answer_question(query: str, vector_store, llm, k: int = 3) -> dict:
    """
    Core RAG Pipeline:
    1. Retrieve the top 'k' most similar chunks from Chroma.
    2. Format chunks into context.
    3. Pass query + context to Llama 3 on Groq.
    4. Return the generated answer and references (sources).
    """
    # 1. Similarity search to retrieve relevant context
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # 2. Format context and citations
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved_docs):
        page_num = doc.metadata.get("page", "Unknown")
        source_name = doc.metadata.get("source", "Document")
        
        context_parts.append(f"--- Chunk {i+1} (Page {page_num}) ---\n{doc.page_content}")
        sources.append({
            "page": page_num,
            "source": source_name,
            "content": doc.page_content
        })
    
    context_str = "\n\n".join(context_parts)
    
    # 3. Formulate the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    # 4. Invoke LLM chain
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": query})
    
    return {
        "answer": response.content.strip(),
        "sources": sources
    }

# test_rag_pipeline.py

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load prompts and RAG module
from rag import (
    extract_text_from_pdf,
    split_documents,
    get_embedding_model,
    create_vector_store,
    get_groq_llm,
    answer_question
)

# 1. Create a dummy PDF programmatically
print("Creating dummy PDF file for testing...")
pdf_path = "test_sample.pdf"
doc = fitz.open()
page1 = doc.new_page()
page1.insert_text((50, 50), "Retrieval-Augmented Generation (RAG) is a technique that uses context from documents to answer questions.")
page1.insert_text((50, 100), "The Groq Llama 3.1 model is used as the generator LLM for this chatbot.")
page1.insert_text((50, 150), "This is Page 1 of the document.")

page2 = doc.new_page()
page2.insert_text((50, 50), "ChromaDB is a database built for storing and querying vector embeddings.")
page2.insert_text((50, 100), "HuggingFace Embeddings are used to convert text to vectors.")
page2.insert_text((50, 150), "This is Page 2 of the document.")

doc.save(pdf_path)
doc.close()
print(f"Dummy PDF '{pdf_path}' created successfully.\n")

# 2. Extract Text
print("Step 1: Extracting text from PDF...")
documents = extract_text_from_pdf(pdf_path)
for doc_obj in documents:
    print(f"  - Extracted Page {doc_obj.metadata['page']} (len={len(doc_obj.page_content)} chars)")
print()

# 3. Split Documents
print("Step 2: Splitting text into chunks...")
chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)
print(f"  - Generated {len(chunks)} chunks.")
for i, chunk in enumerate(chunks):
    print(f"    * Chunk {i+1} (Page {chunk.metadata['page']}): {chunk.page_content[:60]}...")
print()

# 4. Generate Embeddings & Chroma DB
print("Step 3: Generating embeddings and initializing Chroma DB...")
embedding_model = get_embedding_model()
vector_store = create_vector_store(chunks, embedding_model)
print("  - Vector store created successfully in memory.\n")

# 5. Query similarity search
query = "What is ChromaDB used for?"
print(f"Step 4: Performing similarity search for query: '{query}'...")
retrieved = vector_store.similarity_search(query, k=1)
if retrieved:
    print(f"  - Matches found! Top match from Page {retrieved[0].metadata['page']}:")
    print(f"    \"{retrieved[0].page_content}\"")
else:
    print("  - No match found.")
print()

# 6. LLM query test (if GROQ_API_KEY is available)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "")
if api_key:
    print("Step 5: Groq API Key found. Testing end-to-end RAG answer generation...")
    try:
        llm = get_groq_llm(api_key=api_key)
        res = answer_question(query, vector_store, llm, k=2)
        print("\n=== AI Answer ===")
        print(res["answer"])
        print("Citations:")
        for src in res["sources"]:
            print(f"  * Page {src['page']}: {src['content'][:80]}...")
        print("=================\n")
        
        # Test out-of-context question
        ooc_query = "What is the capital of France?"
        print(f"Step 6: Testing out-of-context query: '{ooc_query}'...")
        res_ooc = answer_question(ooc_query, vector_store, llm, k=2)
        print("\n=== AI Answer (OOC) ===")
        print(res_ooc["answer"])
        print("=================\n")
        
    except Exception as e:
        print(f"  - Groq API Call failed: {e}")
else:
    print("Step 5: GROQ_API_KEY not found in environment. Skipping LLM execution test.")

# Clean up
if os.path.exists(pdf_path):
    os.remove(pdf_path)
    print(f"Cleaned up test file '{pdf_path}'.")

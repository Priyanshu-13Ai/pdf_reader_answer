# InsightPDF 🤖💬

InsightPDF is a beginner-friendly, modular **PDF Chatbot** application that implements a complete **Retrieval-Augmented Generation (RAG)** pipeline. Users can upload a PDF and chat with it in a modern, conversational, ChatGPT-like interface.

The application automatically loads your Groq API key from your environment configuration, providing a seamless user experience.

---

## 💡 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique used to optimize the output of a Large Language Model by referencing an authoritative knowledge base outside of its training data sources before generating a response. 

Normally, LLMs are limited to the data they were trained on. RAG addresses this by:
1. **Retrieving** relevant information from a specific document library (e.g., your uploaded PDF).
2. **Augmenting** the user's query by injecting the retrieved chunks as context.
3. **Generating** an accurate, grounded answer using the LLM.

---

## 🛠️ Tech Stack & Technologies Used

* **Python**: Core programming language.
* **LangChain**: The leading orchestration framework for LLM-powered applications.
* **ChromaDB**: An open-source, lightweight vector database designed for AI developers.
* **HuggingFace Embeddings**: Specifically the `all-MiniLM-L6-v2` sentence-transformer model to generate dense vector embeddings of text.
* **Groq (Llama 3)**: High-speed inference engine running Llama 3 for response generation.
* **Streamlit**: Fast, beautiful python-native framework for building interactive user interfaces.
* **PyMuPDF (fitz)**: Super-fast, lightweight library to parse and extract text from PDFs.
* **python-dotenv**: To load credentials securely from a local `.env` configuration.

---

## 📂 Project Structure

```text
pdf-chatbot/
│
├── app.py             # Streamlit web interface & application logic (ChatGPT style)
├── rag.py             # RAG pipeline logic (extraction, chunking, database, QA)
├── prompts.py         # Prompt engineering templates constraining the LLM
├── requirements.txt   # Python dependency list
├── .env               # Private environment variables containing your GROQ_API_KEY
├── .env.example       # Example configuration for environment variables
└── README.md          # Project guide (this document)
```

---

## 🚀 Installation & Setup

Follow these steps to set up and run the project locally on your machine:

### 1. Set Up a Virtual Environment (Recommended)
Open your terminal in the project directory and create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies
Run the following command to install all the necessary packages:
```bash
pip install -r requirements.txt
```

### 3. Configure the Groq API Key
1. Open the `.env` file in the project folder.
2. Insert your Groq API Key:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_goes_here
   ```
   *(You can obtain a key for free from the [Groq Console](https://console.groq.com/))*

---

## 🏃 How to Run the Project

Run the Streamlit application using the command below:
```bash
streamlit run app.py
```

The application will open automatically in your default web browser (typically at `http://localhost:8501`).

---

## 🧠 Understanding the RAG Pipeline in code

Here is what happens step-by-step when you upload a PDF and ask a question:

1. **PDF Upload & Text Extraction (`rag.py` -> `extract_text_from_pdf`)**
   - The PDF is parsed page-by-page using `fitz` (PyMuPDF).
   - Text is formatted into LangChain `Document` objects, storing the page numbers as metadata to enable accurate source citation.

2. **Text Chunking (`rag.py` -> `split_documents`)**
   - Since documents can be extremely long, we split the text into manageable portions (chunks) using `RecursiveCharacterTextSplitter`.
   - The default configuration uses a chunk size of 1000 characters with an overlap of 200 characters to prevent loss of context across splits.

3. **Generating Embeddings & Storing in ChromaDB (`rag.py` -> `get_embedding_model` & `create_vector_store`)**
   - The chunks are sent to a HuggingFace sentence transformer (`all-MiniLM-L6-v2`) which translates text semantics into a 384-dimensional vector.
   - These vectors (embeddings) are indexed inside an in-memory ChromaDB vector store.

4. **Asking Questions & Retrieving Context (`rag.py` -> `answer_question`)**
   - When you enter a question in the bottom chat bar, ChromaDB retrieves the top `k` text chunks that have the highest semantic similarity to the question.
   - The page numbers of these chunks are collected for citations.

5. **LLM Generation (`rag.py` -> `answer_question` & `prompts.py`)**
   - The retrieved text chunks and the user's question are inserted into a template prompt.
   - The prompt explicitly instructs the Llama 3 model (hosted on Groq) to **only** answer based on the provided context, preventing hallucinated responses.
   - If the answer is absent from the text, Llama 3 responds with: *"The information is not available in the uploaded PDF."*

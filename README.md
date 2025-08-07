# 💬 RAG Chatbot (LLMs + Embeddings)

This project is a Retrieval-Augmented Generation (RAG) chatbot built as part of the technical assignment for the **AI Engineer role at Amlgo Labs**. It answers user queries based on a custom `.docx` document using local models and vector search.

---

## Features

- Semantic search using **`all-MiniLM-L6-v2`** sentence embeddings
- Answer generation using **`google/flan-t5-base`** (runs locally)
- Document chunking and vector storage with **ChromaDB**
- Full local pipeline — no internet/API calls needed
- Interactive UI via **Streamlit**

---

## How It Works

1. **Chunking**  
   The document is split into overlapping chunks (~1000 characters) to preserve context.

2. **Embedding + Vector Store**  
   Each chunk is embedded and stored in a persistent ChromaDB collection.

3. **Query + Retrieval**  
   User queries are embedded and matched against the chunk vectors to retrieve relevant context.

4. **LLM Response**  
   The context + query is passed to `flan-t5-base` to generate a grounded answer.

---

## Demo

### Screenshot  
<img width="1918" height="910" alt="Failed_Response_1" src="https://github.com/user-attachments/assets/7413b7ca-c68f-41da-8a0e-21f89771c30b" />


<img width="1918" height="910" alt="Success_Response_1" src="https://github.com/user-attachments/assets/52ca2ee7-28e8-48d7-8c63-892d0c527919" />




### Video Demo  
🎥 [Watch full demo]([media/chatbot-demo.mp4](https://github.com/user-attachments/assets/ed3b7d89-4aa0-4911-bced-5372ee039ebf))

---

## Run Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Chunk the document
python chunk_docx.py

# Step 3: Generate embeddings and store in vector DB
python src/embed_and_store.py

# Step 4: Launch the chatbot UI
streamlit run app.py









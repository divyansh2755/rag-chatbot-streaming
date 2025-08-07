import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from src.llm import call_llm  # Local Transformers model

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector DB (new PersistentClient API)
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_collection("rag_collection")

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot (Local)", layout="wide")
st.title("💬 RAG Chatbot")
st.markdown("Ask a question based on the document...")

query = st.text_input("Your question")

if query:
    # Embed query
    query_embedding = embedding_model.encode(query).tolist()

    # Search top matching document chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results['documents'][0]
    context = "\n\n".join(documents)

    # Call local model
    with st.spinner("Thinking..."):
        response = call_llm(context, query)

    # Show output
    st.markdown("### Answer")
    st.write(response)

    with st.expander("Retrieved Context"):
        st.write(context)

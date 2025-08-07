from sentence_transformers import SentenceTransformer
import chromadb
import json

# Load chunks
with open("chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
ids = [str(chunk["id"]) for chunk in chunks]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts).tolist()

# Connect to persistent Chroma DB (new API)
client = chromadb.PersistentClient(path="vectordb")

# Create or get collection
collection = client.get_or_create_collection(name="rag_collection")

# Add data if not already present
if collection.count() == 0:
    collection.add(documents=texts, ids=ids, embeddings=embeddings)
    print("Chroma vector DB created at /vectordb")
else:
    print("Chroma vector DB already exists at /vectordb")

print(f"Total records in collection: {collection.count()}")

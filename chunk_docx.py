from docx import Document
import nltk
import json
from pathlib import Path

nltk.download('punkt')
nltk.download('punkt_tab')

doc_path = Path("data/AI Training Document.docx")
output_path = Path("chunks/chunks.json")

doc = Document(doc_path)
text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
sentences = nltk.sent_tokenize(text)

chunks = []
chunk, chunk_words, chunk_id = "", 0, 0
for sent in sentences:
    words = len(sent.split())
    if chunk_words + words <= 300:
        chunk += " " + sent
        chunk_words += words
    else:
        if chunk_words >= 100:
            chunks.append({"id": chunk_id, "text": chunk.strip()})
            chunk_id += 1
        chunk = sent
        chunk_words = words
if chunk and chunk_words >= 100:
    chunks.append({"id": chunk_id, "text": chunk.strip()})

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def call_llm(context, query):
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.
Only use the information in the context. Be detailed but concise.

Context:
{context}

Question:
{query}

Answer:
"""
    # Tokenize and truncate properly
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate a longer response with better decoding control
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,           # More room for detailed answers
        do_sample=True,               # Enable sampling for diversity
        temperature=0.7,              # Balance creativity & determinism
        top_k=50,                     # Top-k sampling (limits to top 50 predictions)
        top_p=0.95,                   # Nucleus sampling
        num_return_sequences=1        # Single response
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

def get_text_chunks(text, max_chunk_size=1024):
    """Splits text into chunks of a maximum size, trying to respect sentence boundaries."""
    # Naive sentence splitting
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_index(documents, model_name='all-MiniLM-L6-v2', index_path='faiss_index.bin', data_path='documents.npy'):
    """Creates and saves a FAISS index for a list of documents."""
    model = SentenceTransformer(model_name)
    
    all_chunks = []
    for doc in documents:
        all_chunks.extend(get_text_chunks(doc))

    embeddings = model.encode(all_chunks, convert_to_tensor=False)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    faiss.write_index(index, index_path)
    np.save(data_path, np.array(all_chunks, dtype=object))
    
    print(f"Index created with {len(all_chunks)} chunks and saved to {index_path}")
    print(f"Documents saved to {data_path}")

if __name__ == '__main__':
    document_files = [
        "healthy_maize_remedy.txt",
        "maize_phosphorus_deficiency_remedy.txt",
        "comic_relief.txt",
        "docs/article1.md",
        "docs/article2.md"
    ]
    
    documents = []
    for file_path in document_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")
            
    create_index(documents)

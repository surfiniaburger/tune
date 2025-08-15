
# create_index.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

from database import init_db, get_db_connection, INDEX_FILE, DB_FILE, delete_database_and_index
from security import encrypt_data

# Use a CLIP model that can handle both text and images
MODEL_NAME = 'clip-ViT-B-32'

def create_initial_index(documents_dict):
    """
    Creates an initial encrypted, persistent index from a dictionary of text documents.
    This will delete any existing database to ensure a clean start.
    """
    print("Performing a clean rebuild of the knowledge base...")
    delete_database_and_index()
    init_db()

    conn = get_db_connection()
    cursor = conn.cursor()
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    all_embeddings = []

    for name, content in documents_dict.items():
        # Add document to documents table
        cursor.execute("INSERT INTO documents (name) VALUES (?)", (name,))
        doc_id = cursor.lastrowid

        # For initial docs, we treat the whole content as one chunk
        chunk_text = content
        all_chunks.append((doc_id, 'text', encrypt_data(chunk_text.encode('utf-8')), 1))
        
        # Create text embedding
        text_embedding = model.encode([chunk_text])
        all_embeddings.append(text_embedding)

    # Batch insert chunks
    cursor.executemany(
        "INSERT INTO chunks (doc_id, content_type, encrypted_content, page_num) VALUES (?, ?, ?, ?)",
        all_chunks
    )
    conn.commit()
    conn.close()

    if not all_embeddings:
        print("No content to index.")
        return

    # Create and save the FAISS index
    embeddings_np = np.vstack(all_embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, INDEX_FILE)

    print(f"Initial encrypted index created with {len(all_chunks)} chunks.")
    print(f"Database: {DB_FILE}, FAISS Index: {INDEX_FILE}")



if __name__ == '__main__':
    document_files = ["healthy_maize_remedy.txt", "maize_phosphorus_deficiency_remedy.txt", "comic_relief.txt"]
    documents_content = []
    for file_path in document_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents_content.append(f.read())
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")

    create_initial_index(documents_content)
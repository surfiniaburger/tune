import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
import pickle

# Import our new modules
from database import init_db, get_db_connection, INDEX_FILE, DB_FILE
from security import encrypt_data

def get_text_chunks(text, max_chunk_size=1024):
    """Splits text into chunks."""
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

def create_index(documents, model_name='all-MiniLM-L6-v2'):
    """Creates an encrypted, persistent index for a list of documents."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    model = SentenceTransformer(model_name)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(get_text_chunks(doc))

    embeddings = model.encode(all_chunks, convert_to_tensor=False)
    
    # Store encrypted chunks and embeddings
    for chunk, embedding in zip(all_chunks, embeddings):
        encrypted_chunk = encrypt_data(chunk.encode('utf-8'))
        
        # We need to serialize the numpy array to bytes before encryption
        embedding_bytes = pickle.dumps(embedding)
        encrypted_embedding = encrypt_data(embedding_bytes)
        
        cursor.execute(
            "INSERT INTO documents (encrypted_chunk, encrypted_embedding) VALUES (?, ?)",
            (encrypted_chunk, encrypted_embedding)
        )
    
    conn.commit()
    conn.close()

    # Create and save the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, INDEX_FILE)

    print(f"Encrypted index created with {len(all_chunks)} chunks.")
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
            
    create_index(documents_content)
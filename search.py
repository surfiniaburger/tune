import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Import our new modules
from database import get_db_connection, INDEX_FILE
from security import decrypt_data

def search(query, model_name='all-MiniLM-L6-v2', k=1):
    """Searches the FAISS index and retrieves the decrypted document."""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(INDEX_FILE)
    
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    results = []
    conn = get_db_connection()
    for i, doc_id in enumerate(indices[0]):
        if doc_id != -1:
            # FAISS gives us a 0-based index, which is 1 less than our 1-based SQL id
            sql_id = int(doc_id) + 1
            
            # Retrieve the encrypted chunk from SQLite
            doc_record = conn.execute('SELECT encrypted_chunk FROM documents WHERE id = ?', (sql_id,)).fetchone()
            
            if doc_record:
                decrypted_chunk = decrypt_data(doc_record['encrypted_chunk']).decode('utf-8')
                results.append({
                    'distance': distances[0][i],
                    'document': decrypted_chunk
                })
    conn.close()
    return results
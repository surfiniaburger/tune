# search.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

from database import get_db_connection, INDEX_FILE, check_if_indexed
from security import decrypt_data

MODEL_NAME = 'clip-ViT-B-32'

def search(query, k=1):
    """
    Searches the multimodal FAISS index. The query can be text, and the result can be text or an image.
    """
    if not check_if_indexed():
        return []

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)

    # Create an embedding for the text query
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    results = []
    conn = get_db_connection()
    for i, faiss_id in enumerate(indices[0]):
        if faiss_id != -1:
            # The faiss_id is the row number, which corresponds to the chunk's primary key 'id'
            sql_id = int(faiss_id) + 1
            
            chunk_record = conn.execute('SELECT * FROM chunks WHERE id = ?', (sql_id,)).fetchone()
            
            if chunk_record:
                content_type = chunk_record['content_type']
                decrypted_content_bytes = decrypt_data(chunk_record['encrypted_content'])
                
                # Prepare content based on its type
                if content_type == 'text':
                    content = decrypted_content_bytes.decode('utf-8')
                elif content_type == 'image':
                    content = Image.open(io.BytesIO(decrypted_content_bytes))
                
                results.append({
                    'distance': distances[0][i],
                    'content': content,
                    'type': content_type,
                    'page': chunk_record['page_num']
                })
    conn.close()
    return results
# ingest_document.py

import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
import os

from database import get_db_connection, INDEX_FILE
from security import encrypt_data

MODEL_NAME = 'clip-ViT-B-32'

def ingest_pdf(file_path, file_name):
    """Parses a PDF, encrypts its content (text+images), and adds it to the database and FAISS index."""
    print(f"Starting ingestion for: {file_name}")
    model = SentenceTransformer(MODEL_NAME)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Add document to documents table, or get its ID if it exists
    try:
        cursor.execute("INSERT INTO documents (name) VALUES (?)", (file_name,))
        doc_id = cursor.lastrowid
    except conn.IntegrityError:
        print("Document already exists in DB. Skipping doc table insert.")
        doc_id = cursor.execute("SELECT id FROM documents WHERE name=?", (file_name,)).fetchone()['id']

    doc = fitz.open(file_path)
    new_embeddings = []
    
    # Load existing FAISS index or create a new one
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        # Get dimension from the model if index is new
        dimension = model.encode(["test"]).shape[1]
        index = faiss.IndexFlatL2(dimension)

    for page_num, page in enumerate(doc):
        # 1. Process Text
        text = page.get_text()
        if text.strip():
            encrypted_text = encrypt_data(text.encode('utf-8'))
            cursor.execute(
                "INSERT INTO chunks (doc_id, content_type, encrypted_content, page_num) VALUES (?, ?, ?, ?)",
                (doc_id, 'text', encrypted_text, page_num + 1)
            )
            text_embedding = model.encode([text])
            new_embeddings.append(text_embedding)

        # 2. Process Images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            encrypted_image = encrypt_data(image_bytes)
            cursor.execute(
                "INSERT INTO chunks (doc_id, content_type, encrypted_content, page_num) VALUES (?, ?, ?, ?)",
                (doc_id, 'image', encrypted_image, page_num + 1)
            )
            pil_image = Image.open(io.BytesIO(image_bytes))
            image_embedding = model.encode(pil_image)
            new_embeddings.append(image_embedding.reshape(1, -1))

    conn.commit()
    conn.close()

    if new_embeddings:
        # Add new embeddings to the FAISS index
        embeddings_np = np.vstack(new_embeddings).astype('float32')
        index.add(embeddings_np)
        faiss.write_index(index, INDEX_FILE)
        print(f"Successfully ingested {file_name} and added {len(new_embeddings)} new chunks to the knowledge base.")
    else:
        print(f"No new content found to ingest in {file_name}.")
# database.py

import sqlite3
import os

DB_FILE = "auramind_local.db"
INDEX_FILE = "auramind_faiss.index"

def init_db():
    """
    Initializes a more robust database schema for multimodal data.
    - 'documents' table tracks the source files.
    - 'chunks' table stores the individual encrypted text/image chunks.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Table to track the source documents (e.g., 'healthy_maize.txt', 'user_guide.pdf')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    ''')

    # Table to store each chunk of content (text or image)
    # The faiss_id will correspond to the row number in the FAISS index
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            content_type TEXT NOT NULL, -- 'text' or 'image'
            encrypted_content BLOB NOT NULL,
            page_num INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    ''')
    conn.commit()
    conn.close()

def get_db_connection():
    """Establishes a connection to the database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def check_if_indexed():
    """Checks if the initial database and index file exist."""
    # A basic check. A more robust check might query the db for content.
    return os.path.exists(DB_FILE) and os.path.exists(INDEX_FILE)

def delete_database_and_index():
    """Deletes existing db and index files for a clean rebuild."""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed old database: {DB_FILE}")
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        print(f"Removed old index: {INDEX_FILE}")
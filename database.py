import sqlite3
import os

DB_FILE = "auramind_local.db"
INDEX_FILE = "auramind_faiss.index"

def init_db():
    """Initializes the database and table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Storing data as BLOB for encrypted bytes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encrypted_chunk BLOB NOT NULL,
            encrypted_embedding BLOB NOT NULL
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
    """Checks if the database and index file exist."""
    return os.path.exists(DB_FILE) and os.path.exists(INDEX_FILE)
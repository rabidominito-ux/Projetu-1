import sqlite3
import pandas as pd

DB_NAME = "cfp_database.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'admin123')")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extra_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_pessoal TEXT,
            id_sigap TEXT UNIQUE,
            id_grp TEXT,
            sexo TEXT,
            local_trabalho TEXT,
            cargo TEXT,
            Asiduidade REAL,
            Pontualidade REAL,
            Produtividade REAL,
            Kualidade_Servisu REAL,
            Kooperasaun REAL,
            Inisiativa REAL,
            Disiplina REAL,
            Responsabilidade REAL,
            Rezultadu_Avaliasaun TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

def save_extra_to_db(record):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO extra_records (
            nome_pessoal, id_sigap, id_grp, sexo, local_trabalho, cargo,
            Asiduidade, Pontualidade, Produtividade, Kualidade_Servisu,
            Kooperasaun, Inisiativa, Disiplina, Responsabilidade, Rezultadu_Avaliasaun
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("nome_pessoal"), record.get("id_sigap"), record.get("id_grp"),
        record.get("sexo"), record.get("local_trabalho"), record.get("cargo"),
        record.get("Asiduidade"), record.get("Pontualidade"), record.get("Produtividade"),
        record.get("Kualidade_Servisu"), record.get("Kooperasaun"), record.get("Inisiativa"),
        record.get("Disiplina"), record.get("Responsabilidade"), record.get("Rezultadu_Avaliasaun")
    ))
    conn.commit()
    conn.close()

def load_extra_from_db():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM extra_records", conn)
    conn.close()
    if df.empty:
        return []
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df.to_dict(orient="records")

def delete_extra_from_db_by_index(index):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id_sigap FROM extra_records")
    rows = cursor.fetchall()
    if index < len(rows):
        target_sigap = rows[index][0]
        cursor.execute("DELETE FROM extra_records WHERE id_sigap = ?", (target_sigap,))
        conn.commit()
    conn.close()

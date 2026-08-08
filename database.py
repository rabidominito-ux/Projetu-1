import sqlite3
import pandas as pd

DB_NAME = "cfp_database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extra_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_pessoal TEXT,
            id_sigap TEXT UNIQUE,
            Rezultadu_Avaliasaun TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_data(data_dict):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO extra_reports (nome_pessoal, id_sigap, Rezultadu_Avaliasaun) VALUES (?, ?, ?)",
                       (data_dict['nome'], data_dict['id_sigap'], data_dict['result']))
        conn.commit()
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        conn.close()

def get_all_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM extra_reports", conn)
    conn.close()
    return df

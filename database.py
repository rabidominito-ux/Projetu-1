import sqlite3
import pandas as pd
import bcrypt

def init_db(db_name="cfp_database.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 1. Tabela ba dadus extra reports
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extra_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            controlo_ativo_identificacao TEXT,
            nome_pessoal TEXT,
            id_sigap TEXT UNIQUE,
            sexo TEXT,
            instituicao TEXT,
            local_trabalho TEXT,
            data_de_nascimento TEXT,
            funcao TEXT,
            cargo TEXT,
            id_grp TEXT,
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
    ''')
    
    # 2. Tabela ba Users (Autentikasaun)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT
        )
    ''')
    
    # Kria Default Admin se seidauk iha (Username: admin, Password: admin123)
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_pw = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users VALUES (?, ?, ?)", ('admin', hashed_pw, 'admin'))

    conn.commit()
    conn.close()

def load_extra_from_db(db_name="cfp_database.db"):
    try:
        conn = sqlite3.connect(db_name)
        df = pd.read_sql("SELECT * FROM extra_reports", conn)
        conn.close()
        return df.to_dict(orient="records")
    except Exception:
        return []

def save_extra_to_db(data, db_name="cfp_database.db"):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO extra_reports (
                controlo_ativo_identificacao, nome_pessoal, id_sigap, sexo, instituicao,
                local_trabalho, data_de_nascimento, funcao, cargo, id_grp,
                Asiduidade, Pontualidade, Produtividade, Kualidade_Servisu,
                Kooperasaun, Inisiativa, Disiplina, Responsabilidade, Rezultadu_Avaliasaun
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get("controlo_ativo_identificacao"), data.get("nome_pessoal"), data.get("id_sigap"),
            data.get("sexo"), data.get("instituicao"), data.get("local_trabalho"),
            data.get("data_de_nascimento"), data.get("funcao"), data.get("cargo"), data.get("id_grp"),
            data.get("Asiduidade"), data.get("Pontualidade"), data.get("Produtividade"), data.get("Kualidade_Servisu"),
            data.get("Kooperasaun"), data.get("Inisiativa"), data.get("Disiplina"), data.get("Responsabilidade"),
            data.get("Rezultadu_Avaliasaun")
        ))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def update_extra_in_db_by_index(index, data, db_name="cfp_database.db"):
    records = load_extra_from_db(db_name)
    if 0 <= index < len(records):
        target_id_sigap = records[index].get("id_sigap")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE extra_reports SET
                controlo_ativo_identificacao=?, nome_pessoal=?, id_sigap=?, sexo=?, instituicao=?,
                local_trabalho=?, data_de_nascimento=?, funcao=?, cargo=?, id_grp=?,
                Asiduidade=?, Pontualidade=?, Produtividade=?, Kualidade_Servisu=?,
                Kooperasaun=?, Inisiativa=?, Disiplina=?, Responsabilidade=?, Rezultadu_Avaliasaun=?
            WHERE id_sigap=?
        ''', (
            data.get("controlo_ativo_identificacao"), data.get("nome_pessoal"), data.get("id_sigap"),
            data.get("sexo"), data.get("instituicao"), data.get("local_trabalho"),
            data.get("data_de_nascimento"), data.get("funcao"), data.get("cargo"), data.get("id_grp"),
            data.get("Asiduidade"), data.get("Pontualidade"), data.get("Produtividade"), data.get("Kualidade_Servisu"),
            data.get("Kooperasaun"), data.get("Inisiativa"), data.get("Disiplina"), data.get("Responsabilidade"),
            data.get("Rezultadu_Avaliasaun"), target_id_sigap
        ))
        conn.commit()
        conn.close()

def delete_extra_from_db_by_index(index, db_name="cfp_database.db"):
    records = load_extra_from_db(db_name)
    if 0 <= index < len(records):
        target_id_sigap = records[index].get("id_sigap")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM extra_reports WHERE id_sigap = ?", (target_id_sigap,))
        conn.commit()
        conn.close()
        return True
    return False

# ==========================================
# FUNSAUN AUTENTIKASAUN & JESTAUN USER
# ==========================================
def verify_user(username, password, db_name="cfp_database.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode('utf-8'), result[0])
    return False

def add_user(username, password, role='user', db_name="cfp_database.db"):
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, hashed_pw, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

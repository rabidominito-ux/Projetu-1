import sqlite3
import pandas as pd

DB_NAME = "cfp_database.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
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
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Erro iha inicializasaun database: {e}")

def load_extra_from_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        df_db = pd.read_sql_query("SELECT * FROM extra_reports", conn)
        conn.close()
        if "id" in df_db.columns:
            df_db = df_db.drop(columns=["id"])
        return df_db.to_dict("records")
    except Exception as e:
        return []

def save_extra_to_db(report_dict):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO extra_reports (
                controlo_ativo_identificacao, nome_pessoal, id_sigap, sexo, instituicao, local_trabalho, 
                data_de_nascimento, funcao, cargo, id_grp, Asiduidade, 
                Pontualidade, Produtividade, Kualidade_Servisu, Kooperasaun, 
                Inisiativa, Disiplina, Responsabilidade, Rezultadu_Avaliasaun
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_dict["controlo_ativo_identificacao"],
            report_dict["nome_pessoal"],
            report_dict["id_sigap"],
            report_dict["sexo"],
            report_dict["instituicao"],
            report_dict["local_trabalho"],
            report_dict["data_de_nascimento"],
            report_dict["funcao"],
            report_dict["cargo"],
            report_dict["id_grp"],
            report_dict["Asiduidade"],
            report_dict["Pontualidade"],
            report_dict["Produtividade"],
            report_dict["Kualidade_Servisu"],
            report_dict["Kooperasaun"],
            report_dict["Inisiativa"],
            report_dict["Disiplina"],
            report_dict["Responsabilidade"],
            report_dict["Rezultadu_Avaliasaun"],
        ))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        return False

def update_extra_in_db_by_index(index_val, report_dict):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM extra_reports")
        ids = [row[0] for row in cursor.fetchall()]
        if index_val < len(ids):
            row_id = ids[index_val]
            cursor.execute("""
                UPDATE extra_reports SET 
                    controlo_ativo_identificacao=?, nome_pessoal=?, id_sigap=?, sexo=?, instituicao=?, local_trabalho=?, 
                    data_de_nascimento=?, funcao=?, cargo=?, id_grp=?, Asiduidade=?, 
                    Pontualidade=?, Produtividade=?, Kualidade_Servisu=?, Kooperasaun=?, 
                    Inisiativa=?, Disiplina=?, Responsabilidade=?, Rezultadu_Avaliasaun=?
                WHERE id=?
            """, (
                report_dict["controlo_ativo_identificacao"],
                report_dict["nome_pessoal"],
                report_dict["id_sigap"],
                report_dict["sexo"],
                report_dict["instituicao"],
                report_dict["local_trabalho"],
                report_dict["data_de_nascimento"],
                report_dict["funcao"],
                report_dict["cargo"],
                report_dict["id_grp"],
                report_dict["Asiduidade"],
                report_dict["Pontualidade"],
                report_dict["Produtividade"],
                report_dict["Kualidade_Servisu"],
                report_dict["Kooperasaun"],
                report_dict["Inisiativa"],
                report_dict["Disiplina"],
                report_dict["Responsabilidade"],
                report_dict["Rezultadu_Avaliasaun"],
                row_id,
            ))
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def delete_extra_from_db(index_val):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM extra_reports")
        ids = [row[0] for row in cursor.fetchall()]
        if index_val < len(ids):
            row_id = ids[index_val]
            cursor.execute("DELETE FROM extra_reports WHERE id=?", (row_id,))
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

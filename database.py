from datetime import datetime
import sqlite3
import pandas as pd

DB_NAME = "cfp_database.db"


def init_db():
  """Kria tabela database se seidauk eziste, inklui koluna Audit Trail."""
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
            Rezultadu_Avaliasaun TEXT,
            created_at TEXT,
            updated_at TEXT,
            modified_by TEXT
        )
    """)
  conn.commit()
  conn.close()


def save_extra_to_db(record, username="System"):
  """Rai dadus foun ba database ho Audit Trail."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  try:
    cursor.execute(
        """
            INSERT INTO extra_reports (
                controlo_ativo_identificacao, nome_pessoal, id_sigap, sexo, instituicao,
                local_trabalho, data_de_nascimento, funcao, cargo, id_grp,
                Asiduidade, Pontualidade, Produtividade, Kualidade_Servisu,
                Kooperasaun, Inisiativa, Disiplina, Responsabilidade,
                Rezultadu_Avaliasaun, created_at, updated_at, modified_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.get("controlo_ativo_identificacao"),
            record.get("nome_pessoal"),
            record.get("id_sigap"),
            record.get("sexo"),
            record.get("instituicao"),
            record.get("local_trabalho"),
            record.get("data_de_nascimento"),
            record.get("funcao"),
            record.get("cargo"),
            record.get("id_grp"),
            record.get("Asiduidade"),
            record.get("Pontualidade"),
            record.get("Produtividade"),
            record.get("Kualidade_Servisu"),
            record.get("Kooperasaun"),
            record.get("Inisiativa"),
            record.get("Disiplina"),
            record.get("Responsabilidade"),
            record.get("Rezultadu_Avaliasaun"),
            now,
            now,
            username,
        ),
    )
    conn.commit()
    return True
  except sqlite3.IntegrityError:
    return False
  finally:
    conn.close()


def load_extra_from_db():
  """Foti dadus hotu husi database."""
  conn = sqlite3.connect(DB_NAME)
  df = pd.read_sql_query("SELECT * FROM extra_reports", conn)
  conn.close()
  return df.to_dict(orient="records")


def update_extra_in_db_by_index(index_or_id, record, username="System"):
  """Atualiza dadus tuir index iha database ho Audit Trail."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  try:
    records = load_extra_from_db()
    if index_or_id < len(records):
      db_id = records[index_or_id]["id"]
      cursor.execute(
          """
                UPDATE extra_reports SET
                    controlo_ativo_identificacao = ?, nome_pessoal = ?, id_sigap = ?,
                    sexo = ?, instituicao = ?, local_trabalho = ?, data_de_nascimento = ?,
                    funcao = ?, cargo = ?, id_grp = ?, Asiduidade = ?, Pontualidade = ?,
                    Produtividade = ?, Kualidade_Servisu = ?, Kooperasaun = ?,
                    Inisiativa = ?, Disiplina = ?, Responsabilidade = ?,
                    Rezultadu_Avaliasaun = ?, updated_at = ?, modified_by = ?
                WHERE id = ?
            """,
          (
              record.get("controlo_ativo_identificacao"),
              record.get("nome_pessoal"),
              record.get("id_sigap"),
              record.get("sexo"),
              record.get("instituicao"),
              record.get("local_trabalho"),
              record.get("data_de_nascimento"),
              record.get("funcao"),
              record.get("cargo"),
              record.get("id_grp"),
              record.get("Asiduidade"),
              record.get("Pontualidade"),
              record.get("Produtividade"),
              record.get("Kualidade_Servisu"),
              record.get("Kooperasaun"),
              record.get("Inisiativa"),
              record.get("Disiplina"),
              record.get("Responsabilidade"),
              record.get("Rezultadu_Avaliasaun"),
              now,
              username,
              db_id,
          ),
      )
      conn.commit()
      return True
    return False
  except Exception as e:
    print(f"Erro update: {e}")
    return False
  finally:
    conn.close()


def delete_extra_from_db_by_index(index_or_id):
  """Hamos dadus husi database."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  try:
    records = load_extra_from_db()
    if index_or_id < len(records):
      db_id = records[index_or_id]["id"]
      cursor.execute("DELETE FROM extra_reports WHERE id = ?", (db_id,))
      conn.commit()
      return True
    return False
  except Exception as e:
    print(f"Erro delete: {e}")
    return False
  finally:
    conn.close()

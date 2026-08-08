import io
import re
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st

# ReportLab imports ba PDF Generation
try:
  from reportlab.lib import colors
  from reportlab.lib.pagesizes import letter
  from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
  from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

  REPORTLAB_AVAILABLE = True
except ImportError:
  REPORTLAB_AVAILABLE = False

# 1. Konfigurasaun Pajina Streamlit (Layout Wide)
st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - Decision Tree",
    page_icon="📊",
    layout="wide",
)

# 2. Custom CSS ba UI ne'ebé modernu no profesional
st.markdown(
    """
    <style>
    .main { background-color: #F8FAFC; }
    .main-title { font-size: 2.4rem; color: #0F172A; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0px; }
    .sub-title { color: #475569; font-size: 1.1rem; margin-bottom: 25px; }
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white; border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: 600; border: none;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2); transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #F1F5F9; border-radius: 8px 8px 0px 0px; padding: 10px 20px; font-weight: 600; color: #334155; }
    .stTabs [aria-selected="true"] { background-color: #2563EB !important; color: white !important; }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Principal
st.markdown(
    '<p class="main-title">📊 Sistema Klasifikasaun Dezempenu Funsionáriu'
    ' CFP</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Aplikasaun Inteligénsia Artifisiál uza algoritmu'
    " Decision Tree ho Autentikasaun, Radar Chart no Gera PDF Ofisiál.</p>",
    unsafe_allow_html=True,
)

# 3. Konfigurasaun Database SQLite Local
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
    st.error(f"⚠️ Erro iha inicializasaun database: {e}")


init_db()

# Session State ba Login & Edit
if "logged_in" not in st.session_state:
  st.session_state["logged_in"] = False
if "username" not in st.session_state:
  st.session_state["username"] = ""
if "role" not in st.session_state:
  st.session_state["role"] = "User"
if "edit_index" not in st.session_state:
  st.session_state["edit_index"] = None
if "selected_category" not in st.session_state:
  st.session_state["selected_category"] = None

# --- TELA LOGIN (SE SEIDAAK LOGGED IN, APLIKASAUN TAU PAUSA TO'O LOGIN SUKSESU) ---
if not st.session_state["logged_in"]:
  st.warning("🔒 **Seguransa Sistema:** Favor hatama Ita-nia Username no Password molok asesu ba sistema.")
  col1, col2, col3 = st.columns([1, 1.5, 1])
  with col2:
    with st.form("main_login_form"):
      st.markdown("### 🔐 Autentikasaun Usuáriu")
      u_input = st.text_input("Username")
      p_input = st.text_input("Password", type="password")
      login_btn = st.form_submit_button("🔑 Tama ba Sistema (Login)")
      if login_btn:
        if u_input == "admin" and p_input == "admin123":
          st.session_state["logged_in"] = True
          st.session_state["username"] = "Administrador"
          st.session_state["role"] = "Admin"
          st.success("Login nu'udar Admin susesu!")
          st.rerun()
        elif u_input == "user" and p_input == "user123":
          st.session_state["logged_in"] = True
          st.session_state["username"] = "Funsionáriu"
          st.session_state["role"] = "User"
          st.success("Login nu'udar User susesu!")
          st.rerun()
        else:
          st.error("⚠️ Username ka Password sala! Uza admin/admin123 ka user/user123")
    st.info("💡 **Nota Default:** Admin (`admin`/`admin123`) ka User (`user`/`user123`)")
  st.stop()  # Hapoi prosesu seluk tomak to'o de'it kuandu login ona

# --- SE LOGGED IN ONA, HATUDU KONTROLUS IHA SIDEBAR NO DADOS APLIKASAUN ---
st.sidebar.success(
    f"👤 Logged in as: **{st.session_state['username']}**"
    f" ({st.session_state['role']})"
)
if st.sidebar.button("🚪 Logout"):
  st.session_state["logged_in"] = False
  st.session_state["username"] = ""
  st.session_state["role"] = "User"
  st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Gestaun Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload ficheiru Excel (.xlsx)", type=["xlsx"]
)


@st.cache_data
def load_data(file):
  return pd.read_excel(file, sheet_name="Sheet1", header=0)


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
    cursor.execute(
        """
            INSERT INTO extra_reports (
                controlo_ativo_identificacao, nome_pessoal, id_sigap, sexo, instituicao, local_trabalho, 
                data_de_nascimento, funcao, cargo, id_grp, Asiduidade, 
                Pontualidade, Produtividade, Kualidade_Servisu, Kooperasaun, 
                Inisiativa, Disiplina, Responsabilidade, Rezultadu_Avaliasaun
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
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
        ),
    )
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
      cursor.execute(
          """
                UPDATE extra_reports SET 
                    controlo_ativo_identificacao=?, nome_pessoal=?, id_sigap=?, sexo=?, instituicao=?, local_trabalho=?, 
                    data_de_nascimento=?, funcao=?, cargo=?, id_grp=?, Asiduidade=?, 
                    Pontualidade=?, Produtividade=?, Kualidade_Servisu=?, Kooperasaun=?, 
                    Inisiativa=?, Disiplina=?, Responsabilidade=?, Rezultadu_Avaliasaun=?
                WHERE id=?
            """,
          (
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
          ),
      )
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


# Função ba Gera PDF Ofisiál uza ReportLab
def generate_pdf_report(rep):
  buffer = io.BytesIO()
  doc = SimpleDocTemplate(buffer, pagesize=letter)
  story = []
  styles = getSampleStyleSheet()

  title_style = ParagraphStyle(
      'TitleStyle',
      parent=styles['Heading1'],
      fontSize=18,
      textColor=colors.HexColor('#1E3A8A'),
      alignment=1,
  )

  story.append(Paragraph("REPUBLICA DEMOCRATICA DE TIMOR-LESTE", title_style))
  story.append(Paragraph("COMISAUN FUNÇÃO PÚBLICA (CFP)", title_style))
  story.append(
      Paragraph("<b>RELATÓRIU AVALIAZAUN DEZEMPENU FUNSIÓRIU</b>", title_style)
  )
  story.append(Spacer(1, 20))

  data = [
      ["Naran Pessoal:", rep["nome_pessoal"]],
      ["ID SIGAP:", rep["id_sigap"]],
      ["Sexo:", rep["sexo"]],
      ["Local Trabalhu:", rep["local_trabalho"]],
      ["Cargo / Funsaun:", f"{rep['cargo']} - {rep['funcao']}"],
      ["Asiduidade:", str(rep["Asiduidade"])],
      ["Pontualidade:", str(rep["Pontualidade"])],
      ["Produtividade:", str(rep["Produtividade"])],
      ["Kualidade Servisu:", str(rep["Kualidade_Servisu"])],
      ["Kooperasaun:", str(rep["Kooperasaun"])],
      ["Inisiativa:", str(rep["Inisiativa"])],
      ["Disiplina:", str(rep["Disiplina"])],
      ["Responsabilidade:", str(rep["Responsabilidade"])],
      ["REZULTADU AVALIAZAUN:", str(rep["Rezultadu_Avaliasaun"])],
  ]

  t = Table(data, colWidths=[150, 300])
  t.setStyle(
      TableStyle([
          ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
          ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
          ("ALIGN", (0, 0), (-1, -1), "LEFT"),
          ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
          ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
      ])
  )
  story.append(t)
  doc.build(story)
  buffer.seek(0)
  return buffer


if "extra_reports" not in st.session_state:
  st.session_state["extra_reports"] = load_extra_from_db()

if uploaded_file is not None:
  try:
    df_raw = load_data(uploaded_file)
    rename_map = {
        "Column1": "controlo_ativo_identificacao",
        "Column2": "nome_pessoal",
        "Column3": "id_sigap",
        "Column4": "id_grp",
        "Column5": "sexo",
        "Column6": "data_de_nascimento",
        "Column7": "instituicao",
        "Column8": "local_trabalho",
        "Column9": "funcao",
        "Column10": "cargo",
        "Column11": "data_fim_nao_exercicio",
        "Column12": "temp1",
        "Column13": "Asiduidade",
        "Column14": "Pontualidade",
        "Column15": "Produtividade",
        "Column16": "Kualidade_Servisu",
        "Column17": "Kooperasaun",
        "Column18": "Inisiativa",
        "Column19": "Disiplina",
        "Column20": "Responsabilidade",
        "Column21": "Media",
        "Column22": "Rezultadu_Avaliasaun",
        "Column23": "temp2",
    }
    df_raw.rename(
        columns={k: v for k, v in rename_map.items() if k in df_raw.columns},
        inplace=True,
    )

    nota_cols = [
        "Asiduidade",
        "Pontualidade",
        "Produtividade",
        "Kualidade_Servisu",
        "Kooperasaun",
        "Inisiativa",
        "Disiplina",
        "Responsabilidade",
    ]
    target_col = "Rezultadu_Avaliasaun"

    missing_cols = [
        col for col in nota_cols + [target_col] if col not in df_raw.columns
    ]

    if len(missing_cols) > 0:
      st.sidebar.error(f"⚠️ Falta koluna: {', '.join(missing_cols)}")
    else:
      df_base = df_raw.dropna(subset=nota_cols + [target_col]).copy()
      for col in nota_cols:
        df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

      st.session_state["extra_reports"] = load_extra_from_db()
      if len(st.session_state["extra_reports"]) > 0:
        df_extra = pd.DataFrame(st.session_state["extra_reports"])
        df = pd.concat([df_base, df_extra], ignore_index=True)
        df = df.drop_duplicates(subset=["id_sigap"], keep="last")
      else:
        df = df_base

      st.sidebar.markdown("---")
      csv_full = df.to_csv(index=False).encode("utf-8")
      st.sidebar.download_button(
          label="⬇️ Download Backup (CSV)",
          data=csv_full,
          file_name="dataset_cfp_kompletu.csv",
          mime="text/csv",
      )

      le = LabelEncoder()
      df["target_encoded"] = le.fit_transform(df[target_col].astype(str))
      y = df["target_encoded"]
      X = df[nota_cols].copy()

      try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
      except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

      model = DecisionTreeClassifier(
          criterion="entropy", max_depth=5, random_state=42
      )
      model.fit(X_train, y_train)

      df["Prediksaun"] = le.inverse_transform(model.predict(X))
      acc = accuracy_score(y_test, model.predict(X_test))

      # Tabs Navigasaun
      tab1, tab2, tab3 = st.tabs([
          "📊 Dashboard Analítiku",
          "⚙️ Modelu & Performance",
          "🔮 Prediksaun & Jere Dadus",
      ])

      with tab1:
        st.markdown("### 📈 Sumáriu Dezempenu Funsionáriu")
        total_funs = len(df)
        counts_real = df[target_col].value_counts()

        mb_pct = (
            (counts_real.get("Muito Bom", 0) / total_funs) * 100
            if total_funs > 0
            else 0
        )
        b_pct = (
            (counts_real.get("Bom", 0) / total_funs) * 100
            if total_funs > 0
            else 0
        )
        s_pct = (
            (counts_real.get("Suficiente", 0) / total_funs) * 100
            if total_funs > 0
            else 0
        )
        i_pct = (
            (counts_real.get("Insuficiente", 0) / total_funs) * 100
            if total_funs > 0
            else 0
        )

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
          if st.button(f"📊 Total Funsionáriu\n\n{total_funs}"):
            st.session_state["selected_category"] = (
                "Tomak"
                if st.session_state["selected_category"] != "Tomak"
                else None
            )
        with col_m2:
          if st.button(
              f"⭐ Muito Bom\n\n{counts_real.get('Muito Bom', 0)}\n({mb_pct:.1f}%)"
          ):
            st.session_state["selected_category"] = (
                "Muito Bom"
                if st.session_state["selected_category"] != "Muito Bom"
                else None
            )
        with col_m3:
          if st.button(
              f"✨ Bom\n\n{counts_real.get('Bom', 0)}\n({b_pct:.1f}%)"
          ):
            st.session_state["selected_category"] = (
                "Bom"
                if st.session_state["selected_category"] != "Bom"
                else None
            )
        with col_m4:
          if st.button(
              f"📌 Suficiente\n\n{counts_real.get('Suficiente', 0)}\n({s_pct:.1f}%)"
          ):
            st.session_state["selected_category"] = (
                "Suficiente"
                if st.session_state["selected_category"] != "Suficiente"
                else None
            )
        with col_m5:
          if st.button(
              f"⚠️ Insuficiente\n\n{counts_real.get('Insuficiente', 0)}\n({i_pct:.1f}%)"
          ):
            st.session_state["selected_category"] = (
                "Insuficiente"
                if st.session_state["selected_category"] != "Insuficiente"
                else None
            )

        selected_cat = st.session_state["selected_category"]
        if selected_cat is not None:
          st.markdown("---")
          col_f1, col_f2 = st.columns(2)
          with col_f1:
            if selected_cat == "Tomak":
              df_filtered = df
              st.markdown(
                  f"### 📋 Lista Funsionáriu Tomak ({len(df_filtered)})"
              )
            else:
              df_filtered = df[df[target_col] == selected_cat]
              st.markdown(
                  f"### 📋 Lista Funsionáriu ba Kategoria: `{selected_cat}`"
                  f" ({len(df_filtered)})"
              )
          with col_f2:
            mun_list = ["Tomak (Hotu-hotu)"] + list(
                df["local_trabalho"].dropna().unique()
            )
            selected_mun = st.selectbox("Filtru tuir Munisípiu:", mun_list)
            if selected_mun != "Tomak (Hotu-hotu)":
              df_filtered = df_filtered[
                  df_filtered["local_trabalho"] == selected_mun
              ]

          st.dataframe(
              df_filtered[
                  [
                      "controlo_ativo_identificacao",
                      "nome_pessoal",
                      "id_sigap",
                      "id_grp",
                      "sexo",
                      "local_trabalho",
                      "cargo",
                      target_col,
                  ]
              ],
              use_container_width=True,
          )
          csv_filtered = df_filtered.to_csv(index=False).encode("utf-8")
          st.download_button(
              label="📥 Download Dadus Filtra Ne'e (CSV)",
              data=csv_filtered,
              file_name=f"relatorio_cfp_{selected_cat}.csv",
              mime="text/csv",
          )
          if st.button("❌ Subar Tabela"):
            st.session_state["selected_category"] = None
            st.rerun()

        st.markdown("---")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
          st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")
          fig, ax = plt.subplots(figsize=(6, 4))
          categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
          real_counts = [counts_real.get(cat, 0) for cat in categories]
          pred_counts = [
              df["Prediksaun"].value_counts().get(cat, 0) for cat in categories
          ]
          x = np.arange(len(categories))
          width = 0.35
          ax.bar(
              x - width / 2,
              real_counts,
              width,
              label="Dadus Reál",
              color="#3B82F6",
              alpha=0.9,
          )
          ax.bar(
              x + width / 2,
              pred_counts,
              width,
              label="Prediksaun Tree",
              color="#10B981",
              alpha=0.9,
          )
          ax.set_ylabel("Total Funsionáriu")
          ax.set_xticks(x)
          ax.set_xticklabels(categories)
          ax.legend()
          sns.despine()
          st.pyplot(fig)

        with col_g2:
          st.markdown("##### 🍩 Proporsaun Kategoria Dezempenu")
          fig2, ax2 = plt.subplots(figsize=(6, 4))
          sizes = [counts_real.get(cat, 0) for cat in categories]
          colors_list = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
          ax2.pie(
              sizes,
              labels=categories,
              autopct="%1.1f%%",
              startangle=90,
              colors=colors_list,
              wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
          )
          st.pyplot(fig2)

      with tab2:
        st.subheader("📋 Amostra Dadus (Preview)")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("---")
        st.subheader("🚀 Performance Modelu Decision Tree")
        st.success(f"✅ Akurasi Modelu (Accuracy): **{acc * 100:.2f}%**")

        col_eval1, col_eval2 = st.columns(2)
        y_pred_test = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)

        with col_eval1:
          st.markdown("##### 📉 Confusion Matrix")
          fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
          sns.heatmap(
              cm,
              annot=True,
              fmt="d",
              cmap="Blues",
              cbar=False,
              xticklabels=le.classes_,
              yticklabels=le.classes_,
              ax=ax_cm,
          )
          st.pyplot(fig_cm)

        with col_eval2:
          st.markdown("##### 📑 Classification Report")
          unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
          present_class_names = [le.classes_[i] for i in unique_labels]
          report_dict = classification_report(
              y_test,
              y_pred_test,
              labels=unique_labels,
              target_names=present_class_names,
              output_dict=True,
              zero_division=0,
          )
          df_report = pd.DataFrame(report_dict).transpose()
          st.dataframe(
              df_report.style.format(
                  subset=["precision", "recall", "f1-score", "support"],
                  formatter="{:.2f}",
              ),
              use_container_width=True,
          )

        st.markdown("---")
        st.subheader("🌳 Vizualizasaun Árbore Desizaun")
        max_depth_vis = st.slider("Hili Profundidade Árbore (Max Depth)", 1, 5, 3)
        vis_model = DecisionTreeClassifier(
            criterion="entropy", max_depth=max_depth_vis, random_state=42
        )
        vis_model.fit(X_train, y_train)
        fig_tree, ax_tree = plt.subplots(figsize=(16, 9), dpi=100)
        plot_tree(
            vis_model,
            feature_names=nota_cols,
            class_names=le.classes_,
            filled=True,
            rounded=True,
            ax=ax_tree,
            fontsize=9,
        )
        st.pyplot(fig_tree)

      with tab3:
        st.subheader("🔍 Prediksaun Funsionáriu Foun & Jere Dadus")

        idx_edit = st.session_state["edit_index"]
        def_val = {}
        if (
            idx_edit is not None
            and idx_edit < len(st.session_state["extra_reports"])
        ):
          def_val = st.session_state["extra_reports"][idx_edit]

        with st.form("funsionariu_form"):
          st.markdown("##### 📝 1. Informasaun Identidade Funsionáriu")
          municipios = [
              "Aileu",
              "Ainaro",
              "Baucau",
              "Bobonaro",
              "Covalima",
              "Díli",
              "Ermera",
              "Lautém",
              "Liquiçá",
              "Manatuto",
              "Manufahi",
              "Oe-Cusse Ambeno",
              "Viqueque",
          ]
          funcoes = [
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 10,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 1,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 4,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau C, 1,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau A, 4,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau C, 5,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 3,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 6,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 5,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 1,"
                  " NOMEAÇÃO PROBATÓRIA"
              ),
              (
                  "Regime Geral das Carreiras, Assistente Grau F, 5,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Assistente Grau F, 3,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 1,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 1,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau A, 3,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 2,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 1,"
                  " NOMEAÇÃO PROBATÓRIA"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau C, 1,"
                  " NOMEAÇÃO PROBATÓRIA"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 2,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau A, 8,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 7,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau A, 2,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau A, 1,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Administrativo Grau E, 2,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 3,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau C, 3,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 4,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau C, 4,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau C, 2,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 4,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Assistente Grau F, 2,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Profissional Grau D, 5,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 5,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 7,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 3,"
                  " PERMANENTE"
              ),
              (
                  "Regime Geral das Carreiras, Técnico Superior Grau B, 9,"
                  " PERMANENTE"
              ),
          ]
          cargos = [
              "Técnico Superior",
              "Técnico Profissional",
              "Assistente Administrativo",
              "Oficial Administrativo",
              "Assistente Técnico",
              "Técnico Informática",
              "Analista de Dados",
              "Chefe de Unidade",
              "Chefe de Departamento",
              "Diretor Nacional",
          ]

          col_i1, col_i2, col_i3 = st.columns(3)
          with col_i1:
            ativo_opts = ["Ativo", "La Ativo"]
            cur_ativo = def_val.get("controlo_ativo_identificacao", "Ativo")
            idx_ativo = (
                ativo_opts.index(cur_ativo)
                if cur_ativo in ativo_opts
                else 0
            )
            txt_ativo = st.selectbox(
                "Controlo Ativo Identifikasaun", ativo_opts, index=idx_ativo
            )
            txt_nome = st.text_input(
                "Naran Pessoal*", def_val.get("nome_pessoal", "")
            )
            txt_sigap = st.text_input(
                "ID SIGAP (Numeriku/Símbolu)*", def_val.get("id_sigap", "")
            )
            txt_sexo = st.selectbox(
                "Sexo",
                ["M", "F"],
                index=0 if def_val.get("sexo", "M") == "M" else 1,
            )
          with col_i2:
            txt_inst = st.text_input(
                "Instituisaun", def_val.get("instituicao", "CFP")
            )
            cur_local = def_val.get("local_trabalho", "Díli")
            idx_local = (
                municipios.index(cur_local) if cur_local in municipios else 5
            )
            txt_local = st.selectbox(
                "Local Trabalhu", municipios, index=idx_local
            )
            try:
              default_date = pd.to_datetime(
                  def_val.get("data_de_nascimento", "1995-01-01")
              ).date()
            except:
              default_date = pd.to_datetime("1995-01-01").date()
            txt_nascimento = st.date_input(
                "Data de Nascimento", value=default_date
            )
          with col_i3:
            cur_func = def_val.get("funcao", funcoes[0])
            idx_func = funcoes.index(cur_func) if cur_func in funcoes else 0
            txt_funcao = st.selectbox("Funsaun", funcoes, index=idx_func)
            cur_cargo = def_val.get("cargo", cargos[0])
            idx_cargo = cargos.index(cur_cargo) if cur_cargo in cargos else 0
            txt_cargo = st.selectbox("Kargo", cargos, index=idx_cargo)
            txt_grp = st.text_input(
                "ID GRP (Numeriku/Símbolu)", def_val.get("id_grp", "")
            )

          st.markdown(
              "##### 📊 2. Indikadór Avaliasaun Funsionáriu (Skala 1 - 5, Step"
              " 0.5)"
          )
          col_a, col_b, col_c, col_d = st.columns(4)
          with col_a:
            p_asid = st.slider(
                "Asiduidade",
                1.0,
                5.0,
                float(def_val.get("Asiduidade", 4.0)),
                0.5,
            )
            p_pont = st.slider(
                "Pontualidade",
                1.0,
                5.0,
                float(def_val.get("Pontualidade", 4.0)),
                0.5,
            )
          with col_b:
            p_prod = st.slider(
                "Produtividade",
                1.0,
                5.0,
                float(def_val.get("Produtividade", 4.0)),
                0.5,
            )
            p_kual = st.slider(
                "Kualidade Servisu",
                1.0,
                5.0,
                float(def_val.get("Kualidade_Servisu", 4.0)),
                0.5,
            )
          with col_c:
            p_koop = st.slider(
                "Kooperasaun",
                1.0,
                5.0,
                float(def_val.get("Kooperasaun", 4.0)),
                0.5,
            )
            p_inis = st.slider(
                "Inisiativa",
                1.0,
                5.0,
                float(def_val.get("Inisiativa", 4.0)),
                0.5,
            )
          with col_d:
            p_disp = st.slider(
                "Disiplina",
                1.0,
                5.0,
                float(def_val.get("Disiplina", 4.0)),
                0.5,
            )
            p_resp = st.slider(
                "Responsabilidade",
                1.0,
                5.0,
                float(def_val.get("Responsabilidade", 4.0)),
                0.5,
            )

          btn_label = (
              "💾 Update Relatóriu"
              if idx_edit is not None
              else "🔮 Predict & Hatama Relatóriu"
          )
          submit_pred = st.form_submit_button(btn_label)

          if submit_pred:
            existing_sigaps = df["id_sigap"].astype(str).str.strip().tolist()
            in_sigap = txt_sigap.strip()
            in_grp = txt_grp.strip()
            has_letters = lambda x: bool(re.search(r"[A-Za-z]", x))

            if not txt_nome.strip() or not txt_sigap.strip():
              st.warning("⚠️ Favor prennde Naran Pessoal no ID SIGAP!")
            elif has_letters(in_sigap):
              st.warning("⚠️ ID SIGAP labele uza letra alfabetu (A-Z)!")
            elif in_grp and has_letters(in_grp):
              st.warning("⚠️ ID GRP labele uza letra alfabetu (A-Z)!")
            elif in_sigap in existing_sigaps and idx_edit is None:
              st.error(
                  f"❌ ATENSAUN ERRO: ID SIGAP **'{in_sigap}'** eziste ona!"
              )
            else:
              input_data = np.array([[
                  p_asid,
                  p_pont,
                  p_prod,
                  p_kual,
                  p_koop,
                  p_inis,
                  p_disp,
                  p_resp,
              ]])
              pred_encoded = model.predict(input_data)
              pred_label = le.inverse_transform(pred_encoded)[0]

              new_report = {
                  "controlo_ativo_identificacao": txt_ativo,
                  "nome_pessoal": txt_nome,
                  "id_sigap": txt_sigap,
                  "sexo": txt_sexo,
                  "instituicao": txt_inst,
                  "local_trabalho": txt_local,
                  "data_de_nascimento": str(txt_nascimento),
                  "funcao": txt_funcao,
                  "cargo": txt_cargo,
                  "id_grp": txt_grp,
                  "Asiduidade": p_asid,
                  "Pontualidade": p_pont,
                  "Produtividade": p_prod,
                  "Kualidade_Servisu": p_kual,
                  "Kooperasaun": p_koop,
                  "Inisiativa": p_inis,
                  "Disiplina": p_disp,
                  "Responsabilidade": p_resp,
                  "Rezultadu_Avaliasaun": pred_label,
              }

              if idx_edit is not None:
                if update_extra_in_db_by_index(idx_edit, new_report):
                  st.session_state["edit_index"] = None
                  st.success("✅ Relatóriu atualiza ho suksesu!")
                  st.rerun()
              else:
                success_db = save_extra_to_db(new_report)
                if success_db:
                  st.session_state["edit_index"] = None
                  st.success("✅ Relatóriu foun rai ho suksesu!")
                  st.rerun()
                else:
                  st.error("⚠️ Falha: ID SIGAP duplicadu iha database.")

        if idx_edit is not None:
          if st.button("❌ Kansela Edit"):
            st.session_state["edit_index"] = None
            st.rerun()

        st.session_state["extra_reports"] = load_extra_from_db()
        extra_data_list = st.session_state["extra_reports"]

        if len(extra_data_list) > 0:
          st.markdown("---")
          st.subheader("📋 Lista Relatóriu Funsionáriu Foun")
          search_query = st.text_input("🔍 Buka Naran ka ID SIGAP:", "")

          filtered_reports = [
              (i, r)
              for i, r in enumerate(extra_data_list)
              if search_query.lower() in r["nome_pessoal"].lower()
              or search_query.lower() in r["id_sigap"].lower()
          ]

          for idx, rep in filtered_reports:
            with st.expander(
                f"👤 {rep['nome_pessoal']} (SIGAP: {rep['id_sigap']}) -"
                f" Rezultadu: {rep['Rezultadu_Avaliasaun']}"
            ):
              col_d1, col_d2 = st.columns(2)
              with col_d1:
                st.markdown(
                    f"**Ativu:**"
                    f" {rep.get('controlo_ativo_identificacao', 'Ativo')} |"
                    f" **Sexo:** {rep['sexo']}"
                )
                st.markdown(
                    f"**Fatin:** {rep['local_trabalho']} | **Funsaun:**"
                    f" {rep['funcao']}"
                )
              with col_d2:
                st.markdown(
                    f"✨ **Klasifikasaun:** `{rep['Rezultadu_Avaliasaun']}`"
                )

              # Radar Chart ba individual funsionáriu
              st.markdown("##### 🕷️ Perfíl Kompeténsia (Radar Chart)")
              categories_radar = [
                  "Asiduidade",
                  "Pontualidade",
                  "Produtividade",
                  "Kualidade",
                  "Kooperasaun",
                  "Inisiativa",
                  "Disiplina",
                  "Responsabilidade",
              ]
              values_radar = [
                  rep["Asiduidade"],
                  rep["Pontualidade"],
                  rep["Produtividade"],
                  rep["Kualidade_Servisu"],
                  rep["Kooperasaun"],
                  rep["Inisiativa"],
                  rep["Disiplina"],
                  rep["Responsabilidade"],
              ]
              values_radar += values_radar[:1]

              angles = [
                  n / float(len(categories_radar)) * 2 * np.pi
                  for n in range(len(categories_radar))
              ]
              angles += angles[:1]

              fig_radar, ax_radar = plt.subplots(
                  figsize=(4, 4), subplot_kw=dict(polar=True)
              )
              ax_radar.plot(
                  angles, values_radar, linewidth=2, linestyle="solid", color="#2563EB"
              )
              ax_radar.fill(angles, values_radar, color="#3B82F6", alpha=0.3)
              ax_radar.set_xticks(angles[:-1])
              ax_radar.set_xticklabels(categories_radar, size=8)
              ax_radar.set_yticks([1, 2, 3, 4, 5])
              ax_radar.set_ylim(0, 5)
              st.pyplot(fig_radar)

              c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
              with c1:
                if st.session_state["role"] == "Admin":
                  if st.button("✏️ Edit", key=f"edit_{idx}"):
                    st.session_state["edit_index"] = idx
                    st.rerun()
                else:
                  st.caption("🔒 Admin de'it")
              with c2:
                if st.session_state["role"] == "Admin":
                  if st.button("🗑️ Hamos", key=f"del_{idx}"):
                    if delete_extra_from_db(idx):
                      if st.session_state["edit_index"] == idx:
                        st.session_state["edit_index"] = None
                      st.success("Dadus hamos ona!")
                      st.rerun()
                else:
                  st.caption("🔒 Admin de'it")
              with c3:
                if REPORTLAB_AVAILABLE:
                  pdf_buffer = generate_pdf_report(rep)
                  st.download_button(
                      label="📥 PDF",
                      data=pdf_buffer,
                      file_name=f"relatorio_{rep['id_sigap']}.pdf",
                      mime="application/pdf",
                      key=f"pdf_{idx}",
                  )
  except Exception as e:
    st.error(f"⚠️ Erro iha prosesamentu fail: `{str(e)}`")
else:
  st.info(
      "👈 Favor upload uluk ficheiru Excel (`.xlsx`) iha sidebar sorin karuk hodi"
      " hahú sistema."
  )

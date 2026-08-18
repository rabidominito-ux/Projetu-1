# ============================================================
# APP.PY - SISTEMA KLASIFIKASAUN DESEMPENHU FUNSIONÁRIU CFP
# ============================================================

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

from database import (
    delete_extra_from_db_by_index,
    init_db,
    load_extra_from_db,
    save_extra_to_db,
    update_extra_in_db_by_index,
)
from models import treinar_modelo


# ============================================================
# CONFIGURASAUN PAGE
# ============================================================

st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - RDTL",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = []

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None


# ============================================================
# FUNSAUN LEE DATA
# ============================================================

def load_data(file):
    return pd.read_excel(file)


# ============================================================
# FUNSAUN PDF
# ============================================================

def generate_pdf_report(df_data, title_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=11,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=2,
        alignment=1,
        fontName="Helvetica-Bold",
    )

    subtitle_style = ParagraphStyle(
        "SubTitleStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=15,
        alignment=1,
        fontName="Helvetica",
    )

    elements.append(Paragraph("REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE", title_style))
    elements.append(Paragraph("COMISSÃO DA FUNÇÃO PÚBLICA (CFP)", title_style))
    elements.append(Paragraph(title_report, subtitle_style))
    elements.append(Spacer(1, 5))

    table_data = [["Naran Pessoal", "ID SIGAP", "Munisípiu", "Kargo", "Avaliasaun"]]

    for _, row in df_data.iterrows():
        table_data.append(
            [
                str(row.get("nome_pessoal", "")),
                str(row.get("id_sigap", "")),
                str(row.get("local_trabalho", "")),
                str(row.get("cargo", "")),
                str(row.get("Rezultadu_Avaliasaun", "")),
            ]
        )

    table = Table(
        table_data,
        colWidths=[150, 70, 90, 110, 80],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state["authenticated"]:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #FFFFFF !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        .block-container {
            width: 313px !important;
            max-width: 313px !important;
            min-width: 313px !important;
            padding: 0 !important;
            margin: 20px auto !important;
            background: #347FBD !important;
            border: 1px solid #D7D7D7 !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }
        .cfp-login-header {
            width: 100%;
            height: 49px;
            background: #7528A8;
            color: #FFFFFF;
            text-align: center;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 9px;
            font-weight: bold;
            line-height: 11px;
            padding-top: 5px;
            padding-left: 4px;
            padding-right: 4px;
            border-bottom: 1px solid #FFFFFF;
        }
        .cfp-logo-area {
            width: 100%;
            height: 69px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .cfp-logo-area img {
            width: 63px !important;
            height: 63px !important;
            object-fit: contain;
        }
        .cfp-favor-login {
            width: 100%;
            text-align: center;
            color: #FFFFFF;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 9px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
        }
        div[data-testid="stTextInput"] > label {
            display: none !important;
        }
        div[data-testid="stTextInput"] {
            width: 194px !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stTextInput"] input {
            width: 194px !important;
            height: 25px !important;
            background: #FFFFFF !important;
            color: #222222 !important;
            border: 1px solid #D4D4D4 !important;
            border-radius: 14px !important;
            font-family: Georgia, serif !important;
            font-size: 8px !important;
            padding-left: 12px !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            width: 80px !important;
            height: 23px !important;
            background: #7528A8 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 6px !important;
            font-size: 8px !important;
            font-weight: bold !important;
            margin: 0 auto !important;
        }
        .cfp-login-footer {
            width: 100%;
            text-align: center;
            color: #FFFFFF;
            font-family: Arial, sans-serif;
            font-size: 6px;
            padding: 9px 0 5px 0;
            opacity: 0.9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cfp-login-header">
            SISTEMA KLASIFIKASAUN AUTO DETERMINA<br>
            FUNSIONARIU NE'EBÉ MERESE ATU KOMPETE BA<br>
            PROMOSAUN GERAL
        </div>
        """,
        unsafe_allow_html=True,
    )

    logo_path = BASE_DIR / "logo_cfp.png"
    if not logo_path.exists():
        logo_path = BASE_DIR / "logo cfp.png"

    if logo_path.exists():
        st.markdown('<div class="cfp-logo-area">', unsafe_allow_html=True)
        st.image(str(logo_path), width=63)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cfp-logo-area"><span style="font-size:50px;">🏛️</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="cfp-favor-login">FAVOR LOGIN:</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        col_label_1, col_input_1 = st.columns([1.0, 2.75], gap="small")
        with col_label_1:
            st.markdown('<div style="color:white;font-size:9px;font-weight:bold;padding-left:20px;padding-top:5px;">Username:</div>', unsafe_allow_html=True)
        with col_input_1:
            username = st.text_input("Username", label_visibility="collapsed")

        col_label_2, col_input_2 = st.columns([1.0, 2.75], gap="small")
        with col_label_2:
            st.markdown('<div style="color:white;font-size:9px;font-weight:bold;padding-left:20px;padding-top:5px;">Password:</div>', unsafe_allow_html=True)
        with col_input_2:
            password = st.text_input("Password", type="password", label_visibility="collapsed")

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)
        submit_login = st.form_submit_button("LOGIN")

    if submit_login:
        try:
            secret_username = st.secrets["username"]
            secret_password = st.secrets["password"]
            if username.strip() == secret_username and password == secret_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("⚠️ Username ka Password sala! Favor koko fali.")
        except Exception:
            st.error("⚠️ Konfigurasaun Secrets seidauk iha Streamlit Cloud ka lokál.")

    st.markdown(
        """
        <div class="cfp-login-footer">
            © 2026 Comissão da Função Pública - RDTL.<br>All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# APLIKASAUN PRINSIPAL (PÓS-LOGIN)
# ============================================================

st.sidebar.markdown("### 🏛️ CFP-RDTL Portal")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Kargu Asesu")

if st.sidebar.button("🚪 Logout / Sai", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown(
    """
    <div style="background:#1E3A8A;padding:18px;border-radius:10px;margin-bottom:15px;">
        <div style="color:white;text-align:center;font-size:24px;font-weight:bold;">
            📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP
        </div>
        <div style="color:#DBEAFE;text-align:center;font-size:13px;margin-top:5px;">
            Aplikasaun Intelijénsia Artifisiál uza algoritmu Decision Tree
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

init_db()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Gestaun Dataset")
uploaded_file = st.sidebar.file_uploader("Upload ficheiru Excel (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("👋 Favór upload ficheiru Excel (.xlsx) iha sidebar hodi hahú eksplora sistema klasifikasaun.")
    st.markdown(
        """
        ### Estrutura Sistema
        1. 📁 Upload Dataset Excel
        2. 🧹 Data Cleaning
        3. 📊 Seleksaun Atributu
        4. 🌳 Decision Tree
        5. 📈 Avaliasaun Modelu
        6. 🔮 Prediksaun Funsionáriu
        7. 💾 Database
        8. 📄 Relatóriu PDF / CSV
        """
    )
    st.stop()

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

    df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)

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

    missing_cols = [col for col in nota_cols + [target_col] if col not in df_raw.columns]
    if len(missing_cols) > 0:
        st.error("⚠️ Falta koluna: " + ", ".join(missing_cols))
        st.stop()

    df_base = df_raw.dropna(subset=nota_cols + [target_col]).copy()
    for col in nota_cols:
        df_base[col] = pd.to_numeric(df_base[col], errors="coerce")
    df_base = df_base.dropna(subset=nota_cols)

    extra_records = load_extra_from_db()
    if len(extra_records) > 0:
        df_extra = pd.DataFrame(extra_records)
        df = pd.concat([df_base, df_extra], ignore_index=True)
        if "id_sigap" in df.columns:
            df = df.drop_duplicates(subset=["id_sigap"], keep="last")
    else:
        df = df_base

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filtru Globál Dadus")
    if "cargo" in df.columns:
        cargo_values = df["cargo"].dropna().astype(str).unique().tolist()
        cargo_list = ["Tomak (Hotu-hotu)"] + sorted(cargo_values)
    else:
        cargo_list = ["Tomak (Hotu-hotu)"]

    selected_cargo = st.sidebar.selectbox("Filtru Kargo:", cargo_list)
    df_filtered = df.copy()
    if selected_cargo != "Tomak (Hotu-hotu)":
        df_filtered = df_filtered[df_filtered["cargo"] == selected_cargo]

    st.sidebar.markdown("---")
    csv_full = df_filtered.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="⬇️ Download Backup (CSV)",
        data=csv_full,
        file_name="dataset_cfp_filtrado.csv",
        mime="text/csv",
    )

    model, le, X_train, X_test, y_train, y_test = treinar_modelo(df, nota_cols, target_col)

    pred_encoded = model.predict(df_filtered[nota_cols])
    df_filtered["Prediksaun"] = le.inverse_transform(pred_encoded)

    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Dashboard Analítiku",
            "⚙️ Modelu & Performance",
            "🔮 Predisaun & Gestaun Dadus",
        ]
    )

    # ========================================================
    # TAB 1: DASHBOARD
    # ========================================================
    with tab1:
        st.markdown("### 📈 Sumáriu Dezempenu Funsionáriu")
        total_funs = len(df_filtered)
        counts_real = df_filtered[target_col].value_counts()

        mb_count = counts_real.get("Muito Bom", 0)
        b_count = counts_real.get("Bom", 0)
        s_count = counts_real.get("Suficiente", 0)
        i_count = counts_real.get("Insuficiente", 0)

        mb_pct = (mb_count / total_funs * 100) if total_funs > 0 else 0
        b_pct = (b_count / total_funs * 100) if total_funs > 0 else 0
        s_pct = (s_count / total_funs * 100) if total_funs > 0 else 0
        i_pct = (i_count / total_funs * 100) if total_funs > 0 else 0

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

        with col_m1:
            if st.button(f"📊 Total\n\n{total_funs}", key="btn_m1", use_container_width=True):
                st.session_state["selected_category"] = "Tomak" if st.session_state["selected_category"] != "Tomak" else None
        with col_m2:
            if st.button(f"⭐ Muito Bom\n\n{mb_count}\n({mb_pct:.1f}%)", key="btn_m2", use_container_width=True):
                st.session_state["selected_category"] = "Muito Bom" if st.session_state["selected_category"] != "Muito Bom" else None
        with col_m3:
            if st.button(f"👍 Bom\n\n{b_count}\n({b_pct:.1f}%)", key="btn_m3", use_container_width=True):
                st.session_state["selected_category"] = "Bom" if st.session_state["selected_category"] != "Bom" else None
        with col_m4:
            if st.button(f"⚠️ Suficiente\n\n{s_count}\n({s_pct:.1f}%)", key="btn_m4", use_container_width=True):
                st.session_state["selected_category"] = "Suficiente" if st.session_state["selected_category"] != "Suficiente" else None
        with col_m5:
            if st.button(f"❌ Insuficiente\n\n{i_count}\n({i_pct:.1f}%)", key="btn_m5", use_container_width=True):
                st.session_state["selected_category"] = "Insuficiente" if st.session_state["selected_category"] != "Insuficiente" else None

        df_display = df_filtered.copy()
        if st.session_state["selected_category"] and st.session_state["selected_category"] != "Tomak":
            df_display = df_display[df_display[target_col] == st.session_state["selected_category"]]
            st.info(f"🔍 Hatudu dadus ba kategoria: **{st.session_state['selected_category']}**")

        st.markdown("---")
        st.subheader("📋 Tabela Dadus Funsionáriu & Predisaun")
        st.dataframe(df_display, use_container_width=True)

        csv_display = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Tabela ne'ebé Hili (CSV)",
            data=csv_display,
            file_name="tabela_funsionariu_filtrado.csv",
            mime="text/csv",
        )

    # ========================================================
    # TAB 2: MODELU & PERFORMANCE
    # ========================================================
    with tab2:
        st.markdown("### ⚙️ Evaluasaun Modelu Decision Tree")
        st.write(f"**Acuracy Modelu (Test Set):** `{acc * 100:.2f}%`")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### 📊 Confusion Matrix")
            y_pred_cm = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_cm)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, xticklabels=le.classes_, yticklabels=le.classes_)
            ax_cm.set_xlabel("Predisaun")
            ax_cm.set_ylabel("Real")
            st.pyplot(fig_cm)

        with col_p2:
            st.markdown("#### 📄 Classification Report")
            report_dict = classification_report(y_test, y_pred_cm, output_dict=True)
            df_report = pd.DataFrame(report_dict).transpose()
            st.dataframe(df_report, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🌳 Visualizasaun Árvore de Decisão")
        fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
        plot_tree(
            model,
            feature_names=nota_cols,
            class_names=[str(c) for c in le.classes_],
            filled=True,
            rounded=True,
            ax=ax_tree,
        )
        st.pyplot(fig_tree)

    # ========================================================
    # TAB 3: PREDISAUN & GESTAUN DADUS
    # ========================================================
    with tab3:
        st.markdown("### 🔮 Predisaun Funsionáriu Foun & Gestaun Dadus")

        with st.form("form_predisaun_foun"):
            st.markdown("#### Hatama Dadus Avaliasaun Funsionáriu")
            c1, c2, c3 = st.columns(3)
            with c1:
                f_nome = st.text_input("Naran Pessoal")
                f_id = st.text_input("ID SIGAP")
                f_mun = st.text_input("Munisípiu / Local Trabalho")
            with c2:
                f_cargo = st.text_input("Kargo")
                f_asid = st.number_input("Asiduidade", 0.0, 20.0, 15.0)
                f_pont = st.number_input("Pontualidade", 0.0, 20.0, 15.0)
            with c3:
                f_prod = st.number_input("Produtividade", 0.0, 20.0, 15.0)
                f_kual = st.number_input("Kualidade Servisu", 0.0, 20.0, 15.0)
                f_koop = st.number_input("Kooperasaun", 0.0, 20.0, 15.0)

            c4, c5 = st.columns(2)
            with c4:
                f_inis = st.number_input("Inisiativa", 0.0, 20.0, 15.0)
                f_disp = st.number_input("Disiplina", 0.0, 20.0, 15.0)
            with c5:
                f_resp = st.number_input("Responsabilidade", 0.0, 20.0, 15.0)

            submit_pred = st.form_submit_button("PREDIZ & GERA RELATÓRIU")

        if submit_pred:
            input_vals = [[f_asid, f_pont, f_prod, f_kual, f_koop, f_inis, f_disp, f_resp]]
            pred_enc = model.predict(input_vals)
            pred_label = le.inverse_transform(pred_enc)[0]

            st.success(f"🎯 Rezultadu Predisaun Avaliasaun: **{pred_label}**")

            new_record = {
                "nome_pessoal": f_nome,
                "id_sigap": f_id,
                "local_trabalho": f_mun,
                "cargo": f_cargo,
                "Asiduidade": f_asid,
                "Pontualidade": f_pont,
                "Produtividade": f_prod,
                "Kualidade_Servisu": f_kual,
                "Kooperasaun": f_koop,
                "Inisiativa": f_inis,
                "Disiplina": f_disp,
                "Responsabilidade": f_resp,
                "Rezultadu_Avaliasaun": pred_label,
            }
            save_extra_to_db(new_record)
            st.success("💾 Dadus rai ona ho suksesu iha Database!")

        st.markdown("---")
        st.markdown("### 📄 Gera Relatóriu PDF Ofisiál")
        if st.button("📥 Download Relatóriu PDF (Hotu-hotu)"):
            pdf_buffer = generate_pdf_report(df_filtered, "RELATÓRIU OFISIÁL KLASIFIKASAUN DEZEMPENU CFP")
            st.download_button(
                label="Baixar Ficheiru PDF",
                data=pdf_buffer,
                file_name="relatorio_oficial_cfp.pdf",
                mime="application/pdf",
            )

except Exception as e:
    st.error(f"⚠️ Hutur lia-fuan ka erro iha prosesamentu dadus: {e}")

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st

from database import (
    delete_extra_from_db_by_index,
    init_db,
    load_extra_from_db,
    save_extra_to_db,
    update_extra_in_db_by_index,
)
from models import treinar_modelo


# ============================================================
# KONFIGURASAUN
# ============================================================

st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - RDTL",
    page_icon="🏛️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

# Uza logo_cfp.png. Se seidauk troka naran, code mos buka logo cfp.png.
LOGO_CANDIDATES = [
    BASE_DIR / "logo_cfp.png",
    BASE_DIR / "logo cfp.png",
]


def find_logo():
    for path in LOGO_CANDIDATES:
        if path.exists() and path.is_file():
            return path
    return None


LOGO_PATH = find_logo()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    .login-page {
        min-height: 80vh;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding-top: 45px;
    }

    .login-card {
        background: #ffffff;
        padding: 32px 36px 25px 36px;
        border-radius: 14px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
        border-top: 6px solid #D97706;
        width: 100%;
        max-width: 460px;
        margin: 0 auto;
    }

    .logo-box {
        text-align: center;
        margin-bottom: 15px;
    }

    .logo-box img {
        width: 105px;
        height: 105px;
        object-fit: contain;
        display: block;
        margin: 0 auto;
    }

    .cfp-header-title {
        color: #1E3A8A;
        font-weight: 800;
        font-size: 15px;
        text-align: center;
        line-height: 1.5;
        margin-bottom: 5px;
    }

    .cfp-subtitle {
        color: #64748B;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 18px;
    }

    .login-footer {
        color: #94A3B8;
        text-align: center;
        font-size: 11px;
        margin-top: 18px;
    }

    .main-title {
        color: white;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .sub-title {
        color: #CBD5E1;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,.15);
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 700;
    }

    label {
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNSAUN LEE DATA
# ============================================================

def load_data(file):
    return pd.read_excel(file)


# ============================================================
# LOGIN
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


def get_login_credentials():
    """
    Lee username/password husi Streamlit secrets.
    """
    try:
        username = st.secrets["username"]
        password = st.secrets["password"]
        return str(username), str(password)
    except Exception:
        return "", ""


if not st.session_state["authenticated"]:

    st.markdown('<div class="login-page">', unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    # ----------------------------
    # LOGO
    # ----------------------------
    if LOGO_PATH is not None:
        st.markdown('<div class="logo-box">', unsafe_allow_html=True)
        st.image(str(LOGO_PATH), width=105)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="logo-box">
                <div style="font-size:72px;">🏛️</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "Logo CFP la hetan. Tau `logo_cfp.png` iha pasta hanesan ho `app.py`."
        )

    # ----------------------------
    # TITULU
    # ----------------------------
    st.markdown(
        """
        <div class="cfp-header-title">
            COMISSÃO DA FUNÇÃO PÚBLICA<br>
            REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE
        </div>

        <div class="cfp-subtitle">
            Portal de Gestão e Classificação de Desempenho
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # LOGIN FORM
    # ----------------------------
    with st.form("login_form", clear_on_submit=False):

        username = st.text_input(
            "Username:",
            placeholder="Hatama ita-nia username",
        )

        password = st.text_input(
            "Password:",
            type="password",
            placeholder="Hatama ita-nia password",
        )

        submit_login = st.form_submit_button(
            "ENTRADA / LOGIN",
            use_container_width=True,
        )

    if submit_login:

        correct_username, correct_password = get_login_credentials()

        if not correct_username or not correct_password:
            st.error(
                "⚠️ Username no password seidauk konfiguradu iha "
                "Streamlit Secrets."
            )
            st.info(
                "Configura `[username]` no `[password]` iha Secrets."
            )

        elif (
            username.strip() == correct_username
            and password == correct_password
        ):
            st.session_state["authenticated"] = True
            st.success("Login susesu!")
            st.rerun()

        else:
            st.error(
                "⚠️ Username ka Password sala! Favor koko fali."
            )

    st.markdown(
        """
        <div class="login-footer">
            © 2026 Comissão da Função Pública - RDTL.<br>
            All rights reserved.
        </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# DATABASE
# ============================================================

init_db()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None


# ============================================================
# PDF
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

    elements.append(
        Paragraph(
            "REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE",
            title_style,
        )
    )

    elements.append(
        Paragraph(
            "COMISSÃO DA FUNÇÃO PÚBLICA (CFP)",
            title_style,
        )
    )

    elements.append(
        Paragraph(title_report, subtitle_style)
    )

    elements.append(Spacer(1, 5))

    table_data = [
        [
            "Naran Pessoal",
            "ID SIGAP",
            "Munisípiu",
            "Kargo",
            "Avaliasaun",
        ]
    ]

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
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1E3A8A"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.whitesmoke,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    9,
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#F8FAFC"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#CBD5E1"),
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)

    return buffer


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### 🏛️ CFP-RDTL Portal")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Kargu Asesu")

if st.sidebar.button(
    "🚪 Logout / Sai",
    use_container_width=True,
):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Gestaun Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload ficheiru Excel (.xlsx)",
    type=["xlsx"],
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Aplikasaun Intelijénsia Artifisiál uza algoritmu '
    'Decision Tree bazeia ba indikadór Komisaun Função Pública RDTL.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SEIDAK UPLOAD
# ============================================================

if uploaded_file is None:

    st.info(
        "👋 Favór upload ficheiru Excel (.xlsx) iha sidebar "
        "hodi hahú eksplora sistema klasifikasaun."
    )

    st.markdown(
        """
        ### 📌 Estrutura Dataset

        Sistema espera koluna sira:

        - Asiduidade
        - Pontualidade
        - Produtividade
        - Kualidade_Servisu
        - Kooperasaun
        - Inisiativa
        - Disiplina
        - Responsabilidade
        - Rezultadu_Avaliasaun
        """
    )

    st.stop()


# ============================================================
# PROCESSAMENTO DATASET
# ============================================================

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
        columns={
            k: v
            for k, v in rename_map.items()
            if k in df_raw.columns
        },
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
        col
        for col in nota_cols + [target_col]
        if col not in df_raw.columns
    ]

    if missing_cols:
        st.error(
            "⚠️ Falta koluna iha dataset: "
            + ", ".join(missing_cols)
        )
        st.stop()

    # ----------------------------
    # Cleaning
    # ----------------------------

    df_base = df_raw.copy()

    for col in nota_cols:
        df_base[col] = pd.to_numeric(
            df_base[col],
            errors="coerce",
        )

    df_base = df_base.dropna(
        subset=nota_cols + [target_col]
    ).copy()

    # ----------------------------
    # Extra records
    # ----------------------------

    try:
        extra_records = load_extra_from_db()
    except Exception:
        extra_records = []

    st.session_state["extra_reports"] = extra_records

    if extra_records:

        df_extra = pd.DataFrame(extra_records)

        for col in nota_cols:
            if col in df_extra.columns:
                df_extra[col] = pd.to_numeric(
                    df_extra[col],
                    errors="coerce",
                )

        df = pd.concat(
            [df_base, df_extra],
            ignore_index=True,
        )

        if "id_sigap" in df.columns:
            df = df.drop_duplicates(
                subset=["id_sigap"],
                keep="last",
            )

    else:
        df = df_base.copy()


    # ========================================================
    # SIDEBAR FILTER
    # ========================================================

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filtru Globál Dadus")

    if "cargo" in df.columns:

        cargo_values = (
            df["cargo"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        cargo_list = [
            "Tomak (Hotu-hotu)"
        ] + sorted(cargo_values)

    else:
        cargo_list = ["Tomak (Hotu-hotu)"]

    selected_cargo = st.sidebar.selectbox(
        "Filtru Kargo:",
        cargo_list,
    )

    df_filtered = df.copy()

    if (
        selected_cargo != "Tomak (Hotu-hotu)"
        and "cargo" in df_filtered.columns
    ):
        df_filtered = df_filtered[
            df_filtered["cargo"] == selected_cargo
        ]


    # ========================================================
    # BACKUP CSV
    # ========================================================

    csv_full = (
        df_filtered
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.sidebar.download_button(
        label="⬇️ Download Backup (CSV)",
        data=csv_full,
        file_name="dataset_cfp_filtrado.csv",
        mime="text/csv",
    )


    # ========================================================
    # TREINAMENTU MODELU
    # ========================================================

    try:

        (
            model,
            le,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = treinar_modelo(
            df,
            nota_cols,
            target_col,
        )

    except Exception as model_error:

        st.error(
            "⚠️ Modelu la bele treinamentu."
        )

        st.exception(model_error)

        st.stop()


    # ========================================================
    # PREDIKSAUN DATASET
    # ========================================================

    try:

        pred_encoded_all = model.predict(
            df_filtered[nota_cols]
        )

        df_filtered["Prediksaun"] = (
            le.inverse_transform(
                pred_encoded_all
            )
        )

    except Exception as pred_error:

        st.error(
            "⚠️ Erro iha prediksaun dataset."
        )

        st.exception(pred_error)

        st.stop()


    y_pred_test = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred_test,
    )


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Dashboard Analítiku",
            "⚙️ Modelu & Performance",
            "🔮 Prediksaun & Gestaun Dadus",
        ]
    )


    # ========================================================
    # TAB 1
    # ========================================================

    with tab1:

        st.markdown(
            "### 📈 Sumáriu Dezempenu Funsionáriu"
        )

        total_funs = len(df_filtered)

        counts_real = (
            df_filtered[target_col]
            .value_counts()
        )

        categories = [
            "Muito Bom",
            "Bom",
            "Suficiente",
            "Insuficiente",
        ]

        mb = counts_real.get("Muito Bom", 0)
        bom = counts_real.get("Bom", 0)
        suf = counts_real.get("Suficiente", 0)
        ins = counts_real.get("Insuficiente", 0)

        mb_pct = (
            mb / total_funs * 100
            if total_funs
            else 0
        )

        bom_pct = (
            bom / total_funs * 100
            if total_funs
            else 0
        )

        suf_pct = (
            suf / total_funs * 100
            if total_funs
            else 0
        )

        ins_pct = (
            ins / total_funs * 100
            if total_funs
            else 0
        )


        # ----------------------------
        # KPI
        # ----------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button(
                f"📊 Total\n{total_funs}",
                key="btn_total",
                use_container_width=True,
            ):
                if (
                    st.session_state["selected_category"]
                    == "Tomak"
                ):
                    st.session_state["selected_category"] = None
                else:
                    st.session_state["selected_category"] = "Tomak"


        with col2:
            if st.button(
                f"⭐ Muito Bom\n{mb} ({mb_pct:.1f}%)",
                key="btn_muito_bom",
                use_container_width=True,
            ):
                st.session_state["selected_category"] = (
                    None
                    if st.session_state["selected_category"]
                    == "Muito Bom"
                    else "Muito Bom"
                )


        with col3:
            if st.button(
                f"✨ Bom\n{bom} ({bom_pct:.1f}%)",
                key="btn_bom",
                use_container_width=True,
            ):
                st.session_state["selected_category"] = (
                    None
                    if st.session_state["selected_category"]
                    == "Bom"
                    else "Bom"
                )


        with col4:
            if st.button(
                f"📌 Suficiente\n{suf} ({suf_pct:.1f}%)",
                key="btn_suficiente",
                use_container_width=True,
            ):
                st.session_state["selected_category"] = (
                    None
                    if st.session_state["selected_category"]
                    == "Suficiente"
                    else "Suficiente"
                )


        with col5:
            if st.button(
                f"⚠️ Insuficiente\n{ins} ({ins_pct:.1f}%)",
                key="btn_insuficiente",
                use_container_width=True,
            ):
                st.session_state["selected_category"] = (
                    None
                    if st.session_state["selected_category"]
                    == "Insuficiente"
                    else "Insuficiente"
                )


        # ----------------------------
        # TABELA KATEGORIA
        # ----------------------------

        selected_cat = (
            st.session_state["selected_category"]
        )

        if selected_cat is not None:

            st.markdown("---")

            if selected_cat == "Tomak":

                df_table = df_filtered

                st.markdown(
                    f"### 📋 Lista Funsionáriu Tomak "
                    f"({len(df_table)})"
                )

            else:

                df_table = df_filtered[
                    df_filtered[target_col]
                    == selected_cat
                ]

                st.markdown(
                    f"### 📋 Lista Funsionáriu "
                    f"ba Kategoria `{selected_cat}` "
                    f"({len(df_table)})"
                )


            preferred_cols = [
                "controlo_ativo_identificacao",
                "nome_pessoal",
                "id_sigap",
                "id_grp",
                "sexo",
                "local_trabalho",
                "cargo",
                target_col,
            ]

            table_cols = [
                col
                for col in preferred_cols
                if col in df_table.columns
            ]

            st.dataframe(
                df_table[table_cols],
                use_container_width=True,
            )


            dl1, dl2 = st.columns(2)

            with dl1:

                csv_filtered = (
                    df_table
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_filtered,
                    file_name=(
                        f"relatorio_cfp_"
                        f"{selected_cat}.csv"
                    ),
                    mime="text/csv",
                    key="download_category_csv",
                )


            with dl2:

                pdf_buffer = generate_pdf_report(
                    df_table,
                    f"Relatóriu Dezempenu "
                    f"Funsionáriu - {selected_cat}",
                )

                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_buffer,
                    file_name=(
                        f"relatorio_cfp_"
                        f"{selected_cat}.pdf"
                    ),
                    mime="application/pdf",
                    key="download_category_pdf",
                )


            if st.button(
                "❌ Subar Tabela",
                key="hide_category_table",
            ):
                st.session_state[
                    "selected_category"
                ] = None
                st.rerun()


        # ----------------------------
        # GRAFIKU
        # ----------------------------

        st.markdown("---")

        g1, g2 = st.columns(2)

        with g1:

            st.markdown(
                "##### 📊 Komparasaun Kategoria "
                "(Reál vs Prediksaun)"
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            real_counts = [
                counts_real.get(
                    cat,
                    0,
                )
                for cat in categories
            ]

            pred_counts = [
                df_filtered[
                    "Prediksaun"
                ]
                .value_counts()
                .get(
                    cat,
                    0,
                )
                for cat in categories
            ]

            x = np.arange(
                len(categories)
            )

            width = 0.35

            r1 = ax.bar(
                x - width / 2,
                real_counts,
                width,
                label="Dadus Reál",
            )

            r2 = ax.bar(
                x + width / 2,
                pred_counts,
                width,
                label="Prediksaun Tree",
            )

            ax.bar_label(
                r1,
                padding=3,
                fontsize=8,
            )

            ax.bar_label(
                r2,
                padding=3,
                fontsize=8,
            )

            ax.set_ylabel(
                "Total Funsionáriu"
            )

            ax.set_xticks(x)

            ax.set_xticklabels(
                categories,
                rotation=15,
            )

            ax.legend()

            sns.despine()

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)


        with g2:

            st.markdown(
                "##### 🍩 Proporsaun "
                "Kategoria Dezempenu"
            )

            fig2, ax2 = plt.subplots(
                figsize=(6, 4)
            )

            sizes = [
                counts_real.get(
                    cat,
                    0,
                )
                for cat in categories
            ]

            if sum(sizes) > 0:

                ax2.pie(
                    sizes,
                    labels=categories,
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={
                        "width": 0.4,
                        "edgecolor": "white",
                        "linewidth": 2,
                    },
                )

            else:

                ax2.text(
                    0.5,
                    0.5,
                    "La iha dadus",
                    ha="center",
                    va="center",
                )

            st.pyplot(
                fig2,
                use_container_width=True,
            )

            plt.close(fig2)


        # ----------------------------
        # GRAFIKU LOCAL
        # ----------------------------

        st.markdown("---")

        st.markdown(
            "##### 🗺️ Distribuisaun Dezempenu "
            "tuir Local de Trabalhu"
        )

        if "local_trabalho" in df_filtered.columns:

            df_loc_counts = pd.crosstab(
                df_filtered["local_trabalho"],
                df_filtered[target_col],
            )

            existing_cats = [
                c
                for c in categories
                if c in df_loc_counts.columns
            ]

            if existing_cats:

                df_loc_counts = (
                    df_loc_counts
                    .reindex(
                        columns=existing_cats,
                        fill_value=0,
                    )
                )

                fig_loc, ax_loc = plt.subplots(
                    figsize=(10, 4.5)
                )

                df_loc_counts.plot(
                    kind="bar",
                    stacked=True,
                    ax=ax_loc,
                    colormap="crest",
                    edgecolor="none",
                )

                ax_loc.set_xlabel(
                    "Local de Trabalhu"
                )

                ax_loc.set_ylabel(
                    "Total Funsionáriu"
                )

                plt.xticks(
                    rotation=45,
                    ha="right",
                )

                ax_loc.legend(
                    title="Kategoria",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )

                sns.despine()

                st.pyplot(
                    fig_loc,
                    use_container_width=True,
                )

                plt.close(fig_loc)


    # ========================================================
    # TAB 2
    # ========================================================

    with tab2:

        st.subheader(
            "📋 Amostra Dadus"
        )

        st.dataframe(
            df_filtered.head(10),
            use_container_width=True,
        )

        st.markdown("---")

        st.subheader(
            "🚀 Performance Modelu Decision Tree"
        )

        st.success(
            f"✅ Akurasi Modelu (Accuracy): "
            f"**{acc * 100:.2f}%**"
        )

        e1, e2 = st.columns(2)

        with e1:

            st.markdown(
                "##### 📉 Confusion Matrix"
            )

            cm = confusion_matrix(
                y_test,
                y_pred_test,
            )

            fig_cm, ax_cm = plt.subplots(
                figsize=(5, 4)
            )

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

            st.pyplot(
                fig_cm,
                use_container_width=True,
            )

            plt.close(fig_cm)


        with e2:

            st.markdown(
                "##### 📑 Classification Report"
            )

            unique_labels = np.unique(
                np.concatenate(
                    (
                        y_test,
                        y_pred_test,
                    )
                )
            )

            present_class_names = [
                le.classes_[i]
                for i in unique_labels
            ]

            report_dict = classification_report(
                y_test,
                y_pred_test,
                labels=unique_labels,
                target_names=present_class_names,
                output_dict=True,
                zero_division=0,
            )

            df_report = pd.DataFrame(
                report_dict
            ).transpose()

            st.dataframe(
                df_report,
                use_container_width=True,
            )


        st.markdown("---")

        st.subheader(
            "🌳 Vizualizasaun Árbore Desizaun"
        )

        max_depth_vis = st.slider(
            "Hili Profundidade Árbore (Max Depth)",
            1,
            5,
            3,
            key="tree_depth_slider",
        )

        vis_model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=max_depth_vis,
            random_state=42,
        )

        vis_model.fit(
            X_train,
            y_train,
        )

        fig_tree, ax_tree = plt.subplots(
            figsize=(16, 9),
            dpi=100,
        )

        plot_tree(
            vis_model,
            feature_names=nota_cols,
            class_names=le.classes_,
            filled=True,
            rounded=True,
            ax=ax_tree,
            fontsize=9,
        )

        st.pyplot(
            fig_tree,
            use_container_width=True,
        )

        plt.close(fig_tree)


    # ========================================================
    # TAB 3
    # ========================================================

    with tab3:

        st.subheader(
            "🔍 Prediksaun Funsionáriu Foun "
            "& Gestaun Dadus"
        )

        extra_records = load_extra_from_db()

        st.session_state[
            "extra_reports"
        ] = extra_records


        # ----------------------------------------------------
        # LISTA RECORDS
        # ----------------------------------------------------

        st.markdown(
            "##### 📋 Lista Dadus Funsionáriu "
            "Foun & Asaun Gestaun"
        )

        if extra_records:

            for i, rec in enumerate(
                extra_records
            ):

                c1, c2, c3, c4 = st.columns(
                    [3, 2, 2.5, 2.5]
                )

                with c1:
                    st.write(
                        f"👤 **{rec.get('nome_pessoal', '')}**"
                    )
                    st.code(
                        str(
                            rec.get(
                                "id_sigap",
                                "",
                            )
                        ),
                        language=None,
                    )

                with c2:
                    st.write(
                        f"📍 {rec.get('local_trabalho', '')}"
                    )

                with c3:
                    st.write(
                        f"💼 {rec.get('cargo', '')}"
                    )

                with c4:

                    result = rec.get(
                        "Rezultadu_Avaliasaun",
                        "N/A",
                    )

                    b1, b2, b3 = st.columns(
                        [1.5, 0.7, 0.7]
                    )

                    with b1:
                        st.markdown(
                            f"⭐ **{result}**"
                        )

                    with b2:

                        if st.button(
                            "✏️",
                            key=f"edit_{i}",
                        ):
                            st.session_state[
                                "edit_index"
                            ] = i
                            st.rerun()

                    with b3:

                        if st.button(
                            "🗑️",
                            key=f"delete_{i}",
                        ):

                            try:
                                ok = (
                                    delete_extra_from_db_by_index(
                                        i
                                    )
                                )

                                if ok:
                                    st.success(
                                        "Hamos ona!"
                                    )
                                    st.rerun()
                                else:
                                    st.error(
                                        "Falha atu hamos."
                                    )

                            except Exception as delete_error:
                                st.error(
                                    f"Erro: {delete_error}"
                                )

                st.markdown("---")

        else:

            st.info(
                "ℹ️ Sei la iha dadus foun "
                "rejistadu iha database."
            )


        # ----------------------------------------------------
        # FORM INPUT
        # ----------------------------------------------------

        idx_edit = (
            st.session_state["edit_index"]
        )

        def_val = {}

        if (
            idx_edit is not None
            and 0 <= idx_edit < len(extra_records)
        ):

            def_val = extra_records[idx_edit]

            st.markdown(
                f"#### ✏️ Atualiza Dadus "
                f"Funsionáriu (Index {idx_edit})"
            )

        else:

            st.markdown(
                "#### ➕ Input Funsionáriu Foun "
                "ba Prediksaun"
            )


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

        funcoes = [
            "Regime Geral das Carreiras, Técnico Superior Grau B, 10, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 1, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 4, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau C, 1, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau A, 4, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau C, 5, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 3, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 6, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 5, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 1, NOMEAÇÃO PROBATÓRIA",
            "Regime Geral das Carreiras, Assistente Grau F, 5, PERMANENTE",
            "Regime Geral das Carreiras, Assistente Grau F, 3, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 1, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 1, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau A, 3, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 2, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 1, NOMEAÇÃO PROBATÓRIA",
            "Regime Geral das Carreiras, Técnico Profissional Grau C, 1, NOMEAÇÃO PROBATÓRIA",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 2, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau A, 8, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 7, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau A, 2, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau A, 1, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 2, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 3, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau C, 3, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 4, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau C, 4, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau C, 2, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 4, PERMANENTE",
            "Regime Geral das Carreiras, Assistente Grau F, 2, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Profissional Grau D, 5, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 5, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 7, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 3, PERMANENTE",
            "Regime Geral das Carreiras, Técnico Superior Grau B, 9, PERMANENTE",
        ]


        with st.form(
            "funsionariu_form",
            clear_on_submit=False,
        ):

            st.markdown(
                "##### 📝 1. Informasaun Identidade"
            )

            i1, i2, i3 = st.columns(3)

            with i1:

                active_opts = [
                    "Ativo",
                    "La Ativo",
                ]

                current_active = def_val.get(
                    "controlo_ativo_identificacao",
                    "Ativo",
                )

                txt_ativo = st.selectbox(
                    "Controlo Ativo Identifikasaun",
                    active_opts,
                    index=(
                        active_opts.index(
                            current_active
                        )
                        if current_active in active_opts
                        else 0
                    ),
                )

                txt_nome = st.text_input(
                    "Naran Pessoal*",
                    def_val.get(
                        "nome_pessoal",
                        "",
                    ),
                )

                txt_sigap = st.text_input(
                    "ID SIGAP (Numeriku no Símbolu)*",
                    def_val.get(
                        "id_sigap",
                        "",
                    ),
                )

                current_sex = def_val.get(
                    "sexo",
                    "M",
                )

                txt_sexo = st.selectbox(
                    "Sexo",
                    ["M", "F"],
                    index=(
                        0
                        if current_sex == "M"
                        else 1
                    ),
                )


            with i2:

                txt_inst = st.text_input(
                    "Instituisaun",
                    def_val.get(
                        "instituicao",
                        "CFP",
                    ),
                )

                current_local = def_val.get(
                    "local_trabalho",
                    "Díli",
                )

                txt_local = st.selectbox(
                    "Local Trabalhu",
                    municipios,
                    index=(
                        municipios.index(
                            current_local
                        )
                        if current_local in municipios
                        else 0
                    ),
                )

                try:
                    default_date = pd.to_datetime(
                        def_val.get(
                            "data_de_nascimento",
                            "1995-01-01",
                        )
                    ).date()
                except Exception:
                    default_date = pd.Timestamp(
                        "1995-01-01"
                    ).date()

                txt_nascimento = st.date_input(
                    "Data de Nascimento",
                    value=default_date,
                )


            with i3:

                current_func = def_val.get(
                    "funcao",
                    funcoes[0],
                )

                txt_funcao = st.selectbox(
                    "Funsaun",
                    funcoes,
                    index=(
                        funcoes.index(
                            current_func
                        )
                        if current_func in funcoes
                        else 0
                    ),
                )

                current_cargo = def_val.get(
                    "cargo",
                    cargos[0],
                )

                txt_cargo = st.selectbox(
                    "Kargo",
                    cargos,
                    index=(
                        cargos.index(
                            current_cargo
                        )
                        if current_cargo in cargos
                        else 0
                    ),
                )

                txt_grp = st.text_input(
                    "ID GRP",
                    def_val.get(
                        "id_grp",
                        "",
                    ),
                )


            st.markdown(
                "##### 📊 2. Indikadór Avaliasaun"
            )

            a1, a2, a3, a4 = st.columns(4)

            with a1:

                p_asid = st.slider(
                    "Asiduidade",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Asiduidade",
                            4.0,
                        )
                    ),
                    0.5,
                )

                p_pont = st.slider(
                    "Pontualidade",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Pontualidade",
                            4.0,
                        )
                    ),
                    0.5,
                )


            with a2:

                p_prod = st.slider(
                    "Produtividade",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Produtividade",
                            4.0,
                        )
                    ),
                    0.5,
                )

                p_kual = st.slider(
                    "Kualidade Servisu",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Kualidade_Servisu",
                            4.0,
                        )
                    ),
                    0.5,
                )


            with a3:

                p_koop = st.slider(
                    "Kooperasaun",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Kooperasaun",
                            4.0,
                        )
                    ),
                    0.5,
                )

                p_inis = st.slider(
                    "Inisiativa",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Inisiativa",
                            4.0,
                        )
                    ),
                    0.5,
                )


            with a4:

                p_disp = st.slider(
                    "Disiplina",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Disiplina",
                            4.0,
                        )
                    ),
                    0.5,
                )

                p_resp = st.slider(
                    "Responsabilidade",
                    1.0,
                    5.0,
                    float(
                        def_val.get(
                            "Responsabilidade",
                            4.0,
                        )
                    ),
                    0.5,
                )


            submit_pred = st.form_submit_button(
                "💾 Salva / Prediksaun",
                use_container_width=True,
            )


        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        if submit_pred:

            if not txt_nome.strip():

                st.error(
                    "⚠️ Favór preenxe Naran Pessoal."
                )

            elif not txt_sigap.strip():

                st.error(
                    "⚠️ Favór preenxe ID SIGAP."
                )

            elif any(
                char.isalpha()
                for char in txt_sigap
            ):

                st.error(
                    "⚠️ ID SIGAP labele uza letra. "
                    "Uza númeru no símbolu de'it."
                )

            else:

                input_data = np.array(
                    [
                        [
                            p_asid,
                            p_pont,
                            p_prod,
                            p_kual,
                            p_koop,
                            p_inis,
                            p_disp,
                            p_resp,
                        ]
                    ]
                )

                try:

                    pred_encoded = model.predict(
                        input_data
                    )

                    pred_label = (
                        le.inverse_transform(
                            pred_encoded
                        )[0]
                    )

                except Exception as prediction_error:

                    st.error(
                        "⚠️ Erro iha prediksaun."
                    )

                    st.exception(
                        prediction_error
                    )

                    st.stop()


                new_record = {
                    "controlo_ativo_identificacao": txt_ativo,
                    "nome_pessoal": txt_nome,
                    "id_sigap": txt_sigap,
                    "sexo": txt_sexo,
                    "instituicao": txt_inst,
                    "local_trabalho": txt_local,
                    "data_de_nascimento": str(
                        txt_nascimento
                    ),
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

                    try:

                        ok = (
                            update_extra_in_db_by_index(
                                idx_edit,
                                new_record,
                            )
                        )

                    except Exception as update_error:

                        ok = False

                        st.error(
                            f"Erro atualiza database: "
                            f"{update_error}"
                        )


                    if ok:

                        st.success(
                            f"✅ Atualiza dadus susesu! "
                            f"Prediksaun: **{pred_label}**"
                        )

                        st.session_state[
                            "edit_index"
                        ] = None

                        st.session_state[
                            "extra_reports"
                        ] = load_extra_from_db()

                        st.rerun()

                    else:

                        st.error(
                            "⚠️ Falha atu atualiza dadus."
                        )

                else:

                    try:

                        ok = save_extra_to_db(
                            new_record
                        )

                    except Exception as save_error:

                        ok = False

                        st.error(
                            f"Erro salva database: "
                            f"{save_error}"
                        )


                    if ok:

                        st.success(
                            f"✅ Salva dadus susesu! "
                            f"Prediksaun: **{pred_label}**"
                        )

                        st.session_state[
                            "extra_reports"
                        ] = load_extra_from_db()

                        st.rerun()

                    else:

                        st.error(
                            "⚠️ ID SIGAP ne'e bele iha ona "
                            "ka iha erro iha database."
                        )


except Exception as e:

    st.error(
        "⚠️ Aplikasaun hetan erro bainhira "
        "prosesamentu dataset."
    )

    st.exception(e)

# ============================================================
# APP.PY
# SISTEMA KLASIFIKASAUN DESEMPENHU FUNSIONÁRIU CFP
# MÉTODO DECISION TREE
# REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE
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
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)

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


# ============================================================
# BASE DIRECTORY
# ============================================================

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
    """
    Lee ficheiru Excel.
    """
    return pd.read_excel(file)


# ============================================================
# FUNSAUN PDF
# ============================================================

def generate_pdf_report(df_data, title_report):
    """
    Kria relatóriu PDF ofisiál CFP.
    """

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
        Paragraph(
            title_report,
            subtitle_style,
        )
    )

    elements.append(
        Spacer(1, 5)
    )

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
        colWidths=[
            150,
            70,
            90,
            110,
            80,
        ],
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
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, 0),
                    6,
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
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
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
# LOGIN PAGE CSS
# ============================================================

if not st.session_state["authenticated"]:

    st.markdown(
        """
        <style>

        /* ====================================================
           BACKGROUND
           ==================================================== */

        html,
        body,
        [data-testid="stAppViewContainer"] {

            background: #FFFFFF !important;

        }


        [data-testid="stHeader"] {

            background: transparent !important;

        }


        [data-testid="stToolbar"] {

            display: none !important;

        }


        /* ====================================================
           MAIN LOGIN CONTAINER
           ==================================================== */

        .block-container {

            width: 313px !important;

            max-width: 313px !important;

            min-width: 313px !important;

            padding: 0 !important;

            margin-left: auto !important;

            margin-right: auto !important;

            margin-top: 20px !important;

            margin-bottom: 20px !important;

            background: #347FBD !important;

            border: 1px solid #D7D7D7 !important;

            border-radius: 10px !important;

            overflow: hidden !important;

            box-sizing: border-box !important;

        }


        /* ====================================================
           HEADER PURPLE
           ==================================================== */

        .cfp-login-header {

            width: 100%;

            height: 49px;

            background: #7528A8;

            color: #FFFFFF;

            text-align: center;

            font-family:
                Georgia,
                "Times New Roman",
                serif;

            font-size: 9px;

            font-weight: bold;

            line-height: 11px;

            padding-top: 5px;

            padding-left: 4px;

            padding-right: 4px;

            box-sizing: border-box;

            border-bottom: 1px solid #FFFFFF;

        }


        /* ====================================================
           LOGO AREA
           ==================================================== */

        .cfp-logo-area {

            width: 100%;

            height: 69px;

            display: flex;

            justify-content: center;

            align-items: center;

            margin: 0;

            padding: 0;

        }


        .cfp-logo-area img {

            width: 63px !important;

            height: 63px !important;

            object-fit: contain;

        }


        /* ====================================================
           FAVOR LOGIN
           ==================================================== */

        .cfp-favor-login {

            width: 100%;

            text-align: center;

            color: #FFFFFF;

            font-family:
                Georgia,
                "Times New Roman",
                serif;

            font-size: 9px;

            font-weight: bold;

            margin-top: 0;

            margin-bottom: 8px;

        }


        /* ====================================================
           FORM
           ==================================================== */

        div[data-testid="stForm"] {

            border: none !important;

            padding: 0 !important;

            margin: 0 !important;

            background: transparent !important;

        }


        /* ====================================================
           TEXT INPUT LABEL
           ==================================================== */

        div[data-testid="stTextInput"] > label {

            display: none !important;

        }


        /* ====================================================
           INPUT
           ==================================================== */

        div[data-testid="stTextInput"] {

            width: 194px !important;

            margin: 0 !important;

            padding: 0 !important;

        }


        div[data-testid="stTextInput"] > div {

            width: 194px !important;

            height: 25px !important;

            min-height: 25px !important;

        }


        div[data-testid="stTextInput"] input {

            width: 194px !important;

            height: 25px !important;

            min-height: 25px !important;

            box-sizing: border-box !important;

            background: #FFFFFF !important;

            color: #222222 !important;

            border: 1px solid #D4D4D4 !important;

            border-radius: 14px !important;

            font-family:
                Georgia,
                "Times New Roman",
                serif !important;

            font-size: 8px !important;

            padding-left: 12px !important;

            padding-right: 8px !important;

            outline: none !important;

        }


        div[data-testid="stTextInput"] input:focus {

            border: 1px solid #7528A8 !important;

            box-shadow:
                0 0 2px
                rgba(117,40,168,0.5) !important;

        }


        /* ====================================================
           INPUT PLACEHOLDER
           ==================================================== */

        div[data-testid="stTextInput"] input::placeholder {

            color: #777777 !important;

            opacity: 1 !important;

        }


        /* ====================================================
           LOGIN BUTTON
           ==================================================== */

        div[data-testid="stFormSubmitButton"] {

            width: 100% !important;

            display: flex !important;

            justify-content: center !important;

            margin-top: 7px !important;

        }


        div[data-testid="stFormSubmitButton"] button {

            width: 80px !important;

            min-width: 80px !important;

            max-width: 80px !important;

            height: 23px !important;

            min-height: 23px !important;

            max-height: 23px !important;

            padding: 0 !important;

            margin: 0 auto !important;

            background: #7528A8 !important;

            color: #FFFFFF !important;

            border: none !important;

            border-radius: 6px !important;

            font-family:
                Georgia,
                "Times New Roman",
                serif !important;

            font-size: 8px !important;

            font-weight: bold !important;

        }


        div[data-testid="stFormSubmitButton"] button:hover {

            background: #5E1D88 !important;

            color: #FFFFFF !important;

        }


        /* ====================================================
           ERROR / SUCCESS
           ==================================================== */

        div[data-testid="stAlert"] {

            width: 270px !important;

            margin-left: auto !important;

            margin-right: auto !important;

            margin-top: 6px !important;

            font-size: 8px !important;

            padding: 5px !important;

        }


        /* ====================================================
           FOOTER
           ==================================================== */

        .cfp-login-footer {

            width: 100%;

            text-align: center;

            color: #FFFFFF;

            font-family: Arial, sans-serif;

            font-size: 6px;

            line-height: 9px;

            padding-top: 9px;

            padding-bottom: 5px;

            opacity: 0.9;

        }


        /* ====================================================
           COLUMNS
           ==================================================== */

        div[data-testid="column"] {

            padding-left: 0 !important;

            padding-right: 0 !important;

        }


        /* ====================================================
           VERTICAL SPACING
           ==================================================== */

        div[data-testid="stVerticalBlock"] {

            gap: 0 !important;

        }


        </style>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # HEADER
    # ========================================================

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


    # ========================================================
    # LOGO
    # ========================================================

    logo_path = BASE_DIR / "logo_cfp.png"

    if not logo_path.exists():

        logo_path = BASE_DIR / "logo cfp.png"


    if logo_path.exists():

        st.markdown(
            '<div class="cfp-logo-area">',
            unsafe_allow_html=True,
        )

        st.image(
            str(logo_path),
            width=63,
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="cfp-logo-area">

                <span style="
                    font-size:50px;
                ">
                    🏛️
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # FAVOR LOGIN
    # ========================================================

    st.markdown(
        """
        <div class="cfp-favor-login">
            FAVOR LOGIN:
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # LOGIN FORM
    # ========================================================

    with st.form(
        "login_form",
        clear_on_submit=False,
    ):

        # ----------------------------------------------------
        # USERNAME
        # ----------------------------------------------------

        col_label_1, col_input_1 = st.columns(
            [1.0, 2.75],
            gap="small",
        )

        with col_label_1:

            st.markdown(
                """
                <div style="
                    color:white;
                    font-family:Georgia,serif;
                    font-size:9px;
                    font-weight:bold;
                    padding-left:20px;
                    padding-top:5px;
                ">
                    Username:
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_input_1:

            username = st.text_input(
                "Username",
                placeholder="Hatama ita boot nia username",
                label_visibility="collapsed",
            )


        # ----------------------------------------------------
        # PASSWORD
        # ----------------------------------------------------

        col_label_2, col_input_2 = st.columns(
            [1.0, 2.75],
            gap="small",
        )

        with col_label_2:

            st.markdown(
                """
                <div style="
                    color:white;
                    font-family:Georgia,serif;
                    font-size:9px;
                    font-weight:bold;
                    padding-left:20px;
                    padding-top:5px;
                ">
                    Password:
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_input_2:

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Hatama ita boot nia password",
                label_visibility="collapsed",
            )


        # ----------------------------------------------------
        # LOGIN BUTTON
        # ----------------------------------------------------

        st.markdown(
            "<div style='height:5px;'></div>",
            unsafe_allow_html=True,
        )

        submit_login = st.form_submit_button(
            "LOGIN",
        )


    # ========================================================
    # LOGIN VALIDATION
    # ========================================================

    if submit_login:

        try:

            secret_username = st.secrets["username"]

            secret_password = st.secrets["password"]

            if (
                username.strip() == secret_username
                and password == secret_password
            ):

                st.session_state["authenticated"] = True

                st.rerun()

            else:

                st.error(
                    "⚠️ Username ka Password sala! "
                    "Favor koko fali."
                )

        except Exception:

            st.error(
                "⚠️ Konfigurasaun Secrets seidauk iha "
                "Streamlit Cloud ka lokál."
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="cfp-login-footer">

            © 2026 Comissão da Função Pública - RDTL.<br>

            All rights reserved.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # STOP APPLICATION
    # ========================================================

    st.stop()


# ============================================================
# APLIKASAUN PRINSIPAL
# PÓS-LOGIN
# ============================================================

st.sidebar.markdown(
    "### 🏛️ CFP-RDTL Portal"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 👤 Kargu Asesu"
)


# ============================================================
# LOGOUT
# ============================================================

if st.sidebar.button(
    "🚪 Logout / Sai",
    use_container_width=True,
):

    st.session_state["authenticated"] = False

    st.rerun()


# ============================================================
# HEADER APLIKASAUN
# ============================================================

st.markdown(
    """
    <div style="
        background:#1E3A8A;
        padding:18px;
        border-radius:10px;
        margin-bottom:15px;
    ">

        <div style="
            color:white;
            text-align:center;
            font-size:24px;
            font-weight:bold;
        ">

            📊 Sistema Klasifikasaun
            Dezempenu Funsionáriu CFP

        </div>

        <div style="
            color:#DBEAFE;
            text-align:center;
            font-size:13px;
            margin-top:5px;
        ">

            Aplikasaun Intelijénsia Artifisiál
            uza algoritmu Decision Tree

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

init_db()


if "extra_reports" not in st.session_state:

    st.session_state["extra_reports"] = (
        load_extra_from_db()
    )


if "edit_index" not in st.session_state:

    st.session_state["edit_index"] = None


if "selected_category" not in st.session_state:

    st.session_state["selected_category"] = None


# ============================================================
# SIDEBAR DATASET
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 📁 Gestaun Dataset"
)


uploaded_file = st.sidebar.file_uploader(
    "Upload ficheiru Excel (.xlsx)",
    type=["xlsx"],
)


# ============================================================
# IF DATASET NOT UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "👋 Favór upload ficheiru Excel (.xlsx) "
        "iha sidebar hodi hahú eksplora sistema "
        "klasifikasaun."
    )

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


# ============================================================
# PROCESS DATASET
# ============================================================

try:

    df_raw = load_data(
        uploaded_file
    )


    # ========================================================
    # RENAME COLUMNS
    # ========================================================

    rename_map = {

        "Column1":
            "controlo_ativo_identificacao",

        "Column2":
            "nome_pessoal",

        "Column3":
            "id_sigap",

        "Column4":
            "id_grp",

        "Column5":
            "sexo",

        "Column6":
            "data_de_nascimento",

        "Column7":
            "instituicao",

        "Column8":
            "local_trabalho",

        "Column9":
            "funcao",

        "Column10":
            "cargo",

        "Column11":
            "data_fim_nao_exercicio",

        "Column12":
            "temp1",

        "Column13":
            "Asiduidade",

        "Column14":
            "Pontualidade",

        "Column15":
            "Produtividade",

        "Column16":
            "Kualidade_Servisu",

        "Column17":
            "Kooperasaun",

        "Column18":
            "Inisiativa",

        "Column19":
            "Disiplina",

        "Column20":
            "Responsabilidade",

        "Column21":
            "Media",

        "Column22":
            "Rezultadu_Avaliasaun",

        "Column23":
            "temp2",
    }


    df_raw.rename(
        columns={
            k: v
            for k, v in rename_map.items()
            if k in df_raw.columns
        },
        inplace=True,
    )


    # ========================================================
    # ATRIBUTOS
    # ========================================================

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


    target_col = (
        "Rezultadu_Avaliasaun"
    )


    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    missing_cols = [

        col

        for col in nota_cols + [target_col]

        if col not in df_raw.columns

    ]


    if len(missing_cols) > 0:

        st.error(
            "⚠️ Falta koluna: "
            + ", ".join(missing_cols)
        )

        st.stop()


    # ========================================================
    # DATA CLEANING
    # ========================================================

    df_base = df_raw.dropna(
        subset=nota_cols + [target_col]
    ).copy()


    for col in nota_cols:

        df_base[col] = pd.to_numeric(
            df_base[col],
            errors="coerce",
        )


    df_base = df_base.dropna(
        subset=nota_cols
    )


    # ========================================================
    # LOAD DATABASE RECORDS
    # ========================================================

    st.session_state["extra_reports"] = (
        load_extra_from_db()
    )


    extra_records = (
        st.session_state["extra_reports"]
    )


    # ========================================================
    # COMBINE DATA
    # ========================================================

    if len(extra_records) > 0:

        df_extra = pd.DataFrame(
            extra_records
        )

        df = pd.concat(
            [
                df_base,
                df_extra,
            ],
            ignore_index=True,
        )

        if "id_sigap" in df.columns:

            df = df.drop_duplicates(
                subset=["id_sigap"],
                keep="last",
            )

    else:

        df = df_base


    # ========================================================
    # SIDEBAR FILTER
    # ========================================================

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "### 🔍 Filtru Globál Dadus"
    )


    if "cargo" in df.columns:

        cargo_values = (

            df["cargo"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()

        )

        cargo_list = (
            ["Tomak (Hotu-hotu)"]
            + sorted(cargo_values)
        )

    else:

        cargo_list = [
            "Tomak (Hotu-hotu)"
        ]


    selected_cargo = st.sidebar.selectbox(
        "Filtru Kargo:",
        cargo_list,
    )


    df_filtered = df.copy()


    if (
        selected_cargo
        != "Tomak (Hotu-hotu)"
    ):

        df_filtered = df_filtered[
            df_filtered["cargo"]
            == selected_cargo
        ]


    # ========================================================
    # DOWNLOAD BACKUP
    # ========================================================

    st.sidebar.markdown("---")

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
    # TRAIN MODEL
    # ========================================================

    model, le, X_train, X_test, y_train, y_test = (
        treinar_modelo(
            df,
            nota_cols,
            target_col,
        )
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    df_filtered = df_filtered.copy()


    pred_encoded = model.predict(
        df_filtered[nota_cols]
    )


    df_filtered["Prediksaun"] = (
        le.inverse_transform(
            pred_encoded
        )
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    y_pred_test = model.predict(
        X_test
    )


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
    # DASHBOARD
    # ========================================================

    with tab1:

        st.markdown(
            "### 📈 Sumáriu Dezempenu Funsionáriu"
        )


        total_funs = len(
            df_filtered
        )


        counts_real = (
            df_filtered[target_col]
            .value_counts()
        )


        mb_count = counts_real.get(
            "Muito Bom",
            0,
        )

        b_count = counts_real.get(
            "Bom",
            0,
        )

        s_count = counts_real.get(
            "Suficiente",
            0,
        )

        i_count = counts_real.get(
            "Insuficiente",
            0,
        )


        mb_pct = (
            mb_count / total_funs * 100
            if total_funs > 0
            else 0
        )

        b_pct = (
            b_count / total_funs * 100
            if total_funs > 0
            else 0
        )

        s_pct = (
            s_count / total_funs * 100
            if total_funs > 0
            else 0
        )

        i_pct = (
            i_count / total_funs * 100
            if total_funs > 0
            else 0
        )


        # ====================================================
        # METRIC BUTTONS
        # ====================================================

        col_m1, col_m2, col_m3, col_m4, col_m5 = (
            st.columns(5)
        )


        with col_m1:

            if st.button(
                f"📊 Total Funsionáriu\n\n"
                f"{total_funs}",
                key="btn_m1",
                use_container_width=True,
            ):

                if (
                    st.session_state[
                        "selected_category"
                    ]
                    != "Tomak"
                ):

                    st.session_state[
                        "selected_category"
                    ] = "Tomak"

                else:

                    st.session_state[
                        "selected_category"
                    ] = None


        with col_m2:

            if st.button(
                f"⭐ Muito Bom\n\n"
                f"{mb_count}\n"
                f"({mb_pct:.1f}%)",
                key="btn_m2",
                use_container_width=True,
            ):

                if (
                    st.session_state[
                        "selected_category"
                    ]
                    != "Muito Bom"
                ):

                    st.session_state[
                        "selected_category"
                    ] = "Muito Bom"

                else:

                    st.session_state[
                        "selected_category"
                    ] = None


        with col_m3:

            if st.button(
                f"✨ Bom\n\n"
                f"{b_count}\n"
                f"({b_pct:.1f}%)",
                key="btn_m3",
                use_container_width=True,
            ):

                if (
                    st.session_state[
                        "selected_category"
                    ]
                    != "Bom"
                ):

                    st.session_state[
                        "selected_category"
                    ] = "Bom"

                else:

                    st.session_state[
                        "selected_category"
                    ] = None


        with col_m4:

            if st.button(
                f"📌 Suficiente\n\n"
                f"{s_count}\n"
                f"({s_pct:.1f}%)",
                key="btn_m4",
                use_container_width=True,
            ):

                if (
                    st.session_state[
                        "selected_category"
                    ]
                    != "Suficiente"
                ):

                    st.session_state[
                        "selected_category"
                    ] = "Suficiente"

                else:

                    st.session_state[
                        "selected_category"
                    ] = None


        with col_m5:

            if st.button(
                f"⚠️ Insuficiente\n\n"
                f"{i_count}\n"
                f"({i_pct:.1f}%)",
                key="btn_m5",
                use_container_width=True,
            ):

                if (
                    st.session_state[
                        "selected_category"
                    ]
                    != "Insuficiente"
                ):

                    st.session_state[
                        "selected_category"
                    ] = "Insuficiente"

                else:

                    st.session_state[
                        "selected_category"
                    ] = None


        # ====================================================
        # SELECTED TABLE
        # ====================================================

        selected_cat = (
            st.session_state[
                "selected_category"
            ]
        )


        if selected_cat is not None:

            st.markdown("---")


            if selected_cat == "Tomak":

                df_table = (
                    df_filtered
                )

                st.markdown(
                    f"### 📋 Lista Funsionáriu "
                    f"Tomak ({len(df_table)})"
                )

            else:

                df_table = (
                    df_filtered[
                        df_filtered[
                            target_col
                        ]
                        == selected_cat
                    ]
                )

                st.markdown(
                    f"### 📋 Lista Funsionáriu "
                    f"ba Kategoria: "
                    f"`{selected_cat}` "
                    f"({len(df_table)})"
                )


            display_cols = [

                "controlo_ativo_identificacao",

                "nome_pessoal",

                "id_sigap",

                "id_grp",

                "sexo",

                "local_trabalho",

                "cargo",

                target_col,

            ]


            display_cols = [

                col

                for col in display_cols

                if col in df_table.columns

            ]


            st.dataframe(
                df_table[
                    display_cols
                ],
                use_container_width=True,
            )


            # =================================================
            # DOWNLOAD CSV / PDF
            # =================================================

            dl_col1, dl_col2 = (
                st.columns(2)
            )


            with dl_col1:

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
                    key="dl_filtered_csv",
                )


            with dl_col2:

                pdf_buffer = (
                    generate_pdf_report(
                        df_table,
                        (
                            "Relatóriu Dezempenu "
                            "Funsionáriu - "
                            f"{selected_cat}"
                        ),
                    )
                )


                st.download_button(
                    label="📄 Download Relatóriu PDF Ofisiál",
                    data=pdf_buffer,
                    file_name=(
                        f"relatorio_cfp_"
                        f"{selected_cat}.pdf"
                    ),
                    mime="application/pdf",
                    key="dl_filtered_pdf",
                )


            if st.button(
                "❌ Subar Tabela",
                key="hide_table_btn",
            ):

                st.session_state[
                    "selected_category"
                ] = None

                st.rerun()


        # ====================================================
        # CHARTS
        # ====================================================

        st.markdown("---")


        col_g1, col_g2 = st.columns(2)


        # ====================================================
        # BAR CHART
        # ====================================================

        with col_g1:

            st.markdown(
                "##### 📊 Komparasaun Kategoria "
                "(Reál vs Prediksaun)"
            )


            categories = [

                "Muito Bom",

                "Bom",

                "Suficiente",

                "Insuficiente",

            ]


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
                .get(cat, 0)

                for cat in categories

            ]


            x = np.arange(
                len(categories)
            )


            width = 0.35


            fig, ax = plt.subplots(
                figsize=(6, 4)
            )


            rects1 = ax.bar(
                x - width / 2,
                real_counts,
                width,
                label="Dadus Reál",
                color="#1E3A8A",
                alpha=0.9,
            )


            rects2 = ax.bar(
                x + width / 2,
                pred_counts,
                width,
                label="Prediksaun Tree",
                color="#D97706",
                alpha=0.9,
            )


            ax.bar_label(
                rects1,
                padding=3,
                fontsize=8,
            )


            ax.bar_label(
                rects2,
                padding=3,
                fontsize=8,
            )


            ax.set_ylabel(
                "Total Funsionáriu"
            )


            ax.set_xticks(x)

            ax.set_xticklabels(
                categories
            )


            ax.legend()

            sns.despine()

            st.pyplot(
                fig,
                clear_figure=True,
            )


        # ====================================================
        # DONUT CHART
        # ====================================================

        with col_g2:

            st.markdown(
                "##### 🍩 Proporsaun "
                "Kategoria Dezempenu"
            )


            sizes = [

                counts_real.get(
                    cat,
                    0,
                )

                for cat in categories

            ]


            colors_list = [

                "#1E3A8A",

                "#3B82F6",

                "#D97706",

                "#EF4444",

            ]


            fig2, ax2 = plt.subplots(
                figsize=(6, 4)
            )


            if sum(sizes) > 0:

                ax2.pie(
                    sizes,
                    labels=categories,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors_list,
                    wedgeprops=dict(
                        width=0.4,
                        edgecolor="white",
                        linewidth=2,
                    ),
                )


            st.pyplot(
                fig2,
                clear_figure=True,
            )


        # ====================================================
        # LOCATION CHART
        # ====================================================

        st.markdown("---")

        st.markdown(
            "##### 🗺️ Gráfiku Avansadu: "
            "Desentralizasaun Dezempenu "
            "tuir Local de Trabalhu"
        )


        if "local_trabalho" in df_filtered.columns:

            fig_loc, ax_loc = plt.subplots(
                figsize=(10, 4.5)
            )


            df_loc_counts = pd.crosstab(
                df_filtered[
                    "local_trabalho"
                ],
                df_filtered[
                    target_col
                ],
            )


            existing_cats = [

                c

                for c in categories

                if c in df_loc_counts.columns

            ]


            df_loc_counts = (
                df_loc_counts.reindex(
                    columns=existing_cats,
                    fill_value=0,
                )
            )


            df_loc_counts.plot(
                kind="bar",
                stacked=True,
                ax=ax_loc,
                colormap="crest",
                edgecolor="none",
            )


            ax_loc.set_title(
                "Distribuisaun Avaliasaun "
                "Dezempenu tuir Munisípiu",
                fontsize=11,
                fontweight="bold",
            )


            ax_loc.set_xlabel(
                "Local de Trabalhu",
                fontsize=9,
            )


            ax_loc.set_ylabel(
                "Total Funsionáriu",
                fontsize=9,
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
                clear_figure=True,
            )


        else:

            st.info(
                "Koluna "
                "'local_trabalho' "
                "la dispoñível."
            )


    # ========================================================
    # TAB 2
    # MODEL & PERFORMANCE
    # ========================================================

    with tab2:

        st.subheader(
            "📋 Amostra Dadus (Preview)"
        )


        st.dataframe(
            df_filtered.head(10),
            use_container_width=True,
        )


        st.markdown("---")


        st.subheader(
            "🚀 Performance Modelu "
            "Decision Tree"
        )


        st.success(
            f"✅ Akurasi Modelu "
            f"(Accuracy): "
            f"**{acc * 100:.2f}%**"
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        col_eval1, col_eval2 = (
            st.columns(2)
        )


        with col_eval1:

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


            ax_cm.set_xlabel(
                "Prediksaun"
            )

            ax_cm.set_ylabel(
                "Dadus Reál"
            )


            st.pyplot(
                fig_cm,
                clear_figure=True,
            )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        with col_eval2:

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


            report_dict = (
                classification_report(
                    y_test,
                    y_pred_test,
                    labels=unique_labels,
                    target_names=present_class_names,
                    output_dict=True,
                    zero_division=0,
                )
            )


            df_report = (
                pd.DataFrame(
                    report_dict
                )
                .transpose()
            )


            st.dataframe(
                df_report,
                use_container_width=True,
            )


        # ====================================================
        # DECISION TREE
        # ====================================================

        st.markdown("---")


        st.subheader(
            "🌳 Vizualizasaun "
            "Árbore Desizaun"
        )


        max_depth_vis = st.slider(
            "Hili Profundidade "
            "Árvore (Max Depth)",
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
            clear_figure=True,
        )


    # ========================================================
    # TAB 3
    # PREDIKSAUN & GESTAUN DADUS
    # ========================================================

    with tab3:

        st.subheader(
            "🔍 Prediksaun Funsionáriu "
            "Foun & Gestaun Dadus"
        )


        # ====================================================
        # LOAD DATABASE
        # ====================================================

        extra_records = (
            load_extra_from_db()
        )


        st.session_state[
            "extra_reports"
        ] = extra_records


        # ====================================================
        # LIST DATABASE
        # ====================================================

        st.markdown(
            "##### 📋 Lista Dadus "
            "Funsionáriu Foun & Asaun Gestaun"
        )


        if len(extra_records) > 0:

            h1, h2, h3, h4 = st.columns(
                [3, 2, 2.5, 2.5]
            )


            with h1:

                st.markdown(
                    "**Naran / ID SIGAP**"
                )


            with h2:

                st.markdown(
                    "**Munisípiu**"
                )


            with h3:

                st.markdown(
                    "**Kargo**"
                )


            with h4:

                st.markdown(
                    "**Rezultadu Avaliasaun & Asaun**"
                )


            st.markdown(
                "<hr>",
                unsafe_allow_html=True,
            )


            for i, rec in enumerate(
                extra_records
            ):

                c1, c2, c3, c4 = (
                    st.columns(
                        [3, 2, 2.5, 2.5]
                    )
                )


                with c1:

                    st.write(
                        f"👤 **"
                        f"{rec.get('nome_pessoal', '')}"
                        f"**"
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
                        f"📍 "
                        f"{rec.get('local_trabalho', '')}"
                    )


                with c3:

                    st.write(
                        f"💼 "
                        f"{rec.get('cargo', '')}"
                    )


                with c4:

                    res_val = rec.get(
                        "Rezultadu_Avaliasaun",
                        "N/A",
                    )


                    sub_res, sub_edit, sub_del = (
                        st.columns(
                            [1.5, 0.7, 0.7]
                        )
                    )


                    with sub_res:

                        st.markdown(
                            f"⭐ **{res_val}**"
                        )


                    with sub_edit:

                        if st.button(
                            "✏️",
                            key=f"edit_btn_{i}",
                        ):

                            st.session_state[
                                "edit_index"
                            ] = i

                            st.rerun()


                    with sub_del:

                        with st.popover(
                            "🗑️"
                        ):

                            st.markdown(
                                "Kerteza hakarak "
                                f"hamos **"
                                f"{rec.get('nome_pessoal', '')}"
                                f"**?"
                            )


                            if st.button(
                                "I Hamos Duni",
                                key=f"confirm_del_{i}",
                            ):

                                if (
                                    delete_extra_from_db_by_index(
                                        i
                                    )
                                ):

                                    if (
                                        st.session_state[
                                            "edit_index"
                                        ]
                                        == i
                                    ):

                                        st.session_state[
                                            "edit_index"
                                        ] = None


                                    st.success(
                                        "Hamos ona!"
                                    )

                                    st.rerun()


                st.markdown(
                    "<hr>",
                    unsafe_allow_html=True,
                )


        else:

            st.info(
                "ℹ️ Sei la iha dadus foun "
                "rejisitadu iha database lokal."
            )


        # ====================================================
        # EDIT MODE
        # ====================================================

        if (
            st.session_state[
                "edit_index"
            ]
            is not None
        ):

            st.info(
                "⚠️ Atualmente hela iha "
                "Módudu Edisaun ba Index: "
                f"**{st.session_state['edit_index']}**"
            )


            if st.button(
                "❌ Kansela / Sai husi Módudu Edisaun",
                key="cancel_edit_mode",
            ):

                st.session_state[
                    "edit_index"
                ] = None

                st.rerun()


        # ====================================================
        # FORM
        # ====================================================

        st.markdown("---")


        idx_edit = (
            st.session_state[
                "edit_index"
            ]
        )


        def_val = {}


        if (
            idx_edit is not None
            and idx_edit < len(
                st.session_state[
                    "extra_reports"
                ]
            )
        ):

            def_val = (
                st.session_state[
                    "extra_reports"
                ][idx_edit]
            )


            st.markdown(
                f"#### ✏️ Atualiza Dadus "
                f"Funsionáriu "
                f"(Index: {idx_edit})"
            )

        else:

            st.markdown(
                "#### ➕ Input Funsionáriu "
                "Foun ba Prediksaun"
            )


        # ====================================================
        # OPTIONS
        # ====================================================

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

            "Regime Geral das Carreiras, Técnico Superior Grau B, 10, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 1, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 4, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau C, 1, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Superior Grau A, 4, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau C, 5, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Superior Grau A, 1, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Administrativo Grau E, 2, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau D, 3, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau C, 3, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau D, 4, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau C, 4, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Profissional Grau C, 2, PERMANENTE",

            "Regime Geral das Carreiras, Técnico Superior Grau B, 4, PERMANENTE",

            "Regime Geral das Carreiras, Assistente Grau F, 2, PERMANENTE",

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


        # ====================================================
        # INPUT FORM
        # ====================================================

        with st.form(
            "funsionariu_form"
        ):

            st.markdown(
                "##### 📝 1. Informasaun "
                "Identidade Funsionáriu"
            )


            col_i1, col_i2, col_i3 = (
                st.columns(3)
            )


            # ------------------------------------------------
            # COLUMN 1
            # ------------------------------------------------

            with col_i1:

                ativo_opts = [
                    "Ativo",
                    "La Ativo",
                ]


                cur_ativo = def_val.get(
                    "controlo_ativo_identificacao",
                    "Ativo",
                )


                txt_ativo = st.selectbox(
                    "Controlo Ativo Identifikasaun",
                    ativo_opts,
                    index=(
                        ativo_opts.index(
                            cur_ativo
                        )
                        if cur_ativo in ativo_opts
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
                    "ID SIGAP "
                    "(Numeriku no Símbolu)*",
                    def_val.get(
                        "id_sigap",
                        "",
                    ),
                )


                cur_sexo = def_val.get(
                    "sexo",
                    "M",
                )


                txt_sexo = st.selectbox(
                    "Sexo",
                    ["M", "F"],
                    index=(
                        0
                        if cur_sexo == "M"
                        else 1
                    ),
                )


            # ------------------------------------------------
            # COLUMN 2
            # ------------------------------------------------

            with col_i2:

                txt_inst = st.text_input(
                    "Instituisaun",
                    def_val.get(
                        "instituicao",
                        "CFP",
                    ),
                )


                cur_local = def_val.get(
                    "local_trabalho",
                    "Díli",
                )


                txt_local = st.selectbox(
                    "Local Trabalhu",
                    municipios,
                    index=(
                        municipios.index(
                            cur_local
                        )
                        if cur_local in municipios
                        else 0
                    ),
                )


                try:

                    default_date = (
                        pd.to_datetime(
                            def_val.get(
                                "data_de_nascimento",
                                "1995-01-01",
                            )
                        )
                        .date()
                    )

                except Exception:

                    default_date = (
                        pd.to_datetime(
                            "1995-01-01"
                        )
                        .date()
                    )


                txt_nascimento = (
                    st.date_input(
                        "Data de Nascimento",
                        value=default_date,
                    )
                )


            # ------------------------------------------------
            # COLUMN 3
            # ------------------------------------------------

            with col_i3:

                cur_func = def_val.get(
                    "funcao",
                    funcoes[0],
                )


                txt_funcao = st.selectbox(
                    "Funsaun",
                    funcoes,
                    index=(
                        funcoes.index(
                            cur_func
                        )
                        if cur_func in funcoes
                        else 0
                    ),
                )


                cur_cargo = def_val.get(
                    "cargo",
                    cargos[0],
                )


                txt_cargo = st.selectbox(
                    "Kargo",
                    cargos,
                    index=(
                        cargos.index(
                            cur_cargo
                        )
                        if cur_cargo in cargos
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


            # =================================================
            # INDICADORES
            # =================================================

            st.markdown(
                "##### 📊 2. Indikadór "
                "Avaliasaun Funsionáriu"
            )


            col_a, col_b, col_c, col_d = (
                st.columns(4)
            )


            with col_a:

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


            with col_b:

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


            with col_c:

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


            with col_d:

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


            # =================================================
            # SUBMIT
            # =================================================

            submit_pred = (
                st.form_submit_button(
                    "💾 Salva / Prediksaun",
                    use_container_width=True,
                )
            )


            # =================================================
            # SAVE / PREDICTION
            # =================================================

            if submit_pred:

                if (
                    not txt_nome.strip()
                    or not txt_sigap.strip()
                ):

                    st.error(
                        "⚠️ Favór preenxe "
                        "Naran Pessoal no "
                        "ID SIGAP ho loos!"
                    )


                elif any(
                    c.isalpha()
                    for c in txt_sigap
                ):

                    st.error(
                        "⚠️ ID SIGAP labele uza "
                        "letra/alfabetu! "
                        "Tenke uza de'it "
                        "númeru no símbolu."
                    )


                else:

                    # ----------------------------------------
                    # PREDICTION INPUT
                    # ----------------------------------------

                    input_data = np.array(
                        [[

                            p_asid,

                            p_pont,

                            p_prod,

                            p_kual,

                            p_koop,

                            p_inis,

                            p_disp,

                            p_resp,

                        ]]
                    )


                    # ----------------------------------------
                    # MODEL PREDICTION
                    # ----------------------------------------

                    pred_encoded = (
                        model.predict(
                            input_data
                        )
                    )


                    pred_label = (
                        le.inverse_transform(
                            pred_encoded
                        )[0]
                    )


                    # ----------------------------------------
                    # NEW RECORD
                    # ----------------------------------------

                    new_record = {

                        "controlo_ativo_identificacao":
                            txt_ativo,

                        "nome_pessoal":
                            txt_nome,

                        "id_sigap":
                            txt_sigap,

                        "sexo":
                            txt_sexo,

                        "instituicao":
                            txt_inst,

                        "local_trabalho":
                            txt_local,

                        "data_de_nascimento":
                            str(
                                txt_nascimento
                            ),

                        "funcao":
                            txt_funcao,

                        "cargo":
                            txt_cargo,

                        "id_grp":
                            txt_grp,

                        "Asiduidade":
                            p_asid,

                        "Pontualidade":
                            p_pont,

                        "Produtividade":
                            p_prod,

                        "Kualidade_Servisu":
                            p_kual,

                        "Kooperasaun":
                            p_koop,

                        "Inisiativa":
                            p_inis,

                        "Disiplina":
                            p_disp,

                        "Responsabilidade":
                            p_resp,

                        "Rezultadu_Avaliasaun":
                            pred_label,

                    }


                    # ----------------------------------------
                    # UPDATE
                    # ----------------------------------------

                    if idx_edit is not None:

                        if update_extra_in_db_by_index(
                            idx_edit,
                            new_record,
                        ):

                            st.success(
                                "✅ Atualiza dadus "
                                "susesu! "
                                f"Prediksaun: "
                                f"**{pred_label}**"
                            )


                            st.session_state[
                                "edit_index"
                            ] = None


                            st.session_state[
                                "extra_reports"
                            ] = (
                                load_extra_from_db()
                            )


                            st.rerun()


                        else:

                            st.error(
                                "⚠️ Falha atu "
                                "atualiza dadus."
                            )


                    # ----------------------------------------
                    # INSERT NEW
                    # ----------------------------------------

                    else:

                        if save_extra_to_db(
                            new_record
                        ):

                            st.success(
                                "✅ Salva dadus "
                                "susesu! "
                                f"Prediksaun: "
                                f"**{pred_label}**"
                            )


                            st.session_state[
                                "extra_reports"
                            ] = (
                                load_extra_from_db()
                            )


                            st.rerun()


                        else:

                            st.error(
                                "⚠️ ID SIGAP ne'e "
                                "bele iha ona "
                                "ka iha erro ruma "
                                "iha database."
                            )


except Exception as e:

    st.error(
        "⚠️ Erro iha sistema: "
        f"{e}"
    )

    st.exception(e)

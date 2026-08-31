import matplotlib.pyplot as plt
from io import BytesIO
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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

from database import (
    delete_extra_from_db_by_index,
    init_db,
    load_extra_from_db,
    save_extra_to_db,
    update_extra_in_db_by_index,
)
from models import treinar_modelo
from ui_components import render_custom_css

# Konfigurasaun Pajina ho Icon Árbore Desizaun (🌳)
st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - RDTL",
    page_icon="🌳",
    layout="wide",
)

render_custom_css()

# Funsaun atu lee ficheiru Excel
def load_data(file):
    return pd.read_excel(file)

# ==========================================
# SISTEMA LOGIN / AUTENTIKASAUN (ESTILU CFP RDTL)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 500px !important;
            margin: 0 auto !important;
        }
        .cfp-login-card {
            background-color: #ffffff;
            padding: 35px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            border-top: 6px solid #D97706;
        }
        .cfp-header-title {
            color: #1E3A8A;
            font-weight: 800;
            font-size: 15px;
            text-align: center;
            letter-spacing: 0.5px;
            line-height: 1.5;
            margin-bottom: 5px;
        }
        .cfp-subtitle {
            text-align: center;
            color: #64748B;
            font-size: 12px;
            margin-bottom: 25px;
            font-weight: 600;
        }
        div.stButton > button {
            background-color: #1E3A8A !important;
            color: white !important;
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            padding: 10px;
            border: none;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div.stButton > button:hover {
            background-color: #1D4ED8 !important;
        }
        label {
            color: #1E293B !important;
            font-weight: bold !important;
        }
        .login-footer-text {
            text-align: center;
            color: #94A3B8;
            font-size: 11px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <div class="cfp-login-card">
                <div style="text-align: center; font-size: 65px; line-height: 1; margin-bottom: 15px;">
                    🌳
                </div>
                <div class="cfp-header-title">
                    COMISSÃO DA FUNÇÃO PÚBLICA<br>REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE
                </div>
                <div class="cfp-subtitle">
                    Portal de Gestão e Classificação de Desempenho (Decision Tree)
                </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username:", placeholder="Hatama ita-nia username")
            password = st.text_input("Password:", type="password", placeholder="Hatama ita-nia password")
            st.markdown("<br>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("ENTRADA / LOGIN")
            
            if submit_login:
                try:
                    if username == st.secrets["username"] and password == st.secrets["password"]:
                        st.session_state["authenticated"] = True
                        st.success("Login susesu! Redirecting...")
                        st.rerun()
                    else:
                        st.error("⚠️ Username ka Password sala! Favor koko fali.")
                except Exception:
                    st.error("⚠️ Konfigurasaun Secrets seidauk iha Streamlit Cloud ka lokál.")
        
        st.markdown("""
                <div class="login-footer-text">
                    © 2026 Comissão da Função Pública - RDTL. All rights reserved.
                </div>
            </div>
        """, unsafe_allow_html=True)
                    
    st.stop()

# ==========================================
# APLIKASAUN PRINSIPAL (PÓS-LOGIN)
# ==========================================
col_side_img, col_side_txt = st.sidebar.columns([1, 4])
with col_side_img:
    st.markdown("<h2 style='margin:0; padding:0;'>🌳</h2>", unsafe_allow_html=True)
with col_side_txt:
    st.markdown("### CFP-RDTL Portal")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Kargu Asesu")
if st.sidebar.button("🚪 Logout / Sai"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown('<p class="main-title">🌳 Sistema Klasifikasaun Dezempenu Funsionáriu CFP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasaun Intelijénsia Artifisiál uza algoritmu Decision Tree bazeia ba indikadór Komisaun Função Pública RDTL.</p>', unsafe_allow_html=True)

init_db()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None

if "comparison_selection" not in st.session_state:
    st.session_state["comparison_selection"] = None

if "chart_key_version" not in st.session_state:
    st.session_state["chart_key_version"] = 0

# Funsaun Generál PDF Ofisiál CFP
def generate_pdf_report(df_data, title_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=11, textColor=colors.HexColor('#1E3A8A'), spaceAfter=2, alignment=1, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'SubTitleStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#4B5563'), spaceAfter=15, alignment=1, fontName='Helvetica'
    )

    elements.append(Paragraph("REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE", title_style))
    elements.append(Paragraph("COMISSÃO DA FUNÇÃO PÚBLICA (CFP)", title_style))
    elements.append(Paragraph(title_report, subtitle_style))
    elements.append(Spacer(1, 5))

    table_data = [["Naran Pessoal", "ID SIGAP", "Munisípiu", "Kargo", "Avaliasaun"]]
    for _, row in df_data.iterrows():
        table_data.append([
            str(row.get("nome_pessoal", "")),
            str(row.get("id_sigap", "")),
            str(row.get("local_trabalho", "")),
            str(row.get("cargo", "")),
            str(row.get("Rezultadu_Avaliasaun", ""))
        ])

    t = Table(table_data, colWidths=[150, 70, 90, 110, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Gestaun Dataset")
uploaded_file = st.sidebar.file_uploader("Upload ficheiru Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_raw = load_data(uploaded_file)
        rename_map = {
            "Column1": "controlo_ativo_identificacao", "Column2": "nome_pessoal", "Column3": "id_sigap",
            "Column4": "id_grp", "Column5": "sexo", "Column6": "data_de_nascimento", "Column7": "instituicao",
            "Column8": "local_trabalho", "Column9": "funcao", "Column10": "cargo", "Column11": "data_fim_nao_exercicio",
            "Column12": "temp1", "Column13": "Asiduidade", "Column14": "Pontualidade", "Column15": "Produtividade",
            "Column16": "Kualidade_Servisu", "Column17": "Kooperasaun", "Column18": "Inisiativa",
            "Column19": "Disiplina", "Column20": "Responsabilidade", "Column21": "Media",
            "Column22": "Rezultadu_Avaliasaun", "Column23": "temp2",
        }
        df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)

        nota_cols = [
            "Asiduidade", "Pontualidade", "Produtividade", "Kualidade_Servisu",
            "Kooperasaun", "Inisiativa", "Disiplina", "Responsabilidade"
        ]
        target_col = "Rezultadu_Avaliasaun"

        missing_cols = [col for col in nota_cols + [target_col] if col not in df_raw.columns]

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
            st.sidebar.markdown("### 🔍 Filtru Globál Dadus")
            cargo_list = ["Tomak (Hotu-hotu)"] + sorted(list(df["cargo"].dropna().unique()))
            selected_cargo = st.sidebar.selectbox("Filtru Kargo:", cargo_list)

            df_filtered = df.copy()
            if selected_cargo != "Tomak (Hotu-hotu)":
                df_filtered = df_filtered[df_filtered["cargo"] == selected_cargo]

            st.sidebar.markdown("---")
            csv_full = df_filtered.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                label="⬇️ Download Backup (CSV)", data=csv_full, file_name="dataset_cfp_filtrado.csv", mime="text/csv"
            )

            model, le, X_train, X_test, y_train, y_test = treinar_modelo(df, nota_cols, target_col)
            df_filtered["Prediksaun"] = le.inverse_transform(model.predict(df_filtered[nota_cols]))
            acc = accuracy_score(y_test, model.predict(X_test))

            tab1, tab2, tab3 = st.tabs([
                "📊 Dashboard Analítiku", "⚙️ Modelu & Performance", "🔮 Prediksaun & Gestaun Dadus"
            ])

            with tab1:
                st.markdown("### 📈 Sumáriu Dezempenu Funsionáriu")
                total_funs = len(df_filtered)
                
                # Dadus Reál
                counts_real = df_filtered[target_col].value_counts()
                mb_real_pct = (counts_real.get("Muito Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                b_real_pct = (counts_real.get("Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                s_real_pct = (counts_real.get("Suficiente", 0) / total_funs) * 100 if total_funs > 0 else 0
                i_real_pct = (counts_real.get("Insuficiente", 0) / total_funs) * 100 if total_funs > 0 else 0

                # Dadus Prediksaun
                counts_pred = df_filtered["Prediksaun"].value_counts()
                mb_pred_pct = (counts_pred.get("Muito Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                b_pred_pct = (counts_pred.get("Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                s_pred_pct = (counts_pred.get("Suficiente", 0) / total_funs) * 100 if total_funs > 0 else 0
                i_pred_pct = (counts_pred.get("Insuficiente", 0) / total_funs) * 100 if total_funs > 0 else 0

                # KPI Cards ho Persentajen Reál vs Prediksaun
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                with col_m1:
                    if st.button(f"📊 Total Funsionáriu\n\n{total_funs}", key="btn_m1"):
                        st.session_state["selected_category"] = "Tomak" if st.session_state["selected_category"] != "Tomak" else None
                with col_m2:
                    if st.button(f"⭐ Muito Bom\n\nReál: {mb_real_pct:.1f}%\nPred: {mb_pred_pct:.1f}%", key="btn_m2"):
                        st.session_state["selected_category"] = "Muito Bom" if st.session_state["selected_category"] != "Muito Bom" else None
                with col_m3:
                    if st.button(f"✨ Bom\n\nReál: {b_real_pct:.1f}%\nPred: {b_pred_pct:.1f}%", key="btn_m3"):
                        st.session_state["selected_category"] = "Bom" if st.session_state["selected_category"] != "Bom" else None
                with col_m4:
                    if st.button(f"📌 Suficiente\n\nReál: {s_real_pct:.1f}%\nPred: {s_pred_pct:.1f}%", key="btn_m4"):
                        st.session_state["selected_category"] = "Suficiente" if st.session_state["selected_category"] != "Suficiente" else None
                with col_m5:
                    if st.button(f"⚠️ Insuficiente\n\nReál: {i_real_pct:.1f}%\nPred: {i_pred_pct:.1f}%", key="btn_m5"):
                        st.session_state["selected_category"] = "Insuficiente" if st.session_state["selected_category"] != "Insuficiente" else None

                selected_cat = st.session_state["selected_category"]
                if selected_cat is not None:
                    st.markdown("---")
                    if selected_cat == "Tomak":
                        df_table = df_filtered
                        st.markdown(f"### 📋 Lista Funsionáriu Tomak ({len(df_table)})")
                    else:
                        df_table = df_filtered[df_filtered[target_col] == selected_cat]
                        st.markdown(f"### 📋 Lista Funsionáriu ba Kategoria: `{selected_cat}` ({len(df_table)})")

                    st.dataframe(df_table[["controlo_ativo_identificacao", "nome_pessoal", "id_sigap", "id_grp", "sexo", "local_trabalho", "cargo", target_col]], use_container_width=True)
                    
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        csv_filtered = df_table.to_csv(index=False).encode("utf-8")
                        st.download_button(label="📥 Download CSV", data=csv_filtered, file_name=f"relatorio_cfp_{selected_cat}.csv", mime="text/csv", key="dl_filtered_csv")
                    with dl_col2:
                        pdf_buffer = generate_pdf_report(df_table, f"Relatóriu Dezempenu Funsionáriu - {selected_cat}")
                        st.download_button(label="📄 Download Relatóriu PDF Ofisiál", data=pdf_buffer, file_name=f"relatorio_cfp_{selected_cat}.pdf", mime="application/pdf", key="dl_filtered_pdf")

                    if st.button("❌ Subar Tabela", key="hide_table_btn"):
                        st.session_state["selected_category"] = None
                        st.rerun()

                st.markdown("---")
                col_g1, col_g2 = st.columns(2)
                
                # GRÁFIKU 1: BARRA (REÁL VS PREDIKSAUN)
                with col_g1:
                    st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")

                    if go is None:
                        st.error("Biblioteca Plotly seidauk instala. Halo: pip install plotly")
                    else:
                        categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                        real_counts = [counts_real.get(cat, 0) for cat in categories]
                        pred_counts = [counts_pred.get(cat, 0) for cat in categories]

                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Bar(
                            name="Dadus Reál", x=categories, y=real_counts,
                            marker_color="#1E3A8A",
                            customdata=[[cat, "real"] for cat in categories],
                            hovertemplate="<b>%{x}</b><br>Dadus Reál<br>Total: %{y}<br><extra>Klik iha barra</extra>"
                        ))
                        fig_compare.add_trace(go.Bar(
                            name="Prediksaun Tree", x=categories, y=pred_counts,
                            marker_color="#3B82F6",
                            customdata=[[cat, "prediksaun"] for cat in categories],
                            hovertemplate="<b>%{x}</b><br>Prediksaun Decision Tree<br>Total: %{y}<br><extra>Klik iha barra</extra>"
                        ))
                        fig_compare.update_layout(
                            barmode="group", height=450,
                            xaxis_title="Kategoria Dezempenu",
                            yaxis_title="Total Funsionáriu",
                            legend_title="Tipu Dadus",
                            clickmode="event+select",
                            margin=dict(l=20, r=20, t=30, b=20),
                        )

                        chart_key_bar = f"grafiku_real_vs_pred_{st.session_state['chart_key_version']}"
                        event_bar = st.plotly_chart(
                            fig_compare,
                            use_container_width=True,
                            on_select="rerun",
                            selection_mode="points",
                            key=chart_key_bar,
                        )

                        if event_bar is not None:
                            try:
                                points_bar = event_bar.selection.points
                            except Exception:
                                points_bar = []

                            if points_bar:
                                point = points_bar[0]
                                custom = point.get("customdata") if isinstance(point, dict) else None
                                if custom and len(custom) >= 2:
                                    st.session_state["comparison_selection"] = {
                                        "category": str(custom[0]),
                                        "type": str(custom[1]),
                                    }

                # GRÁFIKU 2: DONUT CHART COMPARAÇÃO (REÁL VS PREDIKSAUN)
                with col_g2:
                    st.markdown("##### 🍩 Proporsaun Persentajen (Reál vs Prediksaun)")
                    
                    if go is None or make_subplots is None:
                        st.error("Biblioteca Plotly seidauk instala. Halo: pip install plotly")
                    else:
                        categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                        colors_map = {
                            "Muito Bom": "#1E3A8A", 
                            "Bom": "#3B82F6", 
                            "Suficiente": "#D97706", 
                            "Insuficiente": "#EF4444"
                        }
                        
                        sizes_real = [counts_real.get(cat, 0) for cat in categories]
                        sizes_pred = [counts_pred.get(cat, 0) for cat in categories]

                        fig_donut = make_subplots(
                            rows=1, cols=2,
                            specs=[[{"type": "domain"}, {"type": "domain"}]],
                            subplot_titles=["<b>Dadus Reál</b>", "<b>Prediksaun</b>"]
                        )
                        
                        fig_donut.add_trace(go.Pie(
                            labels=categories,
                            values=sizes_real,
                            hole=0.4,
                            name="Dadus Reál",
                            marker_colors=[colors_map[cat] for cat in categories],
                            customdata=[[cat, "real"] for cat in categories],
                            hovertemplate="<b>Reál: %{label}</b><br>Total: %{value}<br>Persentajen: %{percent}<extra></extra>"
                        ), 1, 1)

                        fig_donut.add_trace(go.Pie(
                            labels=categories,
                            values=sizes_pred,
                            hole=0.4,
                            name="Prediksaun",
                            marker_colors=[colors_map[cat] for cat in categories],
                            customdata=[[cat, "prediksaun"] for cat in categories],
                            hovertemplate="<b>Prediksaun: %{label}</b><br>Total: %{value}<br>Persentajen: %{percent}<extra></extra>"
                        ), 1, 2)
                        
                        fig_donut.update_layout(
                            height=450,
                            margin=dict(l=10, r=10, t=40, b=10),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                            clickmode="event+select"
                        )

                        chart_key_donut = f"grafiku_donut_{st.session_state['chart_key_version']}"
                        
                        event_donut = st.plotly_chart(
                            fig_donut,
                            use_container_width=True,
                            on_select="rerun",
                            selection_mode="points",
                            key=chart_key_donut
                        )

                        if event_donut is not None:
                            try:
                                points_donut = event_donut.selection.points
                            except Exception:
                                points_donut = []

                            if points_donut:
                                point_d = points_donut[0]
                                custom_d = point_d.get("customdata") if isinstance(point_d, dict) else None
                                if custom_d and len(custom_d) >= 2:
                                    st.session_state["comparison_selection"] = {
                                        "category": str(custom_d[0]),
                                        "type": str(custom_d[1])
                                    }

                # SELEKSAUN TABELA BA GRÁFIKU RUA HOTU
                selection = st.session_state.get("comparison_selection")

                if selection:
                    selected_category = selection["category"]
                    selected_type = selection["type"]

                    st.markdown("---")

                    if selected_type == "real":
                        df_selected = df_filtered[
                            df_filtered[target_col].astype(str).str.strip() == selected_category
                        ].copy()

                        st.markdown(f"### 📋 Dadus Reál – Kategoria: `{selected_category}`")
                        st.info(f"Total funsionáriu: **{len(df_selected)}**")

                        columns_show = [
                            "controlo_ativo_identificacao", "nome_pessoal", "id_sigap",
                            "id_grp", "sexo", "local_trabalho", "cargo",
                            "Asiduidade", "Pontualidade", "Produtividade",
                            "Kualidade_Servisu", "Kooperasaun", "Inisiativa",
                            "Disiplina", "Responsabilidade", target_col
                        ]

                        columns_show = [c for c in columns_show if c in df_selected.columns]
                        st.dataframe(df_selected[columns_show], use_container_width=True, hide_index=True)

                        csv_data = df_selected.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Dadus Reál (CSV)", csv_data,
                            f"dadus_real_{selected_category}.csv", "text/csv",
                            key=f"download_real_chart_{selected_category}"
                        )

                    else:
                        df_selected = df_filtered[
                            df_filtered["Prediksaun"].astype(str).str.strip() == selected_category
                        ].copy()

                        st.markdown(f"### 🔮 Dadus Prediksaun Decision Tree – Kategoria: `{selected_category}`")
                        st.info(f"Total funsionáriu ne'ebé Decision Tree prediz: **{len(df_selected)}**")

                        columns_show = [
                            "controlo_ativo_identificacao", "nome_pessoal", "id_sigap",
                            "id_grp", "sexo", "local_trabalho", "cargo",
                            "Asiduidade", "Pontualidade", "Produtividade",
                            "Kualidade_Servisu", "Kooperasaun", "Inisiativa",
                            "Disiplina", "Responsabilidade", target_col, "Prediksaun"
                        ]

                        columns_show = [c for c in columns_show if c in df_selected.columns]
                        st.dataframe(df_selected[columns_show], use_container_width=True, hide_index=True)

                        csv_data = df_selected.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Dadus Prediksaun (CSV)", csv_data,
                            f"dadus_prediksaun_{selected_category}.csv", "text/csv",
                            key=f"download_pred_chart_{selected_category}"
                        )

                    if st.button("❌ Taka Tabela", key="close_chart_comparison_table"):
                        st.session_state["comparison_selection"] = None
                        st.session_state["chart_key_version"] += 1
                        st.rerun()

                st.markdown("---")
                st.markdown("##### 🗺️ Gráfiku Avansadu: Desentralizasaun Dezempenu tuir Local de Trabalhu (Munisípiu)")
                if "local_trabalho" in df_filtered.columns:
                    fig_loc, ax_loc = plt.subplots(figsize=(10, 4.5))
                    df_loc_counts = pd.crosstab(df_filtered["local_trabalho"], df_filtered[target_col])
                    existing_cats = [c for c in categories if c in df_loc_counts.columns]
                    df_loc_counts = df_loc_counts.reindex(columns=existing_cats, fill_value=0)
                    
                    df_loc_counts.plot(kind="bar", stacked=True, ax=ax_loc, colormap="crest", edgecolor="none")
                    ax_loc.set_title("Distribuisaun Avaliasaun Dezempenu tuir Munisípiu / Local de Trabalhu", fontsize=11, fontweight="bold")
                    ax_loc.set_xlabel("Local de Trabalhu", fontsize=9)
                    ax_loc.set_ylabel("Total Funsionáriu", fontsize=9)
                    plt.xticks(rotation=45, ha="right")
                    ax_loc.legend(title="Kategoria", bbox_to_anchor=(1.02, 1), loc="upper left")
                    sns.despine()
                    st.pyplot(fig_loc)
                else:
                    st.info("Koluna 'local_trabalho' la dispoñível iha dataset atu halo grafiku ne'e.")

            with tab2:
                st.subheader("📋 Amostra Dadus (Preview)")
                st.dataframe(df_filtered.head(10), use_container_width=True)
                st.markdown("---")
                st.subheader("🚀 Performance Modelu Decision Tree")
                st.success(f"✅ Akurasi Modelu (Accuracy): **{acc * 100:.2f}%**")

                col_eval1, col_eval2 = st.columns(2)
                y_pred_test = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred_test)

                with col_eval1:
                    st.markdown("##### 📉 Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
                    st.pyplot(fig_cm)

                with col_eval2:
                    st.markdown("##### 📑 Classification Report")
                    unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
                    present_class_names = [le.classes_[i] for i in unique_labels]
                    report_dict = classification_report(y_test, y_pred_test, labels=unique_labels, target_names=present_class_names, output_dict=True, zero_division=0)
                    df_report = pd.DataFrame(report_dict).transpose()
                    st.dataframe(df_report.style.format(subset=["precision", "recall", "f1-score", "support"], formatter="{:.2f}"), use_container_width=True)

                st.markdown("---")
                st.subheader("🌳 Vizualizasaun Árbore Desizaun")
                max_depth_vis = st.slider("Hili Profundidade Árbore (Max Depth)", 1, 5, 3, key="tree_depth_slider")
                vis_model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth_vis, random_state=42)
                vis_model.fit(X_train, y_train)
                fig_tree, ax_tree = plt.subplots(figsize=(16, 9), dpi=100)
                plot_tree(vis_model, feature_names=nota_cols, class_names=le.classes_, filled=True, rounded=True, ax=ax_tree, fontsize=9)
                st.pyplot(fig_tree)

            with tab3:
                st.subheader("🔍 Prediksaun Funsionáriu Foun & Gestaun Dadus")
                
                extra_records = load_extra_from_db()
                st.session_state["extra_reports"] = extra_records

                st.markdown("##### 📋 Lista Dadus Funsionáriu Foun & Asaun Gestaun")
                if len(extra_records) > 0:
                    h1, h2, h3, h4 = st.columns([3, 2, 2.5, 2.5])
                    with h1: st.markdown("**Naran / ID SIGAP**")
                    with h2: st.markdown("**Munisípiu**")
                    with h3: st.markdown("**Kargo**")
                    with h4: st.markdown("**Rezultadu Avaliasaun & Asaun**")
                    st.markdown("<hr style='margin: 0px 0px 10px 0px;'>", unsafe_allow_html=True)

                    for i, rec in enumerate(extra_records):
                        c1, c2, c3, c4 = st.columns([3, 2, 2.5, 2.5])
                        with c1:
                            st.write(f"👤 **{rec.get('nome_pessoal')}**\n`{rec.get('id_sigap')}`")
                        with c2:
                            st.write(f"📍 {rec.get('local_trabalho')}")
                        with c3:
                            st.write(f"💼 {rec.get('cargo')}")
                        with c4:
                            res_val = rec.get('Rezultadu_Avaliasaun', 'N/A')
                            sub_res, sub_edit, sub_del = st.columns([1.5, 0.7, 0.7])
                            with sub_res:
                                st.markdown(f"⭐ **{res_val}**")
                            with sub_edit:
                                if st.button("✏️", key=f"edit_btn_{i}", help="Edita dadus ne'e"):
                                    st.session_state["edit_index"] = i
                                    st.rerun()
                            with sub_del:
                                with st.popover("🗑️", help="Hamos dadus ne'e"):
                                    st.markdown(f"Kerteza hakarak hamos **{rec.get('nome_pessoal')}**?")
                                    if st.button("I Hamos Duni", key=f"confirm_del_{i}"):
                                        if delete_extra_from_db_by_index(i):
                                            if st.session_state["edit_index"] == i:
                                                st.session_state["edit_index"] = None
                                            st.success("Hamos ona!")
                                            st.rerun()
                        st.markdown("<hr style='margin: 5px 0px; opacity: 0.2;'>", unsafe_allow_html=True)

                    if st.session_state["edit_index"] is not None:
                        st.info(f"⚠️ Atualmente hela iha Módudu Edisaun ba Index: **{st.session_state['edit_index']}**")
                        if st.button("❌ Kansela / Sai husi Módudu Edisaun", key="cancel_edit_mode"):
                            st.session_state["edit_index"] = None
                            st.rerun()
                else:
                    st.info("ℹ️ Sei la iha dadus foun rejisitadu iha database lokal.")

                st.markdown("---")
                idx_edit = st.session_state["edit_index"]
                def_val = {}
                if idx_edit is not None and idx_edit < len(st.session_state["extra_reports"]):
                    def_val = st.session_state["extra_reports"][idx_edit]
                    st.markdown(f"#### ✏️ Atualiza Dadus Funsionáriu (Index: {idx_edit})")
                else:
                    st.markdown("#### ➕ Input Funsionáriu Foun ba Prediksaun")

                with st.form("funsionariu_form"):
                    st.markdown("##### 📝 1. Informasaun Identidade Funsionáriu")
                    municipios = ["Aileu", "Ainaro", "Baucau", "Bobonaro", "Covalima", "Díli", "Ermera", "Lautém", "Liquiçá", "Manatuto", "Manufahi", "Oe-Cusse Ambeno", "Viqueque"]
                    
                    cargos_list = [
                        "Técnico Superior",
                        "Técnico Profissional",
                        "Assistente Administrativo",
                        "Oficial Administrativo",
                        "Assistente Técnico",
                        "Técnico Informática",
                        "Analista de Dados",
                        "Chefe de Unidade",
                        "Chefe de Departamento",
                        "Diretor Nacional"
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
                        "Regime Geral das Carreiras, Assistente Grau F, 1, PERMANENTE",
                    ]

                    cargo_def_val = def_val.get("cargo", "Técnico Superior")
                    cargo_index = cargos_list.index(cargo_def_val) if cargo_def_val in cargos_list else 0

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        nome_input = st.text_input("Naran Pessoal:", value=def_val.get("nome_pessoal", ""))
                        id_sigap_input = st.text_input("ID SIGAP:", value=def_val.get("id_sigap", ""))
                        sexo_input = st.selectbox("Sexo:", ["M", "F"], index=0 if def_val.get("sexo") == "M" else 1)
                        local_input = st.selectbox("Munisípiu / Local Trabalho:", municipios, index=municipios.index(def_val.get("local_trabalho")) if def_val.get("local_trabalho") in municipios else 5)
                    with col_f2:
                        id_grp_input = st.text_input("ID GRP:", value=def_val.get("id_grp", ""))
                        cargo_input = st.selectbox("Kargo:", cargos_list, index=cargo_index)
                        funcao_input = st.selectbox("Funsaun / Karreira:", funcoes, index=0)

                    st.markdown("##### 📊 2. Nota Avaliasaun Dezempenu (1.0 - 5.0)")
                    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                    with col_n1:
                        asid = st.number_input("Asiduidade:", 1.0, 5.0, float(def_val.get("Asiduidade", 4.0)), step=0.1)
                        pont = st.number_input("Pontualidade:", 1.0, 5.0, float(def_val.get("Pontualidade", 4.0)), step=0.1)
                    with col_n2:
                        prod = st.number_input("Produtividade:", 1.0, 5.0, float(def_val.get("Produtividade", 4.0)), step=0.1)
                        kual = st.number_input("Kualidade Servisu:", 1.0, 5.0, float(def_val.get("Kualidade_Servisu", 4.0)), step=0.1)
                    with col_n3:
                        koop = st.number_input("Kooperasaun:", 1.0, 5.0, float(def_val.get("Kooperasaun", 4.0)), step=0.1)
                        inis = st.number_input("Inisiativa:", 1.0, 5.0, float(def_val.get("Inisiativa", 4.0)), step=0.1)
                    with col_n4:
                        disi = st.number_input("Disiplina:", 1.0, 5.0, float(def_val.get("Disiplina", 4.0)), step=0.1)
                        resp = st.number_input("Responsabilidade:", 1.0, 5.0, float(def_val.get("Responsabilidade", 4.0)), step=0.1)

                    st.markdown("<br>", unsafe_allow_html=True)
                    btn_label = "💾 Atualiza Dadus" if idx_edit is not None else "🔮 Halo Prediksaun & Guarda"
                    submit_funs = st.form_submit_button(btn_label)

                    if submit_funs:
                        if not nome_input or not id_sigap_input:
                            st.error("⚠️ Naran no ID SIGAP keta mamuk!")
                        else:
                            input_df = pd.DataFrame([[asid, pont, prod, kual, koop, inis, disi, resp]], columns=nota_cols)
                            pred_encoded = model.predict(input_df)[0]
                            pred_label = le.inverse_transform([pred_encoded])[0]

                            media_calculated = np.mean([asid, pont, prod, kual, koop, inis, disi, resp])

                            record = {
                                "controlo_ativo_identificacao": "ATIVO",
                                "nome_pessoal": nome_input,
                                "id_sigap": id_sigap_input,
                                "id_grp": id_grp_input,
                                "sexo": sexo_input,
                                "local_trabalho": local_input,
                                "funcao": funcao_input,
                                "cargo": cargo_input,
                                "Asiduidade": asid,
                                "Pontualidade": pont,
                                "Produtividade": prod,
                                "Kualidade_Servisu": kual,
                                "Kooperasaun": koop,
                                "Inisiativa": inis,
                                "Disiplina": disi,
                                "Responsabilidade": resp,
                                "Media": float(media_calculated),
                                "Rezultadu_Avaliasaun": pred_label,
                            }

                            if idx_edit is not None:
                                update_extra_in_db_by_index(idx_edit, record)
                                st.session_state["edit_index"] = None
                                st.success(f"✅ Dadus ba **{nome_input}** atualiza ho susesu! Prediksaun foun: **{pred_label}**")
                            else:
                                save_extra_to_db(record)
                                st.success(f"✅ Prediksaun ba **{nome_input}**: **{pred_label}** (Guarda ona ba Database!)")

                            st.session_state["extra_reports"] = load_extra_from_db()
                            st.rerun()

    except Exception as e:
        st.sidebar.error(f"❌ Erru lee ficheiru: {e}")
else:
    st.info("👈 Favor submete (upload) ficheiru Excel iha sidebar atu hahú.")

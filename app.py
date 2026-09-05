from io import BytesIO
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
from models import carregar_modelo_pkl, treinar_modelo
from ui_components import render_custom_css

# ==========================================
# LISTA PADRAUN FUNÇÃO / KARREIRA CFP RDTL
# ==========================================
LISTA_FUNCOES_PADRAO = [
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

LISTA_MUNICIPIOS = [
    "Díli", "Baucau", "Ermera", "Bobonaro", "Aileu", "Ainaro", 
    "Covalima", "Lautém", "Liquiçá", "Manatuto", "Manufahi", 
    "Oé-Cusse Ambeno", "Viqueque", "Atauro"
]

LISTA_CARGOS = [
    "Técnico Superior",
    "Técnico Profissional",
    "Técnico Administrativo",
    "Assistente Administrative",
    "Chefe de Departamento",
    "Director Nacional",
    "Director-Geral",
    "Assessor / Consultor"
]

# ==========================================
# 0. KONFIGURASAUN PAJINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - RDTL",
    page_icon="🌳",
    layout="wide",
)

render_custom_css()

def load_data(file):
    return pd.read_excel(file)

# ==========================================
# 1. SISTEMA AUTENTIKASAUN / LOGIN
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); }
        .block-container { padding-top: 3rem !important; max-width: 450px !important; margin: 0 auto !important; }
        .cfp-login-card { background-color: #ffffff; padding: 35px; border-radius: 12px; border-top: 6px solid #D97706; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
        .cfp-header-title { color: #1E3A8A; font-weight: 800; font-size: 15px; text-align: center; line-height: 1.5; }
        .cfp-subtitle { text-align: center; color: #64748B; font-size: 12px; margin-bottom: 25px; font-weight: 600; }
        div.stButton > button { background-color: #1E3A8A !important; color: white !important; width: 100%; border-radius: 8px; font-weight: bold; height: 45px; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <div class="cfp-login-card">
                <div style="text-align: center; font-size: 50px;">🌳</div>
                <div class="cfp-header-title">COMISSÃO DA FUNÇÃO PÚBLICA<br>REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE</div>
                <div class="cfp-subtitle">Portal de Gestão e Classificação de Desempenho</div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username:", placeholder="Hatama username")
            password = st.text_input("Password:", type="password", placeholder="Hatama password")
            submit_login = st.form_submit_button("ENTRADA / LOGIN")
            
            if submit_login:
                try:
                    if username == st.secrets["username"] and password == st.secrets["password"]:
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("⚠️ Username ka Password sala!")
                except Exception:
                    if username == "admin" and password == "admin123":
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("⚠️ Username ka Password sala!")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ==========================================
# 2. PANEL PRINSIPAL (LOGGED IN)
# ==========================================
st.sidebar.markdown("### 🌳 CFP-RDTL Portal")
if st.sidebar.button("🚪 Logout / Sai"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown('<p class="main-title">🌳 Sistema Klasifikasaun Dezempenu Funsionáriu CFP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Intelijénsia Artifisiál Decision Tree ne\'ebé adaptadu ho modelu ofisiál CFP RDTL.</p>', unsafe_allow_html=True)

init_db()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()
if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

# Funsaun Gerador PDF
def generate_pdf_report(df_data, title_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=11, textColor=colors.HexColor('#1E3A8A'), alignment=1, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#4B5563'), spaceAfter=15, alignment=1)

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

    t = Table(table_data, colWidths=[140, 70, 90, 110, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# 3. GESTAUN DATASET (SIDEBAR)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Gestaun Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

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
            "Column13": "Asiduidade",
            "Column14": "Pontualidade",
            "Column15": "Produtividade",
            "Column16": "Kualidade_Servisu",
            "Column17": "Kooperasaun",
            "Column18": "Inisiativa",
            "Column19": "Disiplina",
            "Column20": "Responsabilidade",
            "Column22": "Rezultadu_Avaliasaun",
        }
        df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)

        nota_cols = [
            "Asiduidade", "Pontualidade", "Produtividade", "Kualidade_Servisu",
            "Kooperasaun", "Inisiativa", "Disiplina", "Responsabilidade"
        ]

        nota_cols = [c for c in nota_cols if c in df_raw.columns]
        target_col = "Rezultadu_Avaliasaun"

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

        # Filtru Kargo
        cargo_list = ["Tomak (Hotu-hotu)"] + sorted(list(df["cargo"].dropna().unique()))
        selected_cargo = st.sidebar.selectbox("Filtru Kargo:", cargo_list)
        df_filtered = df.copy() if selected_cargo == "Tomak (Hotu-hotu)" else df[df["cargo"] == selected_cargo]

        # Dynamic Lista Funcoes husi dataset ka husi padraun
        if "funcao" in df.columns:
            opcoes_funcao = sorted(list(set(df["funcao"].dropna().unique()).union(set(LISTA_FUNCOES_PADRAO))))
        else:
            opcoes_funcao = LISTA_FUNCOES_PADRAO

        # Treina Modelu
        model, le, X_train, X_test, y_train, y_test = treinar_modelo(df, nota_cols, target_col)
        df_filtered["Prediksaun"] = le.inverse_transform(model.predict(df_filtered[nota_cols]))
        acc = accuracy_score(y_test, model.predict(X_test))

        # ==========================================
        # 4. PAINÉL TABS (3 ABA PRINSIPAL)
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard Analítiku", "⚙️ Modelu & Performance", "🔮 Prediksaun & Gestaun Dadus"])

        # ------------------------------------------
        # TAB 1: DASHBOARD
        # ------------------------------------------
        with tab1:
            st.markdown("### 📈 Sumáriu Dezempenu Funsionáriu")
            total_funs = len(df_filtered)

            counts_real = df_filtered[target_col].value_counts()
            counts_pred = df_filtered["Prediksaun"].value_counts()

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Total Funsionáriu", total_funs)
            col_m2.metric("Muito Bom", counts_real.get("Muito Bom", 0), f"Pred: {counts_pred.get('Muito Bom', 0)}")
            col_m3.metric("Bom", counts_real.get("Bom", 0), f"Pred: {counts_pred.get('Bom', 0)}")
            col_m4.metric("Suficiente", counts_real.get("Suficiente", 0), f"Pred: {counts_pred.get('Suficiente', 0)}")
            col_m5.metric("Insuficiente", counts_real.get("Insuficiente", 0), f"Pred: {counts_pred.get('Insuficiente', 0)}")

            st.markdown("---")
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")
                categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                if go is not None:
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(name="Dadus Reál", x=categories, y=[counts_real.get(c, 0) for c in categories], marker_color="#1E3A8A"))
                    fig_compare.add_trace(go.Bar(name="Prediksaun Tree", x=categories, y=[counts_pred.get(c, 0) for c in categories], marker_color="#3B82F6"))
                    fig_compare.update_layout(barmode="group", height=380, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_compare, use_container_width=True)

            with col_g2:
                st.markdown("##### 🍩 Proporsaun Persentajen (Reál vs Prediksaun)")
                if make_subplots is not None and go is not None:
                    fig_donut = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]], subplot_titles=["Reál", "Prediksaun"])
                    fig_donut.add_trace(go.Pie(labels=categories, values=[counts_real.get(c, 0) for c in categories], hole=0.4), 1, 1)
                    fig_donut.add_trace(go.Pie(labels=categories, values=[counts_pred.get(c, 0) for c in categories], hole=0.4), 1, 2)
                    fig_donut.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(fig_donut, use_container_width=True)

            st.markdown("---")
            st.markdown("##### 📄 Eksporta Relatóriu PDF")
            pdf_bytes = generate_pdf_report(df_filtered, f"RELATÓRIU EVALUASAUN - KARGO: {selected_cargo}")
            st.download_button(
                label="📥 Download Relatóriu (PDF)",
                data=pdf_bytes,
                file_name=f"relatorio_cfp_{selected_cargo}.pdf",
                mime="application/pdf",
            )

        # ------------------------------------------
        # TAB 2: PERFORMANCE MODELU
        # ------------------------------------------
        with tab2:
            st.subheader("🚀 Performance Modelu Decision Tree")
            st.success(f"✅ Akurasi Modelu (Accuracy Score): **{acc * 100:.2f}%**")

            col_eval1, col_eval2 = st.columns(2)
            y_pred_test = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_test)

            with col_eval1:
                st.markdown("##### 📉 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
                st.pyplot(fig_cm)

            with col_eval2:
                st.markdown("##### 📑 Classification Report")
                report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.2f}"), use_container_width=True)

            st.markdown("---")
            st.subheader("🌳 Vizualizasaun Árbore Desizaun (Decision Tree)")
            max_depth_vis = st.slider("Hili Profundidade Árbore (Max Depth):", 1, 5, 3)
            vis_model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth_vis, random_state=42)
            vis_model.fit(X_train, y_train)
            fig_tree, ax_tree = plt.subplots(figsize=(14, 7))
            plot_tree(vis_model, feature_names=nota_cols, class_names=le.classes_, filled=True, rounded=True, ax=ax_tree, fontsize=8)
            st.pyplot(fig_tree)

        # ------------------------------------------
        # TAB 3: PREDIKSAUN & GESTAUN DADUS
        # ------------------------------------------
        with tab3:
            st.subheader("🔍 Prediksaun Funsionáriu Foun & Gestaun Dadus")

            # FORMULÁRIU PREDIKSAUN HO LISTA FUNÇOES
            with st.form("funsionariu_form"):
                st.markdown("##### 📝 Informasaun Identidade & Evaluasaun")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    nome_input = st.text_input("Naran Pessoal:")
                    id_sigap_input = st.text_input("ID SIGAP:")
                    sexo_input = st.selectbox("Sexo:", ["M", "F"])
                    local_input = st.selectbox("Munisípiu:", LISTA_MUNICIPIOS)
                with col_f2:
                    id_grp_input = st.text_input("ID GRP:")
                    cargo_input = st.selectbox("Kargo:", LISTA_CARGOS)
                    funcao_input = st.selectbox("Funsaun / Karreira (Regime Geral):", opcoes_funcao)

                st.markdown("##### 📊 Nota Indikadór (1.0 - 5.0)")
                col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                asid = col_n1.number_input("Asiduidade:", 1.0, 5.0, 4.0, 0.1)
                pont = col_n1.number_input("Pontualidade:", 1.0, 5.0, 4.0, 0.1)
                prod = col_n2.number_input("Produtividade:", 1.0, 5.0, 4.0, 0.1)
                kual = col_n2.number_input("Kualidade Servisu:", 1.0, 5.0, 4.0, 0.1)
                koop = col_n3.number_input("Kooperasaun:", 1.0, 5.0, 4.0, 0.1)
                inis = col_n3.number_input("Inisiativa:", 1.0, 5.0, 4.0, 0.1)
                disi = col_n4.number_input("Disiplina:", 1.0, 5.0, 4.0, 0.1)
                resp = col_n4.number_input("Responsabilidade:", 1.0, 5.0, 4.0, 0.1)

                submit_funs = st.form_submit_button("🔮 Halo Prediksaun & Guarda")

                if submit_funs:
                    if not nome_input or not id_sigap_input:
                        st.error("⚠️ Naran no ID SIGAP keta mamuk!")
                    else:
                        input_df = pd.DataFrame([[asid, pont, prod, kual, koop, inis, disi, resp]], columns=nota_cols)
                        pred_encoded = model.predict(input_df)[0]
                        pred_label = le.inverse_transform([pred_encoded])[0]

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
                            "Rezultadu_Avaliasaun": pred_label,
                        }
                        save_extra_to_db(record)
                        st.success(f"✅ Prediksaun ba **{nome_input}**: **{pred_label}**")
                        st.rerun()

            st.markdown("---")
            st.markdown("##### 📋 Gestaun Dadus Extra Iha Database (SQLite)")
            extra_db_records = load_extra_from_db()
            
            if len(extra_db_records) > 0:
                df_db_show = pd.DataFrame(extra_db_records)
                st.dataframe(df_db_show[["nome_pessoal", "id_sigap", "funcao", "cargo", "local_trabalho", "Rezultadu_Avaliasaun"]], use_container_width=True)

                col_db_del, _ = st.columns([1, 3])
                with col_db_del:
                    del_idx = st.number_input("Hamos por Index (0..N):", 0, len(extra_db_records)-1, 0)
                    if st.button("🗑️ Hamos Dadus Husi DB"):
                        delete_extra_from_db_by_index(del_idx)
                        st.success(f"Dadus iha index {del_idx} hamos tiha ho susesu!")
                        st.rerun()
            else:
                st.info("💡 Seidauk iha dadus extra foun ne'ebé rai ba SQLite Database.")

            st.markdown("---")
            st.markdown("##### 📄 Tabela Funsionáriu Hotu-Hotu (Filtru Active)")
            st.dataframe(df_filtered[["nome_pessoal", "id_sigap", "funcao", "cargo", "local_trabalho", "Rezultadu_Avaliasaun", "Prediksaun"]], use_container_width=True)

    except Exception as e:
        st.sidebar.error(f"❌ Erru foti/lee ficheiru: {e}")
else:
    st.info("👈 Favor hili no upload ficheiru Excel (.xlsx) iha sidebar hodi ativa sistema.")

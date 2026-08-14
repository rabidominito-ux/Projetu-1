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

from database import (
    delete_extra_from_db_by_index,
    init_db,
    load_extra_from_db,
    save_extra_to_db,
    update_extra_in_db_by_index,
)
from models import treinar_modelo
from ui_components import render_custom_css

st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - Decision Tree",
    page_icon="📊",
    layout="wide",
)

render_custom_css()

# ==========================================
# SISTEMA LOGIN / AUTENTIKASAUN (Dezain Custom)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <style>
        .stApp {
            background-color: #2563EB;
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        .login-box-wrapper {
            background-color: #2563EB;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            max-width: 450px;
            margin: 0 auto;
        }
        .login-header-title {
            background-color: #6D28D9;
            color: white;
            padding: 18px 12px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 13px;
            text-align: center;
            letter-spacing: 0.5px;
            line-height: 1.4;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .login-instruction {
            text-align: center;
            color: white;
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 15px;
            letter-spacing: 0.5px;
        }
        div.stButton > button {
            background-color: #6D28D9 !important;
            color: white !important;
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            padding: 10px;
            border: none;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        div.stButton > button:hover {
            background-color: #5B21B6 !important;
        }
        label {
            color: white !important;
            font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("""
            <div class="login-header-title">
                SISTEMA KLASIFIKASAUN BA AVALIASAUN DESEMPENHU FUNSIONARIU IHA KOMISAUN FUNSAUN PUBLIKA  UTILIZA  ALGORITMA DECISION TREE
            </div>
            <div class="login-instruction">
                FAVOR LOGIN!
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username:", placeholder="Username")
            password = st.text_input("Password:", type="password", placeholder="Password!")
            st.markdown("<br>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("LOGIN")
            
            if submit_login:
                try:
                    if username == st.secrets["username"] and password == st.secrets["password"]:
                        st.session_state["authenticated"] = True
                        st.success("Login susesu! Kontinua hela...")
                        st.rerun()
                    else:
                        st.error("⚠️ Username ka Password sala! Favor koko fali.")
                except Exception:
                    st.error("⚠️ Konfigurasaun Secrets seidauk iha Streamlit Cloud ka lokál.")
                    
    st.stop()

# ==========================================
# APLIKASAUN PRINCIPAL (Pós-Login)
# ==========================================
st.sidebar.markdown("### 👤 Konta Asesu")
if st.sidebar.button("🚪 Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown('<p class="main-title">📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasaun Inteligénsia Artifisiál uza algoritmu Decision Tree hodi analiza no klasifika dezempenu funsionáriu bazeia ba indikadór Komisaun Função Pública (CFP).</p>', unsafe_allow_html=True)

init_db()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()

if "edit_index" not in st.session_state:
    st.session_state["edit_index"] = None

if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None

def generate_pdf_report(df_data, title_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1E3A8A'), spaceAfter=15, alignment=1
    )
    subtitle_style = ParagraphStyle(
        'SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4B5563'), spaceAfter=15, alignment=1
    )

    elements.append(Paragraph("REPUBLICA DEMOCRATICA DE TIMOR-LESTE", title_style))
    elements.append(Paragraph("COMISSÃO DA FUNÇÃO PÚBLICA (CFP)", title_style))
    elements.append(Paragraph(title_report, subtitle_style))
    elements.append(Spacer(1, 10))

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
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

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
                "📊 Dashboard Analítiku", "⚙️ Modelu & Performance", "🔮 Prediksaun & Jere Dadus"
            ])

            with tab1:
                st.markdown("### 📈 Sumáriu Dezempenu Funsionáriu")
                total_funs = len(df_filtered)
                counts_real = df_filtered[target_col].value_counts()

                mb_pct = (counts_real.get("Muito Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                b_pct = (counts_real.get("Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                s_pct = (counts_real.get("Suficiente", 0) / total_funs) * 100 if total_funs > 0 else 0
                i_pct = (counts_real.get("Insuficiente", 0) / total_funs) * 100 if total_funs > 0 else 0

                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                with col_m1:
                    if st.button(f"📊 Total Funsionáriu\n\n{total_funs}", key="btn_m1"):
                        st.session_state["selected_category"] = "Tomak" if st.session_state["selected_category"] != "Tomak" else None
                with col_m2:
                    if st.button(f"⭐ Muito Bom\n\n{counts_real.get('Muito Bom', 0)}\n({mb_pct:.1f}%)", key="btn_m2"):
                        st.session_state["selected_category"] = "Muito Bom" if st.session_state["selected_category"] != "Muito Bom" else None
                with col_m3:
                    if st.button(f"✨ Bom\n\n{counts_real.get('Bom', 0)}\n({b_pct:.1f}%)", key="btn_m3"):
                        st.session_state["selected_category"] = "Bom" if st.session_state["selected_category"] != "Bom" else None
                with col_m4:
                    if st.button(f"📌 Suficiente\n\n{counts_real.get('Suficiente', 0)}\n({s_pct:.1f}%)", key="btn_m4"):
                        st.session_state["selected_category"] = "Suficiente" if st.session_state["selected_category"] != "Suficiente" else None
                with col_m5:
                    if st.button(f"⚠️ Insuficiente\n\n{counts_real.get('Insuficiente', 0)}\n({i_pct:.1f}%)", key="btn_m5"):
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
                with col_g1:
                    st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                    real_counts = [counts_real.get(cat, 0) for cat in categories]
                    pred_counts = [df_filtered["Prediksaun"].value_counts().get(cat, 0) for cat in categories]
                    x = np.arange(len(categories))
                    width = 0.35
                    rects1 = ax.bar(x - width/2, real_counts, width, label="Dadus Reál", color="#2563EB", alpha=0.9)
                    rects2 = ax.bar(x + width/2, pred_counts, width, label="Prediksaun Tree", color="#10B981", alpha=0.9)
                    ax.bar_label(rects1, padding=3, fontsize=8)
                    ax.bar_label(rects2, padding=3, fontsize=8)
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
                    colors_list = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]
                    if sum(sizes) > 0:
                        ax2.pie(sizes, labels=categories, autopct="%1.1f%%", startangle=90, colors=colors_list, wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
                    st.pyplot(fig2)

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
                        "Regime Geral das Carreiras, Técnico Superior Grau B, 9, PERMANENTE"
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
                        "Diretor Nacional"
                    ]

                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        ativo_opts = ["Ativo", "La Ativo"]
                        cur_ativo = def_val.get("controlo_ativo_identificacao", "Ativo")
                        txt_ativo = st.selectbox("Controlo Ativo Identifikasaun", ativo_opts, index=ativo_opts.index(cur_ativo) if cur_ativo in ativo_opts else 0)
                        txt_nome = st.text_input("Naran Pessoal*", def_val.get("nome_pessoal", ""))
                        txt_sigap = st.text_input("ID SIGAP (Numeriku no Símbolu)*", def_val.get("id_sigap", ""))
                        cur_sexo = def_val.get("sexo", "M")
                        txt_sexo = st.selectbox("Sexo", ["M", "F"], index=0 if cur_sexo == "M" else 1)
                    with col_i2:
                        txt_inst = st.text_input("Instituisaun", def_val.get("instituicao", "CFP"))
                        cur_local = def_val.get("local_trabalho", "Díli")
                        txt_local = st.selectbox("Local Trabalhu", municipios, index=municipios.index(cur_local) if cur_local in municipios else 0)
                        
                        try:
                            default_date = pd.to_datetime(def_val.get("data_de_nascimento", "1995-01-01")).date()
                        except:
                            default_date = pd.to_datetime("1995-01-01").date()
                        txt_nascimento = st.date_input("Data de Nascimento", value=default_date)
                    with col_i3:
                        cur_func = def_val.get("funcao", funcoes[0])
                        txt_funcao = st.selectbox("Funsaun", funcoes, index=funcoes.index(cur_func) if cur_func in funcoes else 0)
                        
                        cur_cargo = def_val.get("cargo", cargos[0])
                        txt_cargo = st.selectbox("Kargo", cargos, index=cargos.index(cur_cargo) if cur_cargo in cargos else 0)
                        txt_grp = st.text_input("ID GRP", def_val.get("id_grp", ""))

                    st.markdown("##### 📊 2. Indikadór Avaliasaun Funsionáriu")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        p_asid = st.slider("Asiduidade", 1.0, 5.0, float(def_val.get("Asiduidade", 4.0)), 0.5)
                        p_pont = st.slider("Pontualidade", 1.0, 5.0, float(def_val.get("Pontualidade", 4.0)), 0.5)
                    with col_b:
                        p_prod = st.slider("Produtividade", 1.0, 5.0, float(def_val.get("Produtividade", 4.0)), 0.5)
                        p_kual = st.slider("Kualidade Servisu", 1.0, 5.0, float(def_val.get("Kualidade_Servisu", 4.0)), 0.5)
                    with col_c:
                        p_koop = st.slider("Kooperasaun", 1.0, 5.0, float(def_val.get("Kooperasaun", 4.0)), 0.5)
                        p_inis = st.slider("Inisiativa", 1.0, 5.0, float(def_val.get("Inisiativa", 4.0)), 0.5)
                    with col_d:
                        p_disp = st.slider("Disiplina", 1.0, 5.0, float(def_val.get("Disiplina", 4.0)), 0.5)
                        p_resp = st.slider("Responsabilidade", 1.0, 5.0, float(def_val.get("Responsabilidade", 4.0)), 0.5)

                    submit_pred = st.form_submit_button("💾 Salva / Prediksaun")

                    if submit_pred:
                        if not txt_nome.strip() or not txt_sigap.strip():
                            st.error("⚠️ Favór preenxe Naran Pessoal no ID SIGAP ho loos!")
                        elif any(c.isalpha() for c in txt_sigap):
                            st.error("⚠️ ID SIGAP labele uza letra/alfabetu! Tenke uza de'it númeru no símbolu.")
                        else:
                            input_data = np.array([[p_asid, p_pont, p_prod, p_kual, p_koop, p_inis, p_disp, p_resp]])
                            pred_encoded = model.predict(input_data)
                            pred_label = le.inverse_transform(pred_encoded)[0]

                            new_report = {
                                "controlo_ativo_identificacao": txt_ativo, "nome_pessoal": txt_nome, "id_sigap": txt_sigap.strip(),
                                "sexo": txt_sexo, "instituicao": txt_inst, "local_trabalho": txt_local,
                                "data_de_nascimento": str(txt_nascimento), "funcao": txt_funcao, "cargo": txt_cargo,
                                "id_grp": txt_grp, "Asiduidade": p_asid, "Pontualidade": p_pont, "Produtividade": p_prod,
                                "Kualidade_Servisu": p_kual, "Kooperasaun": p_koop, "Inisiativa": p_inis,
                                "Disiplina": p_disp, "Responsabilidade": p_resp, "Rezultadu_Avaliasaun": pred_label
                            }

                            if idx_edit is not None:
                                update_extra_in_db_by_index(idx_edit, new_report)
                                st.session_state["edit_index"] = None
                                st.success("✅ Atualizadu ho suksesu!")
                                st.rerun()
                            else:
                                if save_extra_to_db(new_report):
                                    st.success(f"✅ Rai ona! Prediksaun Dezempenu: **{pred_label}**")
                                    st.rerun()
                                else:
                                    st.error("⚠️ Erro: ID SIGAP ne'e iha ona (Duplikadu). Favór uza ID seluk.")

    except Exception as e:
        st.error(f"⚠️ Erro iha prosesamentu fail: {e}")
else:
    st.info("👈 Favor upload uluk ficheiru Excel (`.xlsx`) iha sidebar hodi hahú sistema.")

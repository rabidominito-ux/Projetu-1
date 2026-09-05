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
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import joblib
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
    verify_user,
)
from ui_components import render_custom_css, render_header, render_kpi_card

# --- DEFINE VARIAVEL GLOBAL ---
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
MODEL_PATH = "modelu_cfp.pkl"

def carregar_modelo_colab():
    """Karga diretu modelu PKL husi Colab no prepara LabelEncoder"""
    model = joblib.load(MODEL_PATH)
    le = LabelEncoder()
    le.fit(["Bom", "Insuficiente", "Muito Bom", "Suficiente"])
    return model, le

st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - RDTL",
    page_icon="🌳",
    layout="wide",
)

render_custom_css()
init_db()

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <style>
        .login-card {
            background-color: #FFFFFF;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
            border-top: 5px solid #2563EB;
            max-width: 450px;
            margin: 3rem auto;
        }
        .login-header-title {
            color: #0F172A;
            font-weight: 800;
            font-size: 16px;
            text-align: center;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .login-subtitle {
            text-align: center;
            color: #475569;
            font-size: 13px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-size: 55px; margin-bottom: 10px;">🌳</div>
        <div class="login-header-title">COMISSÃO DA FUNÇÃO PÚBLICA<br>REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE</div>
        <div class="login-subtitle">Portal de Gestão e Classificação de Desempenho (Decision Tree)</div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username:", placeholder="Hatama ita-nia username")
        password = st.text_input("Password:", type="password", placeholder="Hatama ita-nia password")
        st.markdown("<br>", unsafe_allow_html=True)
        submit_login = st.form_submit_button("ENTRADA / LOGIN", use_container_width=True)
        
        if submit_login:
            if verify_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["user_role"] = username
                st.success("Login susesu!")
                st.rerun()
            elif "username" in st.secrets and "password" in st.secrets:
                if username == st.secrets["username"] and password == st.secrets["password"]:
                    st.session_state["authenticated"] = True
                    st.success("Login susesu!")
                    st.rerun()
                else:
                    st.error("⚠️ Username ka Password sala!")
            else:
                st.error("⚠️ Credenciais la válidu ka dadus user la hetan.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

render_header()

st.sidebar.markdown("## 🌳 CFP-RDTL Portal")
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout / Sai", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()

if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = None

if "chart_key_version" not in st.session_state:
    st.session_state["chart_key_version"] = 0

def generate_pdf_report(df_data, title_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=11, textColor=colors.HexColor('#0F172A'), spaceAfter=2, alignment=1, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'SubTitleStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15, alignment=1, fontName='Helvetica'
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
            str(row.get(target_col, ""))
        ])

    t = Table(table_data, colWidths=[150, 70, 90, 110, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
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
                label="⬇️ Download Backup (CSV)", data=csv_full, file_name="dataset_cfp_filtrado.csv", mime="text/csv", use_container_width=True
            )

            # Karga diretu modelo .pkl husi Colab
            model, le = carregar_modelo_colab()

            # Executa prediksaun uzando modelo Colab
            preds_encoded = model.predict(df_filtered[nota_cols])
            df_filtered["Prediksaun"] = le.inverse_transform(preds_encoded)

            y_true_encoded = le.transform(df_filtered[target_col])
            acc = accuracy_score(y_true_encoded, preds_encoded)

            tab1, tab2, tab3 = st.tabs([
                "📊 Dashboard Analítiku", "⚙️ Modelu & Performance", "🔮 Prediksaun & Gestaun Dadus"
            ])

            with tab1:
                st.markdown("### 📈 Sumáriu Dezempenu Funsionáriu")
                total_funs = len(df_filtered)
                
                counts_real = df_filtered[target_col].value_counts()
                mb_real_pct = (counts_real.get("Muito Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                b_real_pct = (counts_real.get("Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                s_real_pct = (counts_real.get("Suficiente", 0) / total_funs) * 100 if total_funs > 0 else 0
                i_real_pct = (counts_real.get("Insuficiente", 0) / total_funs) * 100 if total_funs > 0 else 0

                counts_pred = df_filtered["Prediksaun"].value_counts()
                mb_pred_pct = (counts_pred.get("Muito Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                b_pred_pct = (counts_pred.get("Bom", 0) / total_funs) * 100 if total_funs > 0 else 0
                s_pred_pct = (counts_pred.get("Suficiente", 0) / total_funs) * 100 if total_funs > 0 else 0
                i_pred_pct = (counts_pred.get("Insuficiente", 0) / total_funs) * 100 if total_funs > 0 else 0

                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                with col_m1:
                    render_kpi_card("Total Funsionáriu", str(total_funs), "👥 Dadus Filtradu")
                    if st.button("Ver Dadus Tomak", key="btn_m1", use_container_width=True):
                        st.session_state["selected_category"] = "Tomak" if st.session_state["selected_category"] != "Tomak" else None
                with col_m2:
                    render_kpi_card("Muito Bom", f"{mb_real_pct:.1f}%", f"Pred: {mb_pred_pct:.1f}%")
                    if st.button("Filtru Muito Bom", key="btn_m2", use_container_width=True):
                        st.session_state["selected_category"] = "Muito Bom" if st.session_state["selected_category"] != "Muito Bom" else None
                with col_m3:
                    render_kpi_card("Bom", f"{b_real_pct:.1f}%", f"Pred: {b_pred_pct:.1f}%")
                    if st.button("Filtru Bom", key="btn_m3", use_container_width=True):
                        st.session_state["selected_category"] = "Bom" if st.session_state["selected_category"] != "Bom" else None
                with col_m4:
                    render_kpi_card("Suficiente", f"{s_real_pct:.1f}%", f"Pred: {s_pred_pct:.1f}%")
                    if st.button("Filtru Suficiente", key="btn_m4", use_container_width=True):
                        st.session_state["selected_category"] = "Suficiente" if st.session_state["selected_category"] != "Suficiente" else None
                with col_m5:
                    render_kpi_card("Insuficiente", f"{i_real_pct:.1f}%", f"Pred: {i_pred_pct:.1f}%")
                    if st.button("Filtru Insuficiente", key="btn_m5", use_container_width=True):
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
                        st.download_button(label="📥 Download CSV", data=csv_filtered, file_name=f"relatorio_cfp_{selected_cat}.csv", mime="text/csv", key="dl_filtered_csv", use_container_width=True)
                    with dl_col2:
                        pdf_buffer = generate_pdf_report(df_table, f"Relatóriu Dezempenu Funsionáriu - {selected_cat}")
                        st.download_button(label="📄 Download Relatóriu PDF Ofisiál", data=pdf_buffer, file_name=f"relatorio_cfp_{selected_cat}.pdf", mime="application/pdf", key="dl_filtered_pdf", use_container_width=True)

                    if st.button("❌ Subar Tabela", key="hide_table_btn", use_container_width=True):
                        st.session_state["selected_category"] = None
                        st.rerun()

                st.markdown("---")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")
                    if go is not None:
                        categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                        real_counts = [counts_real.get(cat, 0) for cat in categories]
                        pred_counts = [counts_pred.get(cat, 0) for cat in categories]

                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Bar(
                            name="Dadus Reál", x=categories, y=real_counts, marker_color="#2563EB"
                        ))
                        fig_compare.add_trace(go.Bar(
                            name="Prediksaun Tree", x=categories, y=pred_counts, marker_color="#60A5FA"
                        ))
                        fig_compare.update_layout(barmode="group", height=400, margin=dict(l=20, r=20, t=30, b=20))

                        chart_key_bar = f"grafiku_real_vs_pred_{st.session_state['chart_key_version']}"
                        st.plotly_chart(fig_compare, use_container_width=True, key=chart_key_bar)

                with col_g2:
                    st.markdown("##### 🍩 Proporsaun Persentajen (Reál vs Prediksaun)")
                    if go is not None and make_subplots is not None:
                        categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                        colors_map = {"Muito Bom": "#1D4ED8", "Bom": "#3B82F6", "Suficiente": "#F59E0B", "Insuficiente": "#EF4444"}
                        sizes_real = [counts_real.get(cat, 0) for cat in categories]
                        sizes_pred = [counts_pred.get(cat, 0) for cat in categories]

                        fig_donut = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]], subplot_titles=["<b>Dadus Reál</b>", "<b>Prediksaun</b>"])
                        fig_donut.add_trace(go.Pie(labels=categories, values=sizes_real, hole=0.4, marker_colors=[colors_map[cat] for cat in categories]), 1, 1)
                        fig_donut.add_trace(go.Pie(labels=categories, values=sizes_pred, hole=0.4, marker_colors=[colors_map[cat] for cat in categories]), 1, 2)
                        fig_donut.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)

                        chart_key_donut = f"grafiku_donut_{st.session_state['chart_key_version']}"
                        st.plotly_chart(fig_donut, use_container_width=True, key=chart_key_donut)

            with tab2:
                st.subheader("📋 Amostra Dadus (Preview)")
                st.dataframe(df_filtered.head(10), use_container_width=True)
                st.markdown("---")
                st.subheader("🚀 Performance Modelu Decision Tree (Husi Colab)")
                st.success(f"✅ Akurasi Modelu (Accuracy): **{acc * 100:.2f}%**")

                col_eval1, col_eval2 = st.columns(2)
                cm = confusion_matrix(y_true_encoded, preds_encoded)

                with col_eval1:
                    st.markdown("##### 📉 Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
                    st.pyplot(fig_cm)

                with col_eval2:
                    st.markdown("##### 📑 Classification Report")
                    unique_labels = np.unique(np.concatenate((y_true_encoded, preds_encoded)))
                    present_class_names = [le.classes_[i] for i in unique_labels]
                    report_dict = classification_report(y_true_encoded, preds_encoded, labels=unique_labels, target_names=present_class_names, output_dict=True, zero_division=0)
                    df_report = pd.DataFrame(report_dict).transpose()
                    st.dataframe(df_report.style.format(subset=["precision", "recall", "f1-score", "support"], formatter="{:.2f}"), use_container_width=True)

                st.markdown("---")
                st.subheader("🌳 Vizualizasaun Árbore Desizaun (Modelu Exatu husi PKL)")
                fig_tree, ax_tree = plt.subplots(figsize=(16, 9), dpi=100)
                plot_tree(model, feature_names=nota_cols, class_names=le.classes_, filled=True, rounded=True, ax=ax_tree, fontsize=8)
                st.pyplot(fig_tree)

            with tab3:
                st.subheader("🔍 Prediksaun Funsionáriu Foun & Gestaun Dadus")
                extra_records = load_extra_from_db()
                st.session_state["extra_reports"] = extra_records

                if len(extra_records) > 0:
                    for i, rec in enumerate(extra_records):
                        c1, c2, c3, c4 = st.columns([3, 2, 2.5, 2.5])
                        with c1: st.write(f"👤 **{rec.get('nome_pessoal')}** (`{rec.get('id_sigap')}`)")
                        with c2: st.write(f"📍 {rec.get('local_trabalho')}")
                        with c3: st.write(f"💼 {rec.get('cargo')}")
                        with c4:
                            res_val = rec.get(target_col, 'N/A')
                            sub_res, sub_del = st.columns([2, 1])
                            with sub_res: st.markdown(f"⭐ **{res_val}**")
                            with sub_del:
                                if st.button("🗑️", key=f"del_extra_{i}"):
                                    delete_extra_from_db_by_index(i)
                                    st.rerun()
                else:
                    st.info("ℹ️ Seidauk iha dadus foun rejisitadu iha database lokal.")

                st.markdown("---")
                st.markdown("#### ➕ Input Funsionáriu Foun ba Prediksaun")

                with st.form("funsionariu_form"):
                    municipios = ["Aileu", "Ainaro", "Baucau", "Bobonaro", "Covalima", "Díli", "Ermera", "Lautém", "Liquiçá", "Manatuto", "Manufahi", "Oe-Cusse Ambeno", "Viqueque"]
                    cargos_list = ["Técnico Superior", "Técnico Profissional", "Assistente Administrativo", "Diretor Nacional"]

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        nome_input = st.text_input("Naran Pessoal:")
                        id_sigap_input = st.text_input("ID SIGAP:")
                        sexo_input = st.selectbox("Sexo:", ["M", "F"])
                        local_input = st.selectbox("Munisípiu:", municipios)
                    with col_f2:
                        id_grp_input = st.text_input("ID GRP:")
                        cargo_input = st.selectbox("Kargo:", cargos_list)

                    st.markdown("##### 📊 Nota Avaliasaun Dezempenu (1.0 - 5.0)")
                    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                    with col_n1:
                        asid = st.number_input("Asiduidade:", 1.0, 5.0, 4.0)
                        pont = st.number_input("Pontualidade:", 1.0, 5.0, 4.0)
                    with col_n2:
                        prod = st.number_input("Produtividade:", 1.0, 5.0, 4.0)
                        kual = st.number_input("Kualidade Servisu:", 1.0, 5.0, 4.0)
                    with col_n3:
                        koop = st.number_input("Kooperasaun:", 1.0, 5.0, 4.0)
                        inis = st.number_input("Inisiativa:", 1.0, 5.0, 4.0)
                    with col_n4:
                        disi = st.number_input("Disiplina:", 1.0, 5.0, 4.0)
                        resp = st.number_input("Responsabilidade:", 1.0, 5.0, 4.0)

                    submit_funs = st.form_submit_button("🔮 Halo Prediksaun & Guarda", use_container_width=True)

                    if submit_funs:
                        if not nome_input or not id_sigap_input:
                            st.error("⚠️ Naran no ID SIGAP keta mamuk!")
                        else:
                            input_df = pd.DataFrame([[asid, pont, prod, kual, koop, inis, disi, resp]], columns=nota_cols)
                            pred_encoded = model.predict(input_df)[0]
                            pred_label = le.inverse_transform([pred_encoded])[0]

                            record = {
                                "nome_pessoal": nome_input, "id_sigap": id_sigap_input, "id_grp": id_grp_input,
                                "sexo": sexo_input, "local_trabalho": local_input, "cargo": cargo_input,
                                "Asiduidade": asid, "Pontualidade": pont, "Produtividade": prod,
                                "Kualidade_Servisu": kual, "Kooperasaun": koop, "Inisiativa": inis,
                                "Disiplina": disi, "Responsabilidade": resp, target_col: pred_label
                            }
                            save_extra_to_db(record)
                            st.success(f"✅ Prediksaun ba **{nome_input}**: **{pred_label}**!")
                            st.rerun()

    except Exception as e:
        st.sidebar.error(f"❌ Erru lee ficheiru: {e}")
else:
    st.info("👈 Favor submete (upload) ficheiru Excel iha sidebar atu hahú.")

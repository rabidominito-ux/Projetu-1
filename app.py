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

# Konfigurasaun Pajina
st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - RDTL",
    page_icon="🌳",
    layout="wide",
)

render_custom_css()

def load_data(file):
    return pd.read_excel(file)

# ==========================================
# SISTEMA LOGIN / AUTENTIKASAUN
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); }
        .block-container { padding-top: 3rem !important; max-width: 480px !important; margin: 0 auto !important; }
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
        .cfp-subtitle { text-align: center; color: #64748B; font-size: 12px; margin-bottom: 25px; font-weight: 600; }
        div.stButton > button { background-color: #1E3A8A !important; width: 100%; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <div class="cfp-login-card">
                <div style="text-align: center; font-size: 60px; line-height: 1; margin-bottom: 15px;">🌳</div>
                <div class="cfp-header-title">COMISSÃO DA FUNÇÃO PÚBLICA<br>REPÚBLICA DEMOCRÁTICA DE TIMOR-LESTE</div>
                <div class="cfp-subtitle">Portal de Gestão e Classificação de Desempenho (Decision Tree)</div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username:", placeholder="Hatama ita-nia username")
            password = st.text_input("Password:", type="password", placeholder="Hatama ita-nia password")
            submit_login = st.form_submit_button("ENTRADA / LOGIN")
            
            if submit_login:
                try:
                    if username == st.secrets["username"] and password == st.secrets["password"]:
                        st.session_state["authenticated"] = True
                        st.success("Login susesu!")
                        st.rerun()
                    else:
                        st.error("⚠️ Username ka Password sala!")
                except Exception:
                    # Fallback ba test/demo
                    if username == "admin" and password == "admin123":
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("⚠️ Konfigurasaun Secrets seidauk ihak Streamlit Cloud.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ==========================================
# APLIKASAUN PRINSIPAL (PÓS-LOGIN)
# ==========================================
st.sidebar.markdown("### 🌳 CFP-RDTL Portal")
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout / Sai"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown('<p class="main-title">🌳 Sistema Klasifikasaun Dezempenu CFP - Kapítulu IV</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Intelijénsia Artifisiál Decision Tree ba Analíza, Implementasaun, no Relatóriu Evaluasaun Kapítulu IV.</p>', unsafe_allow_html=True)

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

def generate_pdf_report(df_data, title_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=11, textColor=colors.HexColor('#1E3A8A'), spaceAfter=2, alignment=1, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#4B5563'), spaceAfter=15, alignment=1, fontName='Helvetica')

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

    t = Table(table_data, colWidths=[140, 70, 90, 110, 80])
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

        nota_cols = ["Asiduidade", "Pontualidade", "Produtividade", "Kualidade_Servisu", "Kooperasaun", "Inisiativa", "Disiplina", "Responsabilidade"]
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

            model, le, X_train, X_test, y_train, y_test = treinar_modelo(df, nota_cols, target_col)
            df_filtered["Prediksaun"] = le.inverse_transform(model.predict(df_filtered[nota_cols]))
            acc = accuracy_score(y_test, model.predict(X_test))

            # STRUKTURA TABS ATUALIZADA HO KAPÍTULU IV
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Dashboard Analítiku", 
                "📖 KAPÍTULU IV: Rezultadu & Diskusaun", 
                "⚙️ Performance Modelu", 
                "🔮 Prediksaun & Input Dadus"
            ])

            # ----------------------------------------------------
            # TAB 1: DASHBOARD
            # ----------------------------------------------------
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
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Total Funsionáriu</div><div class="metric-value">{total_funs}</div></div>', unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">⭐ Muito Bom</div><div class="metric-value">{mb_real_pct:.1f}%</div><small>Pred: {mb_pred_pct:.1f}%</small></div>', unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">✨ Bom</div><div class="metric-value">{b_real_pct:.1f}%</div><small>Pred: {b_pred_pct:.1f}%</small></div>', unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">📌 Suficiente</div><div class="metric-value">{s_real_pct:.1f}%</div><small>Pred: {s_pred_pct:.1f}%</small></div>', unsafe_allow_html=True)
                with col_m5:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">⚠️ Insuficiente</div><div class="metric-value">{i_real_pct:.1f}%</div><small>Pred: {i_pred_pct:.1f}%</small></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")
                    if go is not None:
                        categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Bar(name="Dadus Reál", x=categories, y=[counts_real.get(c,0) for c in categories], marker_color="#1E3A8A"))
                        fig_compare.add_trace(go.Bar(name="Prediksaun Tree", x=categories, y=[counts_pred.get(c,0) for c in categories], marker_color="#3B82F6"))
                        fig_compare.update_layout(barmode="group", height=380, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_compare, use_container_width=True)

                with col_g2:
                    st.markdown("##### 🍩 Proporsaun Persentajen (Reál vs Prediksaun)")
                    if go is not None and make_subplots is None:
                        pass
                    else:
                        categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                        fig_donut = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]], subplot_titles=["<b>Reál</b>", "<b>Prediksaun</b>"])
                        fig_donut.add_trace(go.Pie(labels=categories, values=[counts_real.get(c,0) for c in categories], hole=0.4, name="Reál"), 1, 1)
                        fig_donut.add_trace(go.Pie(labels=categories, values=[counts_pred.get(c,0) for c in categories], hole=0.4, name="Prediksaun"), 1, 2)
                        fig_donut.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_donut, use_container_width=True)

            # ----------------------------------------------------
            # TAB 2: KAPÍTULU IV (REZULTADU NO DISKUSAUN)
            # ----------------------------------------------------
            with tab2:
                st.markdown("""
                    <div class="capitulo-box">
                        <h3 style="margin:0; color:#1E3A8A;">📖 KAPÍTULU IV: REZULTADU NO DISKUSAUN</h3>
                        <p style="margin:5px 0 0 0; color:#334155;">Apresentasaun análiza impementasaun modelu algoritmu Decision Tree ba avaliasaun dezempenu funsionáriu públicos iha Comissão da Função Pública (CFP).</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 4.1. Rezultadu Akurasi no Evaluasaun Modelu")
                st.success(f"🎯 **Akurasi Globál Modelu (Accuracy): {acc * 100:.2f}%**")

                y_pred_test = model.predict(X_test)
                unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
                present_class_names = [le.classes_[i] for i in unique_labels]
                report_dict = classification_report(y_test, y_pred_test, labels=unique_labels, target_names=present_class_names, output_dict=True, zero_division=0)
                df_report = pd.DataFrame(report_dict).transpose()

                col_c1, col_c2 = st.columns([1.2, 1])
                with col_c1:
                    st.markdown("**Tabela 4.1: Métrika Evaluasaun Kualidade Klasifikasaun (Precision, Recall, F1-Score)**")
                    st.dataframe(df_report.style.format(subset=["precision", "recall", "f1-score", "support"], formatter="{:.2f}"), use_container_width=True)
                
                with col_c2:
                    st.markdown("**Matris Konfuzaun (Confusion Matrix)**")
                    cm = confusion_matrix(y_test, y_pred_test)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_class_names, yticklabels=present_class_names, ax=ax_cm)
                    plt.ylabel('Valór Ne' + "'" + 'e Duni (Actual)')
                    plt.xlabel('Valór Prediksaun (Predicted)')
                    st.pyplot(fig_cm)

                st.markdown("---")
                st.markdown("#### 4.2. Estrutura Desizaun & Regra Sira (Decision Tree Rules)")
                st.write("Estrutura hirak ne'e mak sai hanesan matadalan desizaun algoritmu nian ba atribuisaun kategoria dezempenu funsionáriu:")
                
                fig_tree_cap, ax_tree_cap = plt.subplots(figsize=(14, 6), dpi=100)
                plot_tree(model, feature_names=nota_cols, class_names=le.classes_, filled=True, rounded=True, ax=ax_tree_cap, fontsize=8)
                st.pyplot(fig_tree_cap)

                st.markdown("---")
                st.markdown("#### 4.3. Diskusaun no Konklusaun Kapítulu IV")
                st.info(f"""
                **Análiza Diskusaun:**
                - Algoritmu Decision Tree konsege halo prediksaun kategoria dezempenu ho akurasi **{acc * 100:.2f}%**.
                - Indikadór sira ne'ebé ho **impaktu boot liu** ba foti desizaun mak kriteria sira hanesan **Produtividade**, **Kualidade Servisu**, no **Asiduidade**.
                - Implementasaun dixital ne'e fo fasilidade ba Komisaun Função Pública (CFP) atu redus tempu halo evaluasaun ho manuál no minimiza erru umannu (*human error*).
                """)

            # ----------------------------------------------------
            # TAB 3: PERFORMANCE MODELU
            # ----------------------------------------------------
            with tab3:
                st.subheader("📋 Amostra Dadus (Preview)")
                st.dataframe(df_filtered.head(10), use_container_width=True)
                st.markdown("---")
                st.subheader("🌳 Vizualizasaun Árbore Desizaun Interativu")
                max_depth_vis = st.slider("Hili Profundidade Árbore (Max Depth)", 1, 5, 3, key="tree_depth_slider")
                vis_model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth_vis, random_state=42)
                vis_model.fit(X_train, y_train)
                fig_tree, ax_tree = plt.subplots(figsize=(16, 8), dpi=100)
                plot_tree(vis_model, feature_names=nota_cols, class_names=le.classes_, filled=True, rounded=True, ax=ax_tree, fontsize=9)
                st.pyplot(fig_tree)

            # ----------------------------------------------------
            # TAB 4: PREDIKSAUN NO INPUT DADUS FOUN
            # ----------------------------------------------------
            with tab4:
                st.subheader("🔍 Prediksaun Funsionáriu Foun & Gestaun Dadus")
                extra_records = load_extra_from_db()
                
                if len(extra_records) > 0:
                    st.markdown("##### 📋 Lista Dadus Funsionáriu Foun Rejisitadu")
                    df_extra_view = pd.DataFrame(extra_records)
                    st.dataframe(df_extra_view[["nome_pessoal", "id_sigap", "local_trabalho", "cargo", "Rezultadu_Avaliasaun"]], use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### ➕ Input Dadus Funsionáriu Foun")
                
                with st.form("funsionariu_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        nome_input = st.text_input("Naran Pessoal:")
                        id_sigap_input = st.text_input("ID SIGAP:")
                        sexo_input = st.selectbox("Sexo:", ["M", "F"])
                        local_input = st.selectbox("Munisípiu:", ["Díli", "Baucau", "Bobonaro", "Ermera", "Liquiçá", "Aileu", "Ainaro", "Covalima", "Manatuto", "Manufahi", "Lautém", "Oe-Cusse Ambeno", "Viqueque"])
                    with c2:
                        id_grp_input = st.text_input("ID GRP:")
                        cargo_input = st.text_input("Kargo:", value="Técnico Superior")
                        funcao_input = st.text_input("Funsaun:", value="Permanente")

                    st.markdown("##### 📊 Nota Indikadór Sira (1.0 - 5.0)")
                    n1, n2, n3, n4 = st.columns(4)
                    with n1:
                        asid = st.number_input("Asiduidade:", 1.0, 5.0, 4.0, step=0.1)
                        pont = st.number_input("Pontualidade:", 1.0, 5.0, 4.0, step=0.1)
                    with n2:
                        prod = st.number_input("Produtividade:", 1.0, 5.0, 4.0, step=0.1)
                        kual = st.number_input("Kualidade Servisu:", 1.0, 5.0, 4.0, step=0.1)
                    with n3:
                        koop = st.number_input("Kooperasaun:", 1.0, 5.0, 4.0, step=0.1)
                        inis = st.number_input("Inisiativa:", 1.0, 5.0, 4.0, step=0.1)
                    with n4:
                        disi = st.number_input("Disiplina:", 1.0, 5.0, 4.0, step=0.1)
                        resp = st.number_input("Responsabilidade:", 1.0, 5.0, 4.0, step=0.1)

                    submit_funs = st.form_submit_button("🔮 Prediz & Rai Dadus")

                    if submit_funs:
                        if not nome_input or not id_sigap_input:
                            st.error("⚠️ Naran no ID SIGAP la bele mamuk!")
                        else:
                            input_df = pd.DataFrame([[asid, pont, prod, kual, koop, inis, disi, resp]], columns=nota_cols)
                            pred_encoded = model.predict(input_df)[0]
                            pred_label = le.inverse_transform([pred_encoded])[0]

                            record = {
                                "controlo_ativo_identificacao": "ATIVO",
                                "nome_pessoal": nome_input, "id_sigap": id_sigap_input, "id_grp": id_grp_input,
                                "sexo": sexo_input, "local_trabalho": local_input, "funcao": funcao_input, "cargo": cargo_input,
                                "Asiduidade": asid, "Pontualidade": pont, "Produtividade": prod, "Kualidade_Servisu": kual,
                                "Kooperasaun": koop, "Inisiativa": inis, "Disiplina": disi, "Responsabilidade": resp,
                                "Rezultadu_Avaliasaun": pred_label
                            }
                            save_extra_to_db(record)
                            st.success(f"✅ Prediksaun ba **{nome_input}**: **{pred_label}**!")
                            st.rerun()

    except Exception as e:
        st.sidebar.error(f"❌ Erru iha procesamentu: {e}")
else:
    st.info("👈 Favor upload ficheiru Excel iha sidebar atu hahú haree análiza Kapítulu IV.")

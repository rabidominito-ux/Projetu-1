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
    page_title="Sistema Klasifikasaun CFP - Kapítulu IV",
    page_icon="🌳",
    layout="wide",
)

render_custom_css()

def load_data(file):
    return pd.read_excel(file)

# Autentikasaun / Login
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
            username = st.text_input("Username:", placeholder="Hatama username")
            password = st.text_input("Password:", type="password", placeholder="Hatama password")
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
                    if username == "admin" and password == "admin123":
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("⚠️ Konfigurasaun Secrets/Credentials seidauk iha.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Main Application Interface
st.sidebar.markdown("### 🌳 CFP-RDTL Portal")
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout / Sai"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown('<p class="main-title">🌳 Sistema Klasifikasaun Dezempenu CFP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Interface ne' + "'" + 'ebé alinha diretu ho Referénsia Kapítulu IV (Rezultadu, Evaluasaun & Diskusaun Decision Tree).</p>', unsafe_allow_html=True)

init_db()

if "extra_reports" not in st.session_state:
    st.session_state["extra_reports"] = load_extra_from_db()

st.sidebar.markdown("### 📁 Upload Dataset")
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

        model, le, X_train, X_test, y_train, y_test = treinar_modelo(df, nota_cols, target_col)
        df["Prediksaun"] = le.inverse_transform(model.predict(df[nota_cols]))
        acc = accuracy_score(y_test, model.predict(X_test))

        # TABS STRUCTURA SINCRO HO KAPÍTULU IV
        tab1, tab2, tab3 = st.tabs([
            "📖 KAPÍTULU IV: Rezultadu & Diskusaun", 
            "📊 Dashboard & Komparasaun Prediksaun", 
            "🔮 Prediksaun Funsionáriu Foun"
        ])

        # ------------------------------------------------------------------
        # TAB 1: KAPÍTULU IV (ESTRUTURA REFERÉNSIA PRINSIPAL)
        # ------------------------------------------------------------------
        with tab1:
            st.markdown("""
                <div style="background-color: #EFF6FF; border-left: 5px solid #1E3A8A; padding: 15px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                    <h3 style="margin:0; color:#1E3A8A;">📖 REFERÉNSIA KAPÍTULU IV: REZULTADU NO DISKUSAUN</h3>
                    <p style="margin:5px 0 0 0; color:#334155;">Seksaun ne'e foti husi dadus analítiku no métrika ne'ebé bazeia ba implementasaun algoritmu Decision Tree.</p>
                </div>
            """, unsafe_allow_html=True)

            # 4.1 Metrics & Confusion Matrix
            st.markdown("#### 4.1. Rezultadu Akurasi no Métrika Evaluasaun")
            
            y_pred_test = model.predict(X_test)
            unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
            present_class_names = [le.classes_[i] for i in unique_labels]
            report_dict = classification_report(y_test, y_pred_test, labels=unique_labels, target_names=present_class_names, output_dict=True, zero_division=0)
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Accuracy</div><div class="metric-value">{acc*100:.1f}%</div></div>', unsafe_allow_html=True)
            with m2:
                macro_prec = report_dict['macro avg']['precision'] * 100
                st.markdown(f'<div class="metric-card"><div class="metric-title">Precision (Avg)</div><div class="metric-value">{macro_prec:.1f}%</div></div>', unsafe_allow_html=True)
            with m3:
                macro_rec = report_dict['macro avg']['recall'] * 100
                st.markdown(f'<div class="metric-card"><div class="metric-title">Recall (Avg)</div><div class="metric-value">{macro_rec:.1f}%</div></div>', unsafe_allow_html=True)
            with m4:
                macro_f1 = report_dict['macro avg']['f1-score'] * 100
                st.markdown(f'<div class="metric-card"><div class="metric-title">F1-Score (Avg)</div><div class="metric-value">{macro_f1:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_tbl, col_cm = st.columns([1.2, 1])

            with col_tbl:
                st.markdown("**Tabela 4.1: Relatóriu Detalladu Klasifikasaun**")
                df_report = pd.DataFrame(report_dict).transpose()
                st.dataframe(df_report.style.format(subset=["precision", "recall", "f1-score", "support"], formatter="{:.2f}"), use_container_width=True)

            with col_cm:
                st.markdown("**Figura 4.1: Confusion Matrix (Matris Konfuzaun)**")
                cm = confusion_matrix(y_test, y_pred_test)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 2.8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_class_names, yticklabels=present_class_names, ax=ax_cm)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig_cm)

            st.markdown("---")

            # 4.2 Decision Tree Visual
            st.markdown("#### 4.2. Estrutura Regra Árbore Desizaun (Decision Tree)")
            fig_tree, ax_tree = plt.subplots(figsize=(15, 6), dpi=100)
            plot_tree(model, feature_names=nota_cols, class_names=le.classes_, filled=True, rounded=True, ax=ax_tree, fontsize=8)
            st.pyplot(fig_tree)

            st.markdown("---")

            # 4.3 Feature Importance (Diskusaun)
            st.markdown("#### 4.3. Diskusaun Indikadór Sira (Feature Importance)")
            importances = model.feature_importances_
            feat_df = pd.DataFrame({"Indikadór": nota_cols, "Importánsia": importances}).sort_values(by="Importánsia", ascending=True)

            fig_feat, ax_feat = plt.subplots(figsize=(8, 3.5))
            ax_feat.barh(feat_df["Indikadór"], feat_df["Importánsia"], color="#1E3A8A")
            ax_feat.set_xlabel("Nível Importánsia")
            st.pyplot(fig_feat)

        # ------------------------------------------------------------------
        # TAB 2: DASHBOARD GENERAL
        # ------------------------------------------------------------------
        with tab2:
            st.markdown("### 📊 Visao Geral & Komparasaun")
            counts_real = df[target_col].value_counts()
            counts_pred = df["Prediksaun"].value_counts()

            if go is not None:
                categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(name="Dadus Reál", x=categories, y=[counts_real.get(c,0) for c in categories], marker_color="#1E3A8A"))
                fig_compare.add_trace(go.Bar(name="Prediksaun Tree", x=categories, y=[counts_pred.get(c,0) for c in categories], marker_color="#3B82F6"))
                fig_compare.update_layout(barmode="group", height=400)
                st.plotly_chart(fig_compare, use_container_width=True)

        # ------------------------------------------------------------------
        # TAB 3: PREDIKSAUN DADUS FOUN
        # ------------------------------------------------------------------
        with tab3:
            st.markdown("### 🔮 Prediksaun Dezempenu Funsionáriu")
            with st.form("form_funsionariu_foun"):
                c1, c2 = st.columns(2)
                with c1:
                    nome_in = st.text_input("Naran Pessoal:")
                    sigap_in = st.text_input("ID SIGAP:")
                    sexo_in = st.selectbox("Sexo:", ["M", "F"])
                    local_in = st.text_input("Munisípiu / Local:", "Díli")
                with c2:
                    grp_in = st.text_input("ID GRP:")
                    cargo_in = st.text_input("Kargo:", "Técnico Superior")
                    funcao_in = st.text_input("Funsaun:", "Permanente")

                st.markdown("##### 📊 Notas Indikadór (1.0 - 5.0)")
                n1, n2, n3, n4 = st.columns(4)
                with n1:
                    asid = st.number_input("Asiduidade:", 1.0, 5.0, 4.0)
                    pont = st.number_input("Pontualidade:", 1.0, 5.0, 4.0)
                with n2:
                    prod = st.number_input("Produtividade:", 1.0, 5.0, 4.0)
                    kual = st.number_input("Kualidade Servisu:", 1.0, 5.0, 4.0)
                with n3:
                    koop = st.number_input("Kooperasaun:", 1.0, 5.0, 4.0)
                    inis = st.number_input("Inisiativa:", 1.0, 5.0, 4.0)
                with n4:
                    disi = st.number_input("Disiplina:", 1.0, 5.0, 4.0)
                    resp = st.number_input("Responsabilidade:", 1.0, 5.0, 4.0)

                sub_btn = st.form_submit_button("🔮 Kalkula & Rai Prediksaun")

                if sub_btn:
                    input_data = pd.DataFrame([[asid, pont, prod, kual, koop, inis, disi, resp]], columns=nota_cols)
                    pred_val = le.inverse_transform(model.predict(input_data))[0]
                    
                    rec = {
                        "controlo_ativo_identificacao": "ATIVO", "nome_pessoal": nome_in, "id_sigap": sigap_in,
                        "id_grp": grp_in, "sexo": sexo_in, "local_trabalho": local_in, "funcao": funcao_in,
                        "cargo": cargo_in, "Asiduidade": asid, "Pontualidade": pont, "Produtividade": prod,
                        "Kualidade_Servisu": kual, "Kooperasaun": koop, "Inisiativa": inis, "Disiplina": disi,
                        "Responsabilidade": resp, "Rezultadu_Avaliasaun": pred_val
                    }
                    save_extra_to_db(rec)
                    st.success(f"✅ Prediksaun ba {nome_in}: **{pred_val}**")
                    st.rerun()

    except Exception as err:
        st.sidebar.error(f"❌ Erru procesa dadus: {err}")
else:
    st.info("👈 Favor upload ficheiru Excel (.xlsx) iha sidebar atu hahú haree referénsia Kapítulu IV.")

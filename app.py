import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Konfigurasaun Pajina Streamlit (Layout Wide)
st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - Decision Tree",
    page_icon="📊",
    layout="wide"
)

# Custom CSS ba UI ne'ebé modernu no kapás
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasaun uza algoritmu Decision Tree hodi klasifika dezempenu funsionáriu bazeia ba indikadór Komisaun Função Pública (CFP).</p>', unsafe_allow_html=True)

# 2. Sidebar ba Upload Dataset
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload ficheiru Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df_raw = pd.read_excel(file, sheet_name='Sheet1', header=0)
        return df_raw

    df_raw = load_data(uploaded_file)
    
    # Remapa koluna sira
    rename_map = {
        'Column1': 'controlo_ativo_identificacao',
        'Column2': 'nome_pessoal',
        'Column3': 'id_sigap',
        'Column4': 'id_grp',
        'Column5': 'sexo',
        'Column6': 'data_de_nascimento',
        'Column7': 'instituicao',
        'Column8': 'local_trabalho',
        'Column9': 'funcao',
        'Column10': 'cargo',
        'Column11': 'data_fim_nao_exercicio',
        'Column12': 'temp1',
        'Column13': 'Asiduidade',
        'Column14': 'Pontualidade',
        'Column15': 'Produtividade',
        'Column16': 'Kualidade_Servisu',
        'Column17': 'Kooperasaun',
        'Column18': 'Inisiativa',
        'Column19': 'Disiplina',
        'Column20': 'Responsabilidade',
        'Column21': 'Media',
        'Column22': 'Rezultadu_Avaliasaun',
        'Column23': 'temp2'
    }
    
    df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)

    nota_cols = ['Asiduidade', 'Pontualidade', 'Produtividade', 'Kualidade_Servisu',
                 'Kooperasaun', 'Inisiativa', 'Disiplina', 'Responsabilidade']
    target_col = 'Rezultadu_Avaliasaun'

    if all(col in df_raw.columns for col in nota_cols + [target_col]):
        df = df_raw.dropna(subset=nota_cols + [target_col]).copy()
        for col in nota_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Treinu Modelu Automátiku ka bazeia ba Cache/Session State
        le = LabelEncoder()
        df['target_encoded'] = le.fit_transform(df[target_col])
        y = df['target_encoded']
        X = df[nota_cols].copy()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        df['Prediksaun'] = le.inverse_transform(model.predict(X))
        acc = accuracy_score(y_test, model.predict(X_test))

        # 3. Organizasaun Interface uza Tabs (Fasilita Navegasaun)
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Sumáriu", "⚙️ Preview & Treinu Modelu", "🔮 Prediksaun Funsionáriu Foun"])

        with tab1:
            st.subheader("📈 Estatistika & Sumáriu Kategoria Dezempenu")
            
            # KPI Metrics Cards
            total_funs = len(df)
            counts_real = df[target_col].value_counts()
            
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Total Funsionáriu", total_funs)
            col_m2.metric("Muito Bom", counts_real.get('Muito Bom', 0))
            col_m3.metric("Bom", counts_real.get('Bom', 0))
            col_m4.metric("Suficiente", counts_real.get('Suficiente', 0))
            col_m5.metric("Insuficiente", counts_real.get('Insuficiente', 0))
            
            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("##### 📁 Distribuisaun Dadus Reál (Excel)")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.countplot(data=df, x=target_col, order=['Muito Bom', 'Bom', 'Suficiente', 'Insuficiente'], palette='Blues', ax=ax)
                plt.title("Dadus Reál CFP")
                st.pyplot(fig)
            with col_g2:
                st.markdown("##### 🤖 Distribuisaun Prediksaun (Decision Tree)")
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                sns.countplot(data=df, x='Prediksaun', order=['Muito Bom', 'Bom', 'Suficiente', 'Insuficiente'], palette='Purples', ax=ax2)
                plt.title("Prediksaun Algoritmu")
                st.pyplot(fig2)

        with tab2:
            st.subheader("📋 Dadus Amostra (Preview)")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🚀 Informasaun Modelu Decision Tree")
            st.success(f"✅ Modelu treinu ho suksesu! Akurasi Modelu (Accuracy): **{acc * 100:.2f}%**")
            st.info("Algoritmu uza kriteria `Entropy` hodi kalkula Information Gain husi indikadór 8 avaliasaun nian.")

        with tab3:
            st.subheader("🔍 Halo Prediksaun ba Funsionáriu Unidade Foun")
            
            with st.form("funsionariu_form"):
                st.markdown("##### 📝 1. Informasaun Identidade Funsionáriu")
                col_i1, col_i2, col_i3 = st.columns(3)
                with col_i1:
                    txt_nome = st.text_input("Naran Pessoal (nome_pessoal)", "Ex: João da Silva")
                    txt_sigap = st.text_input("ID SIGAP (id_sigap)", "SIGAP-001")
                    txt_sexo = st.selectbox("Sexo", ["M", "F"])
                with col_i2:
                    txt_inst = st.text_input("Instituisaun", "CFP")
                    txt_local = st.text_input("Local Trabalhu", "Dili")
                    txt_nascimento = st.text_input("Data de Nascimento", "1995-01-01")
                with col_i3:
                    txt_funcao = st.text_input("Funsaun", "Tékniku")
                    txt_cargo = st.text_input("Kargo", "Staff")
                    txt_grp = st.text_input("ID GRP", "GRP-123")

                st.markdown("##### 📊 2. Indikadór Avaliasaun Funsionáriu (Skala 1 - 5)")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    p_asid = st.slider("Asiduidade", 1.0, 5.0, 4.0, 1.0)
                    p_pont = st.slider("Pontualidade", 1.0, 5.0, 4.0, 1.0)
                with col_b:
                    p_prod = st.slider("Produtividade", 1.0, 5.0, 4.0, 1.0)
                    p_kual = st.slider("Kualidade Servisu", 1.0, 5.0, 4.0, 1.0)
                with col_c:
                    p_koop = st.slider("Kooperasaun", 1.0, 5.0, 4.0, 1.0)
                    p_inis = st.slider("Inisiativa", 1.0, 5.0, 4.0, 1.0)
                with col_d:
                    p_disp = st.slider("Disiplina", 1.0, 5.0, 4.0, 1.0)
                    p_resp = st.slider("Responsabilidade", 1.0, 5.0, 4.0, 1.0)
                
                submit_pred = st.form_submit_button("🔮 Predict & Hare Relatóriu")
                
                if submit_pred:
                    input_data = np.array([[p_asid, p_pont, p_prod, p_kual, p_koop, p_inis, p_disp, p_resp]])
                    pred_encoded = model.predict(input_data)
                    pred_label = le.inverse_transform(pred_encoded)[0]
                    
                    st.session_state['last_report'] = {
                        'nome_pessoal': txt_nome,
                        'id_sigap': txt_sigap,
                        'sexo': txt_sexo,
                        'funcao': txt_funcao,
                        'cargo': txt_cargo,
                        'local_trabalho': txt_local,
                        'Asiduidade': p_asid,
                        'Pontualidade': p_pont,
                        'Produtividade': p_prod,
                        'Kualidade_Servisu': p_kual,
                        'Kooperasaun': p_koop,
                        'Inisiativa': p_inis,
                        'Disiplina': p_disp,
                        'Responsabilidade': p_resp,
                        'Rezultadu_Prediksaun': pred_label
                    }

            # Relatóriu no Download Button
            if 'last_report' in st.session_state:
                rep = st.session_state['last_report']
                st.markdown("---")
                st.subheader("📄 Relatóriu Rezultadu Avaliasaun Funsionáriu Foun")
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown(f"**Naran Pessoal:** {rep['nome_pessoal']}")
                    st.markdown(f"**ID SIGAP:** {rep['id_sigap']}")
                    st.markdown(f"**Sexo:** {rep['sexo']}")
                    st.markdown(f"**Funsaun:** {rep['funcao']}")
                with col_r2:
                    st.markdown(f"**Kargo:** {rep['cargo']}")
                    st.markdown(f"**Fatin Trabalhu:** {rep['local_trabalho']}")
                    st.markdown(f"✨ **Rezultadu Klasifikasaun:** `{rep['Rezultadu_Prediksaun']}`")
                
                # Botaun Download CSV
                rep_df = pd.DataFrame([rep])
                csv_data = rep_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Relatóriu (CSV)",
                    data=csv_data,
                    file_name=f"relatorio_{rep['id_sigap']}.csv",
                    mime='text/csv'
                )
else:
    st.info("👈 Favor upload uluk ficheiru Excel (`.xlsx`) iha sidebar sorin karuk hodi hahú sistema.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Konfigurasaun Pajina Streamlit
st.set_page_config(page_title="Sistema Klasifikasaun CFP - Decision Tree", page_icon="📊", layout="wide")

st.title("📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP")
st.markdown("Aplikasaun ne'e uza algoritmu **Decision Tree** hodi klasifika dezempenu funsionáriu bazeia ba indikadór avaliasaun iha Komisaun Função Pública (CFP).")

# Sidebar ba Upload Ficheiru
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

        st.subheader("📋 Dadus Amostra (Preview)")
        st.dataframe(df.head())

        # Botão Treinu Modelu
        if st.button("🚀 Prosesa no Treinu Modelu (Train Model)"):
            with st.spinner("Modelu dada hela treinu..."):
                le = LabelEncoder()
                df['target_encoded'] = le.fit_transform(df[target_col])
                y = df['target_encoded']
                X = df[nota_cols].copy()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ Modelu treinu ho susesu! Akurasi: {acc * 100:.2f}%")

        # Formuláriu Identidade no Prediksaun Funsionáriu Foun
        st.markdown("---")
        st.subheader("🔍 Halo Prediksaun no Input Identidade Funsionáriu Foun")
        
        with st.form("funsionariu_form"):
            st.markdown("##### 📝 Informasaun Identidade Funsionáriu")
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                txt_nome = st.text_input("Naran Pessoal (nome_pessoal)", "Ex: João da Silva")
                txt_sigap = st.text_input("ID SIGAP (id_sigap)", "SIGAP-001")
                txt_sexo = st.selectbox("Sexo", ["M", "F"])
            with col_i2:
                txt_inst = st.text_input("Instituisaun", "KFP")
                txt_local = st.text_input("Local Trabalhu", "Dili")
                txt_nascimento = st.text_input("Data de Nascimento", "1995-01-01")
            with col_i3:
                txt_funcao = st.text_input("Funsaun", "Tékniku")
                txt_cargo = st.text_input("Kargo", "Staff")
                txt_grp = st.text_input("ID GRP", "GRP-123")

            st.markdown("##### 📊 Indikadór Avaliasaun Funsionáriu (Skala 1 - 5)")
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
                le = LabelEncoder()
                df['target_encoded'] = le.fit_transform(df[target_col])
                X = df[nota_cols].copy()
                y = df['target_encoded']
                
                model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
                model.fit(X, y)
                
                input_data = np.array([[p_asid, p_pont, p_prod, p_kual, p_koop, p_inis, p_disp, p_resp]])
                pred_encoded = model.predict(input_data)
                pred_label = le.inverse_transform(pred_encoded)[0]
                
                # Rai iha session_state atu bele hatudu relatóriu
                st.session_state['last_report'] = {
                    'nome': txt_nome,
                    'sigap': txt_sigap,
                    'sexo': txt_sexo,
                    'funcao': txt_funcao,
                    'cargo': txt_cargo,
                    'local': txt_local,
                    'asid': p_asid,
                    'pont': p_pont,
                    'prod': p_prod,
                    'kual': p_kual,
                    'koop': p_koop,
                    'inis': p_inis,
                    'disp': p_disp,
                    'resp': p_resp,
                    'result': pred_label
                }

        # Seksaun Relatóriu Funsionáriu Foun ne'ebé foin hatama
        if 'last_report' in st.session_state:
            rep = st.session_state['last_report']
            st.markdown("---")
            st.subheader("📄 Relatóriu Rezultadu Avaliasaun Funsionáriu Foun")
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"**Naran Pessoal:** {rep['nome']}")
                st.markdown(f"**ID SIGAP:** {rep['sigap']}")
                st.markdown(f"**Sexo:** {rep['sexo']}")
                st.markdown(f"**Funsaun:** {rep['funcao']}")
            with col_r2:
                st.markdown(f"**Kargo:** {rep['cargo']}")
                st.markdown(f"**Fatin Trabalhu:** {rep['local']}")
                st.markdown(f"✨ **Rezultadu Klasifikasaun:** `{rep['result']}`")

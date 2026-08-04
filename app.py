import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

        # Botão Treinu Modelu no Kontajen Kategoria
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
                
                # Prediksaun ba dataset tomak hodi hatudu total
                df['Prediksaun'] = le.inverse_transform(model.predict(X))
                
                y_pred_test = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred_test)

                st.success(f"✅ Modelu treinu ho susesu! Akurasi Modelu: {acc * 100:.2f}%")

                # Seksaun Hatudu Total Funsionáriu tuir Kategoria (Reál vs Prediksaun)
                st.markdown("---")
                st.subheader("📊 Sumáriu Total Funsionáriu tuir Kategoria Avaliasaun")
                
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.markdown("##### 📁 Dadus Reál (Iha Ficheiru Excel)")
                    counts_real = df[target_col].value_counts()
                    st.dataframe(counts_real)

                with col_sum2:
                    st.markdown("##### 🤖 Rezultadu Prediksaun (Decision Tree)")
                    counts_pred = df['Prediksaun'].value_counts()
                    st.dataframe(counts_pred)

                # Gráfiku Bar Chart
                st.markdown("##### 📈 Gráfiku Distribuisaun Kategoria Dezempenu")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df, x='Prediksaun', order=['Muito Bom', 'Bom', 'Suficiente', 'Insuficiente'], palette='viridis', ax=ax)
                plt.title("Total Funsionáriu tuir Prediksaun Decision Tree")
                plt.xlabel("Kategoria Rezultadu Avaliasaun")
                plt.ylabel("Total Funsionáriu")
                st.pyplot(fig)

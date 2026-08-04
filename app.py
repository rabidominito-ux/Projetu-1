import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Konfigurasaun Pajina Streamlit
st.set_page_config(page_title="Sistema Klasifikasaun CFP - Decision Tree", page_icon="📊", layout="wide")

st.title("📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP")
st.markdown("Aplikasaun ne'e uza algoritmu **Decision Tree** hodi klasifika dezempenu funsionáriu bazeia ba indikadór avaliasaun iha Komisaun Função Pública (CFP).")

# Sidebar ba Upload Ficheiru
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload ficheiru Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Karga Dadus
    @st.cache_data
    def load_data(file):
        df_raw = pd.read_excel(file, sheet_name='Sheet1', header=0)
        return df_raw

    df_raw = load_data(uploaded_file)
    
    # Remapa koluna sira tuir kódigu Colab
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
    
    # Asegura katak koluna sira iha baze de dadus
    df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)

    nota_cols = ['Asiduidade', 'Pontualidade', 'Produtividade', 'Kualidade_Servisu',
                 'Kooperasaun', 'Inisiativa', 'Disiplina', 'Responsabilidade']
    target_col = 'Rezultadu_Avaliasaun'

    if all(col in df_raw.columns for col in nota_cols + [target_col]):
        df = df_raw[nota_cols + [target_col]].copy()
        for col in nota_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        st.subheader("📋 Dadus Amostra (Preview)")
        st.dataframe(df.head())

        # Botão 1: Treinu no Evaluasaun Modelu
        if st.button("🚀 Prosesa no Treinu Modelu (Train Model)"):
            with st.spinner("Modelu dada hela treinu... Favor hein minutu uitoan."):
                le = LabelEncoder()
                df['target_encoded'] = le.fit_transform(df[target_col])
                y = df['target_encoded']
                X = df[nota_cols].copy()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # GridSearch ba Hyperparameter Tuning
                param_grid = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['gini', 'entropy']
                }
                grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)
                
                dt_best = grid.best_estimator_
                y_pred_best = dt_best.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred_best)
                cv_scores = cross_val_score(dt_best, X_train, y_train, cv=5)

                st.success("✅ Modelu treinu ho susesu!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Akurasi (Test Accuracy)", f"{acc * 100:.2f}%")
                col2.metric("Cross-Validation Média", f"{cv_scores.mean() * 100:.2f}%")
                col3.metric("Parametru Di'ak liu", str(grid.best_params_))

                # Visualizasaun Confusion Matrix
                st.subheader("📊 Matriz Konfuzaun (Confusion Matrix)")
                cm = confusion_matrix(y_test, y_pred_best)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                plt.xlabel("Prediksaun")
                plt.ylabel("Reál")
                st.pyplot(fig)

        # Botão 2 / Formuláriu Prediksaun Foun ba Funsionáriu
        st.markdown("---")
        st.subheader("🔍 Halo Prediksaun ba Funsionáriu Unidade Foun")
        
        with st.form("pred_form"):
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
            
            submit_pred = st.form_submit_button("🔮 Predict / Prediksa Rezultadu")
            
            if submit_pred:
                le = LabelEncoder()
                df['target_encoded'] = le.fit_transform(df[target_col])
                X = df[nota_cols].copy()
                y = df['target_encoded']
                
                model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, random_state=42)
                model.fit(X, y)
                
                input_data = np.array([[p_asid, p_pont, p_prod, p_kual, p_koop, p_inis, p_disp, p_resp]])
                pred_encoded = model.predict(input_data)
                pred_label = le.inverse_transform(pred_encoded)[0]
                
                st.info(f"✨ Rezultadu Klasifikasaun Dezempenu Funsionáriu: **{pred_label}**")
    else:
        st.error("⚠️ Koluna sira iha ficheiru Excel la hanesan ho estrutura ne'ebé espere (Keta haluha koluna nota no rezultadu avaliasaun).")
else:
    st.info("👈 Favor upload ficheiru `Dadus CFP.xlsx` liuhusi sidebar iha sorin karuk hಿತು hahú.")

import streamlit as st
import pandas as pd
import joblib

# Karga modelu treinadu
modelu = joblib.load('modelu_cfp.pkl')

# Koluna nota sira
nota_cols = ['Asiduidade','Pontualidade','Produtividade','Kualidade_Servisu',
             'Kooperasaun','Inisiativa','Disiplina','Responsabilidade']

st.title("🌐 Avaliasaun Funsionáriu - Hugging Face Space")

st.write("Prense valor ba kriteriu sira, klik botão atu hetan predisaun.")

# Input ho slider
inputs = {}
for col in nota_cols:
    inputs[col] = st.slider(col, 0, 10, 5)

# Predisaun
X_novo = pd.DataFrame([inputs])
if st.button("Prediz Rezultadu"):
    y_pred = modelu.predict(X_novo)
    st.success(f"Rezultadu Avaliasaun: {y_pred[0]}")

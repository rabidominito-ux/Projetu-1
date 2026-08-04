import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Konfiguratuan Pajina
st.set_page_config(
    page_title="Avaliasaun Desempenhu Funsionariu - CFP",
    page_icon="📊",
    layout="centered",
)

# Titulu Aplikasaun
st.title("💼 Sistema Avaliasaun Desempenhu Funsionariu")
st.write(
    "Komisaun Funsaun Publika (CFP) - Uza Algoritma Decision Tree atu foti desizaun avaliasaun."
)
st.markdown("---")

# Nota: Karik Ita salva ona imi nia modelu ne'ebé treinu ona ho joblib.dump(model, 'model_decision_tree.pkl')
# Ita bele karga fali uza:
# @st.cache_resource
# def load_model():
#     return joblib.load('model_decision_tree.pkl')
# model = load_model()

# Sidebar ba Input Dadus Funsionariu
st.sidebar.header("📝 Hatama Dadus Funsionariu")


def user_input_features():
    # Bainhira iha kriteria espesífiku sira husi imi nia dataset, bele adapta iha ne'e:
    asiduidade = st.sidebar.slider("Asiduidade / Prezensa (0 - 100)", 0, 100, 85)
    pontualidade = st.sidebar.slider("Pontualidade (0 - 100)", 0, 100, 90)
    produtividade = st.sidebar.slider(
        "Produtividade Servisu (0 - 100)", 0, 100, 80
    )
    kualidade = st.sidebar.slider("Kualidade Servisu (0 - 100)", 0, 100, 85)
    disiplina = st.sidebar.selectbox(
        "Disiplina", ["Di'ak Teves", "Di'ak", "Presiza Mellora"]
    )
    kooperasaun = st.sidebar.selectbox(
        "Kooperasaun iha Tim", ["Di'ak Teves", "Di'ak", "Presiza Mellora"]
    )

    # Konverte dadus kategoriál ba numeru (tuir kodigu enkodiamentu orijen nian)
    disiplina_map = {"Presiza Mellora": 0, "Di'ak": 1, "Di'ak Teves": 2}
    kooperasaun_map = {"Presiza Mellora": 0, "Di'ak": 1, "Di'ak Teves": 2}

    data = {
        "Asiduidade": asiduidade,
        "Pontualidade": pontualidade,
        "Produtividade": produtividade,
        "Kualidade": kualidade,
        "Disiplina": disiplina_map[disiplina],
        "Kooperasaun": kooperasaun_map[kooperasaun],
    }
    features = pd.DataFrame(data, index=[0])
    return features


df_input = user_input_features()

# Display dadus ne'ebé uzuáriu hatama ona
st.subheader("📊 Dadus ne'ebé hatama ona:")
st.write(df_input)

# Botaun Prediksaun
if st.button("🔍 Halo Avaliasaun / Prediksaun"):
    # Hanesan ezemplu, ita simula prediksaun ka uza modelu ne'ebé karga ona:
    # prediction = model.predict(df_input)

    # Ezemplu simulasuan logika bazeia ba media kriteria:
    media_score = (
        df_input["Asiduidade"].values[0]
        + df_input["Pontualidade"].values[0]
        + df_input["Produtividade"].values[0]
        + df_input["Kualidade"].values[0]
    ) / 4

    if media_score >= 85:
        hasil = "Muito Bom (Di'ak Tebes)"
        color = "green"
    elif media_score >= 70:
        hasil = "Bom (Di'ak)"
        color = "blue"
    elif media_score >= 55:
        hasil = "Suficiente (Sufisiente)"
        color = "orange"
    else:
        hasil = "Insuficiente (Kadiak)"
        color = "red"

    st.markdown("---")
    st.subheader("🎯 Rezultadu Avaliasaun Desempenhu:")
    st.markdown(
        f"<h3 style='color: {color};'>Kategoria: {hasil}</h3>",
        unsafe_allow_html=True,
    )
    st.info(
        f"Media pontu ba kriteria kuantitativu mak: **{media_score:.2f}%**"
    )

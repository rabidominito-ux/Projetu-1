import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Konfiguratuan Pajina
st.set_page_config(
    page_title="Sistema Avaliasaun Desempenhu - CFP",
    page_icon="🏛️",
    layout="wide",
)

# Karga Dataset CFP
@st.cache_data
def load_data():
    df = pd.read_excel('Dadus CFP.xlsx', header=1)
    # Kria model / train decision tree kedas husi dadus atu halo prediksaun real
    features = [
        'Asiduidade',
        'Pontualidade',
        'Produtividade',
        'Kualidade_Servisu',
        'Kooperasaun',
        'Inisiativa',
        'Disiplina',
        'Responsabilidade',
    ]
    X = df[features]
    y = df['Rezultadu_Avaliasaun']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return df, model, features


df, model, features = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Menu Navigasaun")
menu = st.sidebar.selectbox(
    "Hili Pajina:",
    [
        "🏠 Dashboard & Análize Dadus",
        "🔍 Prediksaun Funsionáriu Foun",
        "📋 Lista Funsionáriu & Filtru",
    ],
)

st.sidebar.markdown('---')
st.sidebar.info(
    'Sistema Inteligente Uza Algoritma Decision Tree ba Komisaun Funsaun Públika (CFP).'
)

# ==========================================
# 1. DASHBOARD & ANÁLIZE DADUS
# ==========================================
if menu == '🏠 Dashboard & Análize Dadus':
  st.title('📊 Dashboard Análize Desempenhu Funsionáriu CFP')
  st.write(
     'Visaun jerál ba dadus avaliasaun desempenhu funsionáriu sira nian bazeia ba'
     ' kriteria kuantitativu no kualitativu.'
  )

  # Metrika xave
  col1, col2, col3, col4 = st.columns(4)
  col1.metric('Total Funsionáriu', len(df))
  col2.metric('Média Jerál Pontu', f"{df['Media'].mean():.2f} / 5.0")
  col3.metric(
      'Funsionáriu Muito Bom',
      len(df[df['Rezultadu_Avaliasaun'] == 'Muito Bom']),
  )
  col4.metric(
      'Funsionáriu Insuficiente',
      len(df[df['Rezultadu_Avaliasaun'] == 'Insuficiente']),
  )

  st.markdown('---')

  c1, c2 = st.columns(2)

  with c1:
    st.subheader('📈 Distribuisaun Kategoria Rezultadu Avaliasaun')
    rez_counts = df['Rezultadu_Avaliasaun'].value_counts()
    st.bar_chart(rez_counts)

  with c2:
    st.subheader('👥 Distribuisaun tuir Kargu (Cargo)')
    cargo_counts = df['cargo'].value_counts()
    st.bar_chart(cargo_counts)

  st.markdown('---')
  st.subheader('📉 Média Kriteria Avaliasaun Hotu-Hotu')
  kriteria_mean = df[features].mean()
  st.bar_chart(kriteria_mean)

# ==========================================
# 2. PREDIPSAUN FUNSIÓNARIU FOUN
# ==========================================
elif menu == '🔍 Prediksaun Funsionáriu Foun':
  st.title('🔍 Simula Prediksaun Desempenhu Funsionáriu')
  st.write(
      'Hatama pontu husi kriteria oioin iha sorin (sidebar) atu hatene kategoria'
      ' desizaun husi Decision Tree.'
  )

  st.sidebar.subheader('🎛️ Ajusta Pontu Kriteria (Skala 1 - 5)')
  asiduidade = st.sidebar.slider('Asiduidade', 1, 5, 4)
  pontualidade = st.sidebar.slider('Pontualidade', 1, 5, 4)
  produtividade = st.sidebar.slider('Produtividade', 1, 5, 4)
  kualidade_servisu = st.sidebar.slider('Kualidade Servisu', 1, 5, 4)
  kooperasaun = st.sidebar.slider('Kooperasaun', 1, 5, 4)
  inisiativa = st.sidebar.slider('Inisiativa', 1, 5, 4)
  disiplina = st.sidebar.slider('Disiplina', 1, 5, 4)
  responsabilidade = st.sidebar.slider('Responsabilidade', 1, 5, 4)

  # DataFrame ba input
  input_data = pd.DataFrame(
      {
          'Asiduidade': [asiduidade],
          'Pontualidade': [pontualidade],
          'Produtividade': [produtividade],
          'Kualidade_Servisu': [kualidade_servisu],
          'Kooperasaun': [kooperasaun],
          'Inisiativa': [inisiativa],
          'Disiplina': [disiplina],
          'Responsabilidade': [responsabilidade],
      }
  )

  st.subheader('📝 Dadus Kriteria Ne\'ebé Hili:')
  st.dataframe(input_data, use_container_width=True)

  if st.button('🚀 Halo Prediksaun Modelu', type='primary'):
    prediction = model.predict(input_data)[0]
    media_input = input_data.mean(axis=1)[0]

    st.markdown('---')
    st.subheader('🎯 Rezultadu Prediksaun Decision Tree:')

    if prediction == 'Muito Bom':
      st.success(
          f'🌟 Kategoria: **{prediction}** (Média Pontu:'
          f' {media_input:.2f}/5.0)'
      )
    elif prediction == 'Bom':
      st.info(
          f'👍 Kategoria: **{prediction}** (Média Pontu:'
          f' {media_input:.2f}/5.0)'
      )
    elif prediction == 'Suficiente':
      st.warning(
          f'⚠️ Kategoria: **{prediction}** (Média Pontu:'
          f' {media_input:.2f}/5.0)'
      )
    else:
      st.error(
          f'❌ Kategoria: **{prediction}** (Média Pontu:'
          f' {media_input:.2f}/5.0 - Presiza Mellora)'
      )

# ==========================================
# 3. LISTA FUNSIÓNARIU & FILTRU
# ==========================================
elif menu == '📋 Lista Funsionáriu & Filtru':
  st.title('📋 Dadus Detalhadu Funsionáriu CFP')
  st.write('Buscador no filtru ba dadus funsionáriu sira iha sistema.')

  # Filtru bazeia ba Kategoria Rezultadu
  selected_cat = st.selectbox(
      'Filtru tuir Kategoria Rezultadu Avaliasaun:',
      ['Hotu-Hotu'] + list(df['Rezultadu_Avaliasaun'].unique()),
  )

  if selected_cat != 'Hotu-Hotu':
    filtered_df = df[df['Rezultadu_Avaliasaun'] == selected_cat]
  else:
    filtered_df = df

  # Peskiza naran
  search_name = st.text_input('🔍 Peskiza Naran Funsionáriu:')
  if search_name:
    filtered_df = filtered_df[
        filtered_df['nome_pessoal'].str.contains(search_name, case=False, na=False)
    ]

  st.write(f'Hatudu **{len(filtered_df)}** rejistu funsionáriu:')
  st.dataframe(
      filtered_df[
          [
              'nome_pessoal',
              'cargo',
              'sexo',
              'Media',
              'Rezultadu_Avaliasaun',
          ]
      ],
      use_container_width=True,
  )

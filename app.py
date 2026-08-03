import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 1. KONFIGURASAUN PÁJINA
# ============================================================
st.set_page_config(page_title="Avaliasaun Funzionáriu", layout="centered")

# ============================================================
# 2. DADUS LOGIN (SIMPLES)
# ============================================================
# Iha prototipu, armazena password iha kódigu.
# Iha produsaun, uza hash (bcrypt) ka autentikasaun externa.
USERS = {
    "admin": "admin123",
    "funcionario": "123456",
    "gestor": "gestor2025"
}

# ============================================================
# 3. FUNSAUN LOGIN
# ============================================================
def login():
    st.markdown("## 🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Entra")

    if submit:
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("✅ Login ho suksesu! Redireciona...")
            st.rerun()
        else:
            st.error("❌ Username ka password sala!")

    # Avisu ba utilizadór
    st.caption("Utilizadór disponível: admin, funcionario, gestor")

# ============================================================
# 4. FUNSAUN LOGOUT
# ============================================================
def logout():
    if st.sidebar.button("Sai (Logout)"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.rerun()

# ============================================================
# 5. APP PRINSIPÁL (SÓ DEPOIS DE LOGIN)
# ============================================================
def main_app():
    # Sidbar ho informasaun utilizadór no botão logout
    st.sidebar.markdown(f"### 👤 Utilizadór: `{st.session_state['username']}`")
    logout()

    # Títulu
    st.title("📊 Avaliasaun Funzionáriu - Hugging Face Space")
    st.markdown("Prense valor ba kriteriu sira, klik botão atu hetan predisaun.")

    # Karega modelu
    try:
        modelu = joblib.load("modelu_cfp.pkl")
    except FileNotFoundError:
        st.error("Ficheiru modelu 'modelu_cfp.pkl' la hetan! Haree se ita karega ona.")
        st.stop()

    # Definisaun koluna sira
    nota_cols = [
        "Asiduidade",
        "Pontualidade",
        "Produtividade",
        "Kualidade_Servisu",
        "Kooperasaun",
        "Inisiativa",
        "Disiplina",
        "Responsabilidade"
    ]

    # Kria slider sira
    inputs = {}
    for col in nota_cols:
        inputs[col] = st.slider(col, 0, 10, 5)

    # Botão predisaun
    if st.button("Prediz Rezultadu"):
        X_novo = pd.DataFrame([inputs])
        y_pred = modelu.predict(X_novo)
        nota = y_pred[0]

        # Mapeia ba kategoria
        if nota >= 4.1:
            kategoria = "🌟 Muito Bom"
        elif nota >= 3.5:
            kategoria = "✅ Bom"
        elif nota >= 2.5:
            kategoria = "📘 Suficiente"
        else:
            kategoria = "❌ Insuficiente"

        st.success(f"Rezultadu Avaliasaun: **{kategoria}** (Nota: {nota:.2f})")

# ============================================================
# 6. KONTROLU FLUXU (LOGIN vs APP)
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None

if st.session_state["logged_in"]:
    main_app()
else:
    login()

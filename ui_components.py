import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def exibir_sidebar():
    st.sidebar.markdown("### 📁 Gestaun Dataset")
    return st.sidebar.file_uploader("Upload ficheiru Excel (.xlsx)", type=["xlsx"])

def exibir_kpi(total_funs, counts_real):
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("📊 Total Funsionáriu", total_funs)
    with col_m2:
        st.metric("⭐ Muito Bom", counts_real.get('Muito Bom', 0))
    with col_m3:
        st.metric("✨ Bom", counts_real.get('Bom', 0))
    with col_m4:
        st.metric("📌 Suficiente", counts_real.get('Suficiente', 0))
    with col_m5:
        st.metric("⚠️ Insuficiente", counts_real.get('Insuficiente', 0))

def exibir_grafico_performance(counts_real):
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Muito Bom", "Bom", "Suficiente", "Insuficiente"]
    sizes = [counts_real.get(cat, 0) for cat in categories]
    colors_list = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
    ax.pie(sizes, labels=categories, autopct="%1.1f%%", startangle=90, colors=colors_list, wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
    st.pyplot(fig)

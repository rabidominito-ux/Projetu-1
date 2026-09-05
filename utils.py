import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file):
    """Lê ficheiro Excel (.xlsx) ho re-uso em cache"""
    return pd.read_excel(file)

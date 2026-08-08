import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file):
    return pd.read_excel(file, sheet_name="Sheet1", header=0)

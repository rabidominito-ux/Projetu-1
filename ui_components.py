import streamlit as st

def render_custom_css():
    st.markdown("""
        <style>
        /* Estilu Fundu no Tipografia Prinsipal */
        .main {
            background-color: #F8FAFC;
        }
        .main-title {
            font-size: 2.4rem;
            color: #0F172A;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0px;
        }
        .sub-title {
            color: #475569;
            font-size: 1.1rem;
            margin-bottom: 25px;
        }
        
        /* Estilu Botun Generál */
        .stButton > button {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        }
        
        /* Estilu Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #F1F5F9;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
            color: #334155;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563EB !important;
            color: white !important;
        }
        
        /* Estilu Tabelas no Container sira */
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }
        </style>
    """, unsafe_allow_html=True)

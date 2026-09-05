import streamlit as st

def render_custom_css():
    st.markdown("""
        <style>
        /* Import Font Google */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Estilu Fundu no Tipografia Prinsipal */
        .stApp {
            background-color: #F8FAFC;
        }
        
        .main-title {
            font-size: 2.2rem;
            color: #0F172A;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }
        .sub-title {
            color: #475569;
            font-size: 1.05rem;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        
        /* Metric Cards Customizados */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 18px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-top: 4px;
        }
        
        /* Estilu Botun Generál */
        .stButton > button {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            color: white !important;
            border-radius: 8px;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.15);
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.25);
        }
        
        /* Estilu Tabs Ne'ebé Bonitu */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 2px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px 8px 0px 0px;
            padding: 12px 24px;
            font-weight: 600;
            color: #64748B;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E3A8A !important;
            color: #FFFFFF !important;
            border-bottom: 3px solid #D97706 !important;
        }
        
        /* Estilu Tabelas no Container sira */
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        
        /* Chapter IV Highlight Box */
        .capitulo-box {
            background-color: #EFF6FF;
            border-left: 5px solid #1E3A8A;
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

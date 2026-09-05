import streamlit as st

def render_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #F8FAFC;
        }
        .metric-card {
            background-color: #FFFFFF;
            padding: 1.2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #E2E8F0;
            text-align: center;
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1E293B;
            margin: 0.3rem 0;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #2563EB;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div style="background-color: #1E293B; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2.5rem;">🌳</div>
                <div>
                    <h2 style="margin: 0; color: #FFFFFF; font-size: 1.5rem; font-weight: 700;">
                        COMISSÃO DA FUNÇÃO PÚBLICA (CFP)
                    </h2>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.9rem;">
                        República Democrática de Timor-Leste | Sistema Inteligente de Classificação de Desempenho
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_kpi_card(title, value, subtext=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

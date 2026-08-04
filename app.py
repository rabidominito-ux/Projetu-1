import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Konfigurasaun Pajina Streamlit (Layout Wide)
st.set_page_config(
    page_title="Sistema Klasifikasaun CFP - Decision Tree",
    page_icon="📊",
    layout="wide"
)

# Custom CSS ba UI ne'ebé modernu no kapás
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 Sistema Klasifikasaun Dezempenu Funsionáriu CFP</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasaun uza algoritmu Decision Tree hodi klasifika dezempenu funsionáriu bazeia ba indikadór Komisaun Função Pública (CFP).</p>', unsafe_allow_html=True)

# Inicializa session_state
if 'reports_list' not in st.session_state:
    st.session_state['reports_list'] = []

if 'edit_index' not in st.session_state:
    st.session_state['edit_index'] = None

# 2. Sidebar ba Upload Dataset
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload ficheiru Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df_raw = pd.read_excel(file, sheet_name='Sheet1', header=0)
        return df_raw

    df_raw = load_data(uploaded_file)
    
    # Remapa koluna sira
    rename_map = {
        'Column1': 'controlo_ativo_identificacao',
        'Column2': 'nome_pessoal',
        'Column3': 'id_sigap',
        'Column4': 'id_grp',
        'Column5': 'sexo',
        'Column6': 'data_de_nascimento',
        'Column7': 'instituicao',
        'Column8': 'local_trabalho',
        'Column9': 'funcao',
        'Column10': 'cargo',
        'Column11': 'data_fim_nao_exercicio',
        'Column12': 'temp1',
        'Column13': 'Asiduidade',
        'Column14': 'Pontualidade',
        'Column15': 'Produtividade',
        'Column16': 'Kualidade_Servisu',
        'Column17': 'Kooperasaun',
        'Column18': 'Inisiativa',
        'Column19': 'Disiplina',
        'Column20': 'Responsabilidade',
        'Column21': 'Media',
        'Column22': 'Rezultadu_Avaliasaun',
        'Column23': 'temp2'
    }
    
    df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)

    nota_cols = ['Asiduidade', 'Pontualidade', 'Produtividade', 'Kualidade_Servisu',
                 'Kooperasaun', 'Inisiativa', 'Disiplina', 'Responsabilidade']
    target_col = 'Rezultadu_Avaliasaun'

    if all(col in df_raw.columns for col in nota_cols + [target_col]):
        df = df_raw.dropna(subset=nota_cols + [target_col]).copy()
        for col in nota_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        le = LabelEncoder()
        df['target_encoded'] = le.fit_transform(df[target_col])
        y = df['target_encoded']
        X = df[nota_cols].copy()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        df['Prediksaun'] = le.inverse_transform(model.predict(X))
        acc = accuracy_score(y_test, model.predict(X_test))

        # 3. Tabs Navegasaun
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Sumáriu", "⚙️ Preview & Treinu Modelu", "🔮 Prediksaun & Jere Relatóriu"])

        with tab1:
            st.subheader("📈 Dashboard Estatistika Dezempenu Funsionáriu")
            
            total_funs = len(df)
            counts_real = df[target_col].value_counts()
            
            # Kalkula porsentu
            mb_pct = (counts_real.get('Muito Bom', 0) / total_funs) * 100 if total_funs > 0 else 0
            b_pct = (counts_real.get('Bom', 0) / total_funs) * 100 if total_funs > 0 else 0
            s_pct = (counts_real.get('Suficiente', 0) / total_funs) * 100 if total_funs > 0 else 0
            i_pct = (counts_real.get('Insuficiente', 0) / total_funs) * 100 if total_funs > 0 else 0
            
            # KPI Cards modernu ho porsentu
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Total Funsionáriu", f"{total_funs}")
            col_m2.metric("Muito Bom", f"{counts_real.get('Muito Bom', 0)}", f"{mb_pct:.1f}%")
            col_m3.metric("Bom", f"{counts_real.get('Bom', 0)}", f"{b_pct:.1f}%")
            col_m4.metric("Suficiente", f"{counts_real.get('Suficiente', 0)}", f"{s_pct:.1f}%")
            col_m5.metric("Insuficiente", f"{counts_real.get('Insuficiente', 0)}", f"{i_pct:.1f}%")
            
            st.markdown("---")
            
            # Gráfiku rua: Bar Chart & Donut Chart
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("##### 📊 Komparasaun Kategoria (Reál vs Prediksaun)")
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Bar plot komparativa
                categories = ['Muito Bom', 'Bom', 'Suficiente', 'Insuficiente']
                real_counts = [counts_real.get(cat, 0) for cat in categories]
                pred_counts = [df['Prediksaun'].value_counts().get(cat, 0) for cat in categories]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax.bar(x - width/2, real_counts, width, label='Dadus Reál', color='#3B82F6')
                ax.bar(x + width/2, pred_counts, width, label='Prediksaun Tree', color='#10B981')
                
                ax.set_ylabel('Total Funsionáriu')
                ax.set_title('Distribuisaun Kategoria Dezempenu')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()
                st.pyplot(fig)

            with col_g2:
                st.markdown("##### 🍩 Donut Chart (Proporsaun Reál)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                
                sizes = [counts_real.get(cat, 0) for cat in categories]
                colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
                
                wedges, texts, autotexts = ax2.pie(
                    sizes, labels=categories, autopct='%1.1f%%', 
                    startangle=90, colors=colors, wedgeprops=dict(width=0.4, edgecolor='w')
                )
                plt.setp(autotexts, size=9, weight="bold")
                ax2.set_title('Proporsaun Kategoria Reál')
                st.pyplot(fig2)

        with tab2:
            st.subheader("📋 Dadus Amostra (Preview)")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🚀 Informasaun Modelu Decision Tree")
            st.success(f"✅ Modelu treinu ho suksesu! Akurasi Modelu: **{acc * 100:.2f}%**")

        with tab3:
            st.subheader("🔍 Halo Prediksaun no Jere Relatóriu Funsionáriu Foun")
            
            idx_edit = st.session_state['edit_index']
            def_val = {}
            if idx_edit is not None and idx_edit < len(st.session_state['reports_list']):
                def_val = st.session_state['reports_list'][idx_edit]
            
            with st.form("funsionariu_form"):
                st.markdown("##### 📝 1. Informasaun Identidade Funsionáriu")
                col_i1, col_i2, col_i3 = st.columns(3)
                with col_i1:
                    txt_nome = st.text_input("Naran Pessoal (nome_pessoal)", def_val.get('nome_pessoal', "João da Silva"))
                    txt_sigap = st.text_input("ID SIGAP (id_sigap)", def_val.get('id_sigap', "SIGAP-001"))
                    txt_sexo = st.selectbox("Sexo", ["M", "F"], index=0 if def_val.get('sexo', 'M')=='M' else 1)
                with col_i2:
                    txt_inst = st.text_input("Instituisaun", def_val.get('instituicao', "CFP"))
                    txt_local = st.text_input("Local Trabalhu", def_val.get('local_trabalho', "Dili"))
                    txt_nascimento = st.text_input("Data de Nascimento", def_val.get('data_de_nascimento', "1995-01-01"))
                with col_i3:
                    txt_funcao = st.text_input("Funsaun", def_val.get('funcao', "Tékniku"))
                    txt_cargo = st.text_input("Kargo", def_val.get('cargo', "Staff"))
                    txt_grp = st.text_input("ID GRP", def_val.get('id_grp', "GRP-123"))

                st.markdown("##### 📊 2. Indikadór Avaliasaun Funsionáriu (Skala 1 - 5)")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    p_asid = st.slider("Asiduidade", 1.0, 5.0, float(def_val.get('Asiduidade', 4.0)), 1.0)
                    p_pont = st.slider("Pontualidade", 1.0, 5.0, float(def_val.get('Pontualidade', 4.0)), 1.0)
                with col_b:
                    p_prod = st.slider("Produtividade", 1.0, 5.0, float(def_val.get('Produtividade', 4.0)), 1.0)
                    p_kual = st.slider("Kualidade Servisu", 1.0, 5.0, float(def_val.get('Kualidade_Servisu', 4.0)), 1.0)
                with col_c:
                    p_koop = st.slider("Kooperasaun", 1.0, 5.0, float(def_val.get('Kooperasaun', 4.0)), 1.0)
                    p_inis = st.slider("Inisiativa", 1.0, 5.0, float(def_val.get('Inisiativa', 4.0)), 1.0)
                with col_d:
                    p_disp = st.slider("Disiplina", 1.0, 5.0, float(def_val.get('Disiplina', 4.0)), 1.0)
                    p_resp = st.slider("Responsabilidade", 1.0, 5.0, float(def_val.get('Responsabilidade', 4.0)), 1.0)
                
                btn_label = "💾 Update Relatóriu" if idx_edit is not None else "🔮 Predict & Hatama Relatóriu"
                submit_pred = st.form_submit_button(btn_label)
                
                if submit_pred:
                    input_data = np.array([[p_asid, p_pont, p_prod, p_kual, p_koop, p_inis, p_disp, p_resp]])
                    pred_encoded = model.predict(input_data)
                    pred_label = le.inverse_transform(pred_encoded)[0]
                    
                    new_report = {
                        'nome_pessoal': txt_nome,
                        'id_sigap': txt_sigap,
                        'sexo': txt_sexo,
                        'instituicao': txt_inst,
                        'local_trabalho': txt_local,
                        'data_de_nascimento': txt_nascimento,
                        'funcao': txt_funcao,
                        'cargo': txt_cargo,
                        'id_grp': txt_grp,
                        'Asiduidade': p_asid,
                        'Pontualidade': p_pont,
                        'Produtividade': p_prod,
                        'Kualidade_Servisu': p_kual,
                        'Kooperasaun': p_koop,
                        'Inisiativa': p_inis,
                        'Disiplina': p_disp,
                        'Responsabilidade': p_resp,
                        'Rezultadu_Prediksaun': pred_label
                    }
                    
                    if idx_edit is not None:
                        st.session_state['reports_list'][idx_edit] = new_report
                        st.session_state['edit_index'] = None
                        st.success("✅ Relatóriu atualiza ho suksesu!")
                        st.rerun()
                    else:
                        st.session_state['reports_list'].append(new_report)
                        st.success("✅ Relatóriu foun rejista ho suksesu!")

            if len(st.session_state['reports_list']) > 0:
                st.markdown("---")
                st.subheader("📋 Lista Relatóriu Funsionáriu Foun (Jere Dadus)")
                
                for idx, rep in enumerate(st.session_state['reports_list']):
                    with st.expander(f"👤 {rep['nome_pessoal']} (SIGAP: {rep['id_sigap']}) - Rezultadu: {rep['Rezultadu_Prediksaun']}"):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.markdown(f"**Sexo:** {rep['sexo']} | **Fatin:** {rep['local_trabalho']}")
                            st.markdown(f"**Funsaun:** {rep['funcao']} | **Kargo:** {rep['cargo']}")
                        with col_d2:
                            st.markdown(f"✨ **Klasifikasaun:** `{rep['Rezultadu_Prediksaun']}`")
                        
                        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
                        with col_btn1:
                            if st.button("✏️ Edit", key=f"edit_{idx}"):
                                st.session_state['edit_index'] = idx
                                st.rerun()
                        with col_btn2:
                            if st.button("🗑️ Hamos", key=f"del_{idx}"):
                                st.session_state['reports_list'].pop(idx)
                                if st.session_state['edit_index'] == idx:
                                    st.session_state['edit_index'] = None
                                st.success("Dadus hamos ona!")
                                st.rerun()
                                
                        rep_df = pd.DataFrame([rep])
                        csv_data = rep_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Relatóriu (CSV)",
                            data=csv_data,
                            file_name=f"relatorio_{rep['id_sigap']}.csv",
                            mime='text/csv',
                            key=f"dl_{idx}"
                        )
else:
    st.info("👈 Favor upload uluk ficheiru Excel (`.xlsx`) iha sidebar sorin karuk hodi hahú sistema.")

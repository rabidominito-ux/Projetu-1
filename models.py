import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Koluna notas 8 tuir ordem exatu ne'ebé iha modelo .pkl Colab
nota_cols = [
    "Asiduidade",
    "Pontualidade",
    "Produtividade",
    "Kualidade_Servisu",
    "Kooperasaun",
    "Inisiativa",
    "Disiplina",
    "Responsabilidade",
]

target_col = "Rezultadu_Avaliasaun"
MODEL_PATH = "modelu_cfp.pkl"

def carregar_modelo_colab():
    """
    Karga diretu modelu DecisionTree husi 'modelu_cfp.pkl' (Colab).
    Kria LabelEncoder ho kategoria 4 exatu husi Colab.
    """
    # Karga modelo .pkl
    model = joblib.load(MODEL_PATH)

    # Kria LabelEncoder no fit ho kategoria ne'ebé existi iha modelo
    le = LabelEncoder()
    # Kategoria exatu 4 husi Colab: 'Bom', 'Insuficiente', 'Muito Bom', 'Suficiente'
    le.fit(["Bom", "Insuficiente", "Muito Bom", "Suficiente"])

    return model, le

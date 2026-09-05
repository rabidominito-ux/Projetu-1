import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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

def carregar_ou_treinar_modelo(df):
    """
    Karga modelu husi 'modelu_cfp.pkl' se iha tiha ona,
    ka treina foun no guarda ba .pkl se faíl ne'e la iha.
    """
    X = df[nota_cols].copy()
    y = df[target_col].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    if os.path.exists(MODEL_PATH):
        # Karga modelu tiha husi faíl .pkl
        model = joblib.load(MODEL_PATH)
    else:
        # Treina no guarda foun
        model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    return model, le, X_train, X_test, y_train, y_test

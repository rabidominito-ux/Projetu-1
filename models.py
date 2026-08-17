import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def treinar_modelo(df, nota_cols, target_col):
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df[target_col].astype(str))
    y = df["target_encoded"]
    X = df[nota_cols].copy()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Salva automatikamente modelu ne'ebé foin treinu ba fail .pkl
    salvar_modelo(model, le)
    
    return model, le, X_train, X_test, y_train, y_test

def carregar_modelo_pkl(filename="modelu_cfp.pkl"):
    try:
        dados_salvos = joblib.load(filename)
        return dados_salvos
    except Exception as e:
        return None

def salvar_modelo(model, le, filename="modelu_cfp.pkl"):
    try:
        # Rai hamutuk model no LabelEncoder iha formatu dictionary
        joblib.dump({"model": model, "le": le}, filename)
        return True
    except Exception as e:
        print(f"Erro atu salva modelu: {e}")
        return False

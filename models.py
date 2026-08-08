import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 1. Funsaun hodi treinu foun (Train Model)
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
    
    return model, le, X_train, X_test, y_train, y_test


# 2. Funsaun foun hodi carrega Modelu ne'ebé Rai ona (.pkl)
def carregar_modelo_pkl(filename="modelu_cfp.pkl"):
    try:
        # Karrega fail pkl ne'ebé iha model no LabelEncoder hamutuk (se save hamutuk)
        dados_salvos = joblib.load(filename)
        return dados_salvos
    except Exception as e:
        print(f"Erro iha carrega model pkl: {e}")
        return None

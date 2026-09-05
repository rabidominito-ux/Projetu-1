import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

DEFAULT_NOTA_COLS = [
    "Asiduidade",
    "Pontualidade",
    "Produtividade",
    "Kualidade_Servisu",
    "Kooperasaun",
    "Inisiativa",
    "Disiplina",
    "Responsabilidade",
]


def carregar_modelo_pkl(filename="modelu_cfp (4).pkl"):
    """Karga modelu .pkl directu husi original ne'ebe user manda."""
    try:
        dados = joblib.load(filename)
        if isinstance(dados, dict):
            return dados.get("model"), dados.get("le")
        elif isinstance(dados, DecisionTreeClassifier):
            return dados, None
        return None, None
    except Exception as e:
        print(f"Erru karga pkl: {e}")
        return None, None


def treinar_modelo(df, nota_cols, target_col):
    """Treina modelu foun uza algoritmo Decision Tree (entropy, max_depth=5)."""
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

    model = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)

    salvar_modelo(model, le)
    return model, le, X_train, X_test, y_train, y_test


def salvar_modelo(model, le, filename="modelu_cfp_novo.pkl"):
    try:
        joblib.dump({"model": model, "le": le}, filename)
        return True
    except Exception as e:
        print(f"Erro atu salva modelu: {e}")
        return False

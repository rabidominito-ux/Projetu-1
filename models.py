import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Nomeasaun padronizada ba koluna nota no target
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

def treinar_modelo(df):
    """
    Funsaun atu treina Decision Tree Classifier no transforma label sira.
    """
    X = df[nota_cols].copy()
    y = df[target_col].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_state=42, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X_train, y_train)

    return model, le, X_train, X_test, y_train, y_test

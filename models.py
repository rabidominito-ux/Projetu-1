import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def treinar_modelo_novo(df, nota_cols, target_col="Rezultadu_Avaliasaun"):
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df[target_col].astype(str))
    
    X = df[nota_cols]
    y = df["target_encoded"]
    
    model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    model.fit(X, y)
    return model, le

def carregar_modelo_pkl(filename="modelu_cfp.pkl"):
    try:
        dados_carregados = joblib.load(filename)
        # Nota: Depende oinsá Ita save ona file .pkl (bele de'it model ka tuple [model, le])
        return dados_carregados
    except Exception as e:
        return None

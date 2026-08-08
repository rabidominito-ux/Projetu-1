import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def treinar_modelo(df, nota_cols, target_col="Rezultadu_Avaliasaun"):
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df[target_col].astype(str))
    
    X = df[nota_cols]
    y = df["target_encoded"]
    
    model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    model.fit(X, y)
    return model, le

def fazer_predicao(model, le, input_data):
    pred_encoded = model.predict(input_data)
    return le.inverse_transform(pred_encoded)[0]

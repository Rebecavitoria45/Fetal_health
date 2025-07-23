import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  

def tratar_outliers_iqr(df, fator=1.5):
    df_tratado = df.copy()
    for col in df_tratado.columns:
        if pd.api.types.is_numeric_dtype(df_tratado[col]):
            Q1 = df_tratado[col].quantile(0.25)
            Q3 = df_tratado[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - fator * IQR
            limite_superior = Q3 + fator * IQR
            df_tratado[col] = np.where(
                df_tratado[col] < limite_inferior,
                limite_inferior,
                np.where(df_tratado[col] > limite_superior, limite_superior, df_tratado[col])
            )
    return df_tratado


df = pd.read_csv("/data/fetal_health.csv")
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

X = tratar_outliers_iqr(X) 

# Escalonando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)


models = {
    "KNeighbors": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced"),
        {
            "n_estimators": [50, 100, 150, 200, 250],
            "max_depth": [5, 10, 15, 20, None],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 4, 6, 8, 10],
        },
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced"),
        {
            "C": [0.1, 1, 10, 100, 1000],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.01, 0.001],
            "degree": [2, 3, 4],
        },
    ),
    "DecisionTree": (
        DecisionTreeClassifier(class_weight="balanced"),
        {
            "max_depth": [5, 10, 15, 20, None],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 4, 6],
        },
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, solver="saga", class_weight="balanced"),
        {
            "C": [0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet"],
            "fit_intercept": [True, False],
            "l1_ratio": [0, 0.5, 1],
        },
    ),
    "NaiveBayes": (
        GaussianNB(),
        {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
    ),
}

mlflow.set_tracking_uri("file:/app/mlruns")
mlflow.set_experiment("fetal_health_exp")

best_score = 0
best_model = None


for name, (model, params) in models.items():
    print(f"\nTreinando: {name}")
    with mlflow.start_run(run_name=name):
        grid = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train,y_train)

        final_model = grid.best_estimator_

        preds = final_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")

        print(f"Acurácia: {acc:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)

        if acc > best_score:
            best_score = acc
            best_model = final_model


if best_model:
    os.makedirs("/app/models", exist_ok=True)
    joblib.dump(best_model, "/app/models/best_model.pkl")
    print("Melhor modelo salvo com sucesso!")

    with mlflow.start_run(run_name="Best_Model"):
        mlflow.sklearn.log_model(best_model, "best_model")

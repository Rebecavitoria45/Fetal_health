from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
model_path = "../models/best_model.pkl"


if not os.path.exists(model_path):
    print("Modelo não encontrado. Treinando...")
    os.system("python train.py")

# Carrega o modelo treinado
model = joblib.load(model_path)

app = Flask(__name__)
model = joblib.load("../models/best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
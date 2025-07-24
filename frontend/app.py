from flask import Flask, request, render_template, jsonify
import pandas as pd
import requests

app = Flask(__name__)
MODEL_URL = "http://fetal_health_model:5000/predict"

LABELS_MAP = {
    1: "Normal",
    2: "Suspeito",
    3: "Patológico"
}

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)

            # Remove a coluna alvo se existir
            if 'fetal_health' in df.columns:
                df = df.drop(columns=['fetal_health'])

            results = []
            for _, row in df.iterrows():
                payload = {"features": row.tolist()}
                try:
                    response = requests.post(MODEL_URL, json=payload)
                    prediction = response.json().get("prediction", [None])[0]
                    prediction_label = LABELS_MAP.get(prediction, "Erro")
                    results.append(prediction_label)
                except Exception as e:
                    results.append(f"Erro: {str(e)}")

            df.insert(0, "Classificação Fetal", results)
            table_html = df.to_html(classes="table table-striped table-bordered", index=False, justify="center")
            return render_template("results.html", table=table_html)

        return "Arquivo inválido ou ausente."
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)
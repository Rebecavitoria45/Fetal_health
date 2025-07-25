# 🧠 Fetal Health Prediction

Este projeto tem como objetivo treinar, rastrear e servir modelos de machine learning para prever a saúde fetal com base em dados de cardiotocografia. Utilizamos o *MLflow* para rastreamento de experimentos.

## 📁 Estrutura do Projeto

├── data/                  # Base de dados fetal_health.csv <br>
├── mlruns/               # Logs do MLflow <br>
├── models/               # Modelo final treinado <br>
├── src/                
│   ├── train.py          # Treinamento com GridSearchCV e MLflow <br>
│   ├── serve.py          # API Flask para servir modelo <br>
├── frontend/             # Aplicação Frontend <br>
├── Dockerfile            # Dockerfile do serviço de modelo <br>
├── docker-compose.yml    # Orquestração dos serviços <br>

## 🚀 Tecnologias Utilizadas

- Python 3.10+
- Pandas, Scikit-learn, Imbalanced-learn
- MLflow
- Docker & Docker Compose
- Flask (Serviço de predição)

## 📊 Dataset

O dataset é baseado em sinais obtidos por cardiotocografia (CTG), contendo 21 variáveis numéricas e um rótulo (fetal_health) com 3 classes:

- *1.0* - Normal
- *2.0* - Suspeito
- *3.0* - Patológico


## ⚙️ Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/fetal-health-mlflow.git
cd fetal-health-mlflow
```

### 2. Execute com Docker Compose

```bash
docker-compose up --build
```

Isso irá iniciar:

- ✅ fetal_health_model: serviço Flask com o modelo treinado
- ✅ frontend: interface para upload de CSV
- ✅ mlflow_ui: painel para rastrear experimentos em http://localhost:5001


## 🧪 Treinamento dos Modelos

- Algoritmos usados: Random Forest, SVM, KNN, Naive Bayes, Decision Tree e Regressão Logística.
- Hiperparâmetros ajustados via GridSearchCV.
- Métrica de avaliação: *Acurracy*.

O melhor modelo é salvo em models/best_model.pkl e logado no MLflow.

---

## 🖼️ Interface Web

Abra [http://localhost:8501](http://localhost:8501) para acessar o frontend.  
Lá é possível:

- Fazer upload de um arquivo .csv
- Visualizar os dados
- Enviar os dados ao modelo via API Flask
- Obter as previsões da saúde fetal

---

## 📈 MLflow UI

Visualize e compare todos os experimentos de treinamento acessando:

📍 http://localhost:5001

---

## 🧼 Limpeza

Para parar e remover os containers:

```bash
docker-compose down
```

---

## 📌 Requisitos para ambiente local (opcional)

Caso deseje rodar sem Docker:

- Python 3.10+
- Instale dependências:

```bash
pip install -r requirements.txt
```

- Execute localmente:

```bash
python src/train.py
python src/serve.py
streamlit run frontend/app.py
```

---

## Autores
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Euerickbruno">
        <img src="https://avatars.githubusercontent.com/u/176349629?v=4" width="100px;" alt="Erick"/>
        <br>
        <b>Erick Bruno</b>
      </a>
    </td>
      <td align="center">
      <a href="https://github.com/gabriellamarinho">
        <img src="https://avatars.githubusercontent.com/u/186753301?v=4" width="100px;" alt="Marilia"/>
        <br>
        <b>Marilia Gabriella</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Rebecavitoria45">
       <img src="https://avatars.githubusercontent.com/u/117654851?v=4" width="100px;" alt="Rebeca"/>
        <br>
        <b>Rebeca vitória</b>
      </a>
    </td>

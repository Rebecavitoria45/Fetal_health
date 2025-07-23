FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p ./models


COPY src/ ./src/



WORKDIR /app/src
CMD ["python", "serve.py"]
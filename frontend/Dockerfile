FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install flask pandas requests

EXPOSE 8501
CMD ["python", "app.py"]
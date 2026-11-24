# Imagen para ENTRENAR (src/train.py). El código y los datos se montan desde docker-compose.yml.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt

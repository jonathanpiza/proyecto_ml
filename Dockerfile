FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

# Ejecutar aplicación
CMD ["python"]

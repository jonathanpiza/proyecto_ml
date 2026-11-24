## Cómo ejecutar el proyecto

Requisito: Docker Desktop abierto. Todos los comandos se ejecutan desde la raíz del proyecto.

### 1. Preparar el entorno de entrenamiento
```powershell
docker compose build ml
```

### 2. Entrenar el modelo
```powershell
docker compose run --rm ml python src/train.py --csv data/ventas_demo.csv --fecha fecha --valor monto --agg sum --grupo producto
```

### 3. Encender la API y el dashboard
```powershell
docker compose up --build
```

- Dashboard: http://localhost:8501
- API (documentación): http://localhost:8000/docs

Para detener todo, presiona `Ctrl+C` y ejecuta `docker compose down`.

### Con PQRS en lugar de ventas
```powershell
docker compose run --rm ml python src/train.py --csv data/pqrs_demo.csv --fecha fecha_radicado --agg count --grupo tipo
```

### Despliegue en Kubernetes (Minikube)
```powershell
minikube start
powershell -ExecutionPolicy Bypass -File k8s/desplegar.ps1
kubectl port-forward svc/dashboard 8501:8501
```
# Despliega todo el proyecto en Minikube. Ejecutar desde la RAÍZ del proyecto:
#   powershell -ExecutionPolicy Bypass -File k8s/desplegar.ps1
# Requisitos: Docker Desktop abierto, y Minikube iniciado (minikube start).

function Paso($texto) { Write-Host "`n== $texto" -ForegroundColor Cyan }
function Verificar { if ($LASTEXITCODE -ne 0) { throw "Falló el comando anterior (código $LASTEXITCODE). Revisa el mensaje de arriba." } }

Paso "1/6 Construyendo las imágenes"
docker build -t prediccion-api:1.0 ./api; Verificar
docker build -t prediccion-dashboard:1.0 ./dashboard; Verificar
docker build -f Dockerfile.train -t prediccion-train:1.0 .; Verificar

Paso "2/6 Cargando las imágenes en Minikube"
minikube image load prediccion-api:1.0; Verificar
minikube image load prediccion-dashboard:1.0; Verificar
minikube image load prediccion-train:1.0; Verificar

Paso "3/6 Volumen y CronJob"
kubectl apply -f k8s/pvc.yaml; Verificar
kubectl apply -f k8s/cronjob.yaml; Verificar

Paso "4/6 Entrenamiento inicial (crea el modelo en el volumen)"
kubectl delete job entrenamiento-inicial --ignore-not-found
kubectl create job entrenamiento-inicial --from=cronjob/reentrenamiento; Verificar
kubectl wait --for=condition=complete job/entrenamiento-inicial --timeout=300s; Verificar

Paso "5/6 API y dashboard"
kubectl apply -f k8s/configmap.yaml; Verificar
kubectl apply -f k8s/api.yaml; Verificar
kubectl apply -f k8s/dashboard.yaml; Verificar
kubectl rollout status deployment/api --timeout=180s; Verificar
kubectl rollout status deployment/dashboard --timeout=180s; Verificar

Paso "6/6 Estado final"
kubectl get pods,svc,cronjob
Write-Host "`nListo. Para abrir el dashboard:  kubectl port-forward svc/dashboard 8501:8501" -ForegroundColor Green
Write-Host "y luego entra a http://localhost:8501" -ForegroundColor Green

# Setup Guide — Heart Disease MLOps Project

## Prerequisites

| Tool | Version | When needed |
|------|---------|-------------|
| Python | 3.11+ | All tasks (local training) |
| pip / conda | latest | All tasks |
| Git | any | Clone the repo |

### For Monitoring & Deployment (pick ONE)

**Option A: Docker Compose** (recommended for learning)
- Docker | 24+ | Section 8 (local dev with full stack)

**Option B: Kubernetes** (production-realistic)
- Docker | 24+ | Build the image
- Minikube | latest | Section 9 (local K8s cluster)
- kubectl | latest | Deploy to cluster
- On Windows: WSL 2 backend for Docker Desktop recommended

---

## 1. Clone and Install

```bash
git clone https://github.com/ravikiran-k-19/MLOps-Assignment-1.git
cd heart-disease-mlops
```

**Option A — pip (virtualenv)**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Option B — conda**
```bash
conda env create -f environment.yml
conda activate heart-disease-mlops
```

---

## 2. Add the Dataset

Download the Heart Disease UCI dataset (Cleveland subset):

```
https://archive.ics.uci.edu/dataset/45/heart+disease
```

Copy the processed Cleveland file to:
```
data/heart_disease.csv
```

The file must have **no header row**, 14 comma-separated columns in this order:
```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num
```
Missing values are represented as `?` — the preprocessing pipeline handles them automatically.

---

## 3. Run EDA (Task 1)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 4. Train Models + MLflow Tracking (Tasks 2 & 3)

Start the MLflow tracking server (in a separate terminal):
```bash
mlflow server --host 0.0.0.0 --port 5000
```

Run training:
```bash
# From the project root so 'from src.xxx import ...' resolves
PYTHONPATH=. python -m src.train        # macOS / Linux
$env:PYTHONPATH="."; python -m src.train  # Windows PowerShell
```

What happens:
- Trains Logistic Regression, Random Forest, XGBoost with GridSearchCV
- Logs every run (params, metrics, ROC curves, confusion matrices) to MLflow
- Saves the best model to `models/best_model.joblib`
- Registers the best model in the MLflow Model Registry

**View experiments:**  http://localhost:5000

---

## 5. Run the Prediction API locally (Task 6)

```bash
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

**API docs (Swagger UI):** http://localhost:8000/docs

**Test a prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
       "fbs":1,"restecg":0,"thalach":150,"exang":0,
       "oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Expected response:
```json
{"prediction": 0, "confidence": 0.32, "model_version": "1.0.0"}
```

### Prometheus metrics

Your API already exposes these custom metrics at `http://localhost:8000/metrics`:
- `api_request_total` — total requests by method, endpoint, and status
- `api_request_latency_seconds` — request latency histogram
- `prediction_total` — total predictions by result class (0 or 1)
- `model_loaded` — gauge indicating if model is loaded (requires code addition below)

**Add MODEL_LOADED gauge to `app/main.py`:**

After the other Prometheus imports, add:
```python
from prometheus_client import Gauge

MODEL_LOADED = Gauge("model_loaded", "Whether the model is currently loaded (1=yes, 0=no)")
```

In the `lifespan()` function, set the gauge when the model loads:
```python
if model_path.exists():
    _model = joblib.load(model_path)
    MODEL_LOADED.set(1)  # ← Add this line
    logger.info(...)
else:
    MODEL_LOADED.set(0)  # ← Add this line
    logger.warning(...)
```

This enables the "Model Status" panel in the Grafana dashboard (green = loaded, red = not loaded).

---

## 6. Run Tests (Task 5 — CI/CD)

```bash
PYTHONPATH=. pytest tests/ -v --cov=src --cov=app --cov-report=term-missing
```

Lint:
```bash
flake8 src/ app/ tests/ --max-line-length=100
```

---

## 7. Docker — Build and Run (Task 6)

```bash
docker build -t heart-disease-api:latest .

docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  heart-disease-api:latest
```

Windows PowerShell:
```powershell
docker run -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  heart-disease-api:latest
```

---

## 8. Docker Compose — Full Stack (Tasks 6 & 8)

**⚠️ IMPORTANT:** Use **EITHER Docker Compose OR Kubernetes**, not both. They both try to bind to the same ports and will conflict.

### When to use Docker Compose
- ✅ Local development and testing
- ✅ Quick prototyping of the monitoring stack
- ✅ Learning Prometheus + Grafana without K8s complexity
- ✅ CI/CD pipeline validation
- ✅ **Recommended for this assignment if you're just learning MLOps**

### Before starting, clean up any conflicting processes

**On Windows PowerShell:**
```powershell
# Kill any process using ports 8000, 9090, 3000
$ports = @(8000, 9090, 3000)
foreach ($port in $ports) {
  $pid = (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue).OwningProcess
  if ($pid) { Stop-Process -Id $pid -Force; Write-Host "Killed process on port $port" }
}

# Stop Docker Compose if already running
docker-compose down
```

**On Linux / macOS:**
```bash
# Kill processes on ports 8000, 9090, 3000
for port in 8000 9090 3000; do
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

# Stop Docker Compose if already running
docker-compose down
```

### Start the full stack

```bash
docker-compose up --build
```

This starts three services on a **shared Docker network**:
- **API** — binds to port 8000, exposes `/metrics` endpoint
- **Prometheus** — binds to port 9090, scrapes from `http://api:8000/metrics`
- **Grafana** — binds to port 3000, reads from `http://prometheus:9090`

| Service | URL |
|---------|-----|
| API + Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

**Grafana is auto-configured:**
- Datasource "Prometheus" is already added (URL: `http://prometheus:9090`)
- Dashboard "Heart Disease Prediction API" is already imported
- No manual setup needed — just log in and view data

**Test the stack:**
```bash
# Generate some traffic to populate metrics
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# Check Prometheus is scraping the API
# Open http://localhost:9090/targets → heart-disease-api should show UP (green)

# View the monitoring dashboard
# Open http://localhost:3000 → login admin/admin → Dashboards → Heart Disease Prediction API
```

---

## 9. Kubernetes Deployment (Task 7)

**⚠️ IMPORTANT:** Use **EITHER Docker Compose OR Kubernetes**, not both. Pick one approach for your assignment.

### When to use Kubernetes
- ✅ Production-like deployment with orchestration
- ✅ Learning container orchestration and scaling
- ✅ Multi-pod HA (Horizontal Pod Autoscaling)
- ✅ Showcasing DevOps / platform engineering knowledge
- ⚠️ More complex setup; recommended if you want K8s experience for your resume

### Architecture

Kubernetes runs **three separate deployments** on the same cluster, communicating via **Kubernetes DNS**:

```
┌─────────────────┐
│   Grafana Pod   │ ← Reads metrics from prometheus:9090
└─────────────────┘
        ↓
┌─────────────────┐
│ Prometheus Pod  │ ← Scrapes from heart-disease-api:8000/metrics
└─────────────────┘
        ↓
┌─────────────────┐
│   API Pod #1    │ ← Exposes /metrics endpoint
│   API Pod #2    │ ← (2 replicas for HA)
└─────────────────┘
```

**Files needed:**
- `k8s/deployment.yaml` — API (2 replicas)
- `k8s/service.yaml` — API Service
- `k8s/prometheus-deployment.yaml` — Prometheus ConfigMap + Deployment + Service
- `k8s/grafana-deployment.yaml` — Grafana ConfigMaps + Deployment + Service

### Before starting, clean up

**Stop Docker Compose first (if running):**
```bash
docker-compose down
```

**Stop any prior Kubernetes resources:**
```bash
kubectl delete -f k8s/ 2>/dev/null || true
```

### Minikube setup (local Kubernetes)

```bash
# Start cluster
minikube start

# Point Docker CLI to Minikube's daemon (so docker build runs inside Minikube)
eval $(minikube docker-env)              # Linux / macOS
minikube docker-env | Invoke-Expression  # Windows PowerShell

# Build image inside Minikube (not on your host)
docker build -t heart-disease-api:latest .

# Deploy all Kubernetes resources
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml

# Verify all pods are running
kubectl get pods
kubectl get services
```

### Access the services from your browser

Kubernetes services aren't accessible from localhost by default. Use `port-forward` to tunnel from your machine into the cluster:

**Open three terminals and run (one per terminal):**

```bash
# Terminal 1: API
kubectl port-forward service/heart-disease-api 8000:80

# Terminal 2: Prometheus
kubectl port-forward service/prometheus 9090:9090

# Terminal 3: Grafana
kubectl port-forward service/grafana 3000:3000
```

| Service | URL |
|---------|-----|
| API + Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

### Verify the monitoring stack works

**1. Check Prometheus can scrape the API:**
```bash
# Open http://localhost:9090/targets
# You should see: heart-disease-api job → UP (green)
```

If it shows **DOWN**, the pods may still be starting. Wait 10-15 seconds and refresh.

**2. Generate some traffic:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

**3. View the Grafana dashboard:**
- Open http://localhost:3000
- Login: admin / admin
- Dashboards → "Heart Disease Prediction API"
- You should see 7 panels with live data (request rate, latency, errors, predictions, model status)

### Useful kubectl commands

```bash
# Check pod status
kubectl get pods
kubectl get deployments

# See pod logs
kubectl logs -l app=heart-disease-api
kubectl logs -l app=prometheus
kubectl logs -l app=grafana

# Get more details
kubectl describe deployment heart-disease-api
kubectl describe pod <pod-name>

# Scale the API to 3 replicas
kubectl scale deployment heart-disease-api --replicas=3

# Delete all resources when done
kubectl delete -f k8s/
```

### Key differences: Docker Compose vs Kubernetes

| Feature | Docker Compose | Kubernetes |
|---------|----------------|-----------|
| **Setup complexity** | Easy (one command) | Moderate (3 files) |
| **Service discovery** | By service name (`api:8000`) | By DNS (`heart-disease-api:8000`) |
| **Port binding** | Direct (localhost:8000) | Via `port-forward` |
| **Scaling** | Manual (edit replicas in compose) | Dynamic (`kubectl scale`) |
| **Learning curve** | Low | Medium-High |
| **Best for** | Dev/testing | Production, portfolios |

---

## 10. CI/CD Pipeline (Task 5)

The `.github/workflows/ci.yml` workflow runs automatically on every push to `main` or `develop`:

1. **Lint** — `flake8` checks code style
2. **Test** — `pytest` runs unit tests + coverage report
3. **Build** — Docker image is built to validate `Dockerfile`

The build step is skipped if linting or tests fail — this enforces code quality gates.

---

## 11. Troubleshooting

### Port Already Allocated

**Problem:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:** Kill the process using that port before starting docker-compose or kubectl.

**Windows PowerShell:**
```powershell
$pid = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess
if ($pid) { Stop-Process -Id $pid -Force }
```

**Linux/macOS:**
```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
```

---

### Prometheus shows "DOWN" (can't scrape API)

**Problem:** Prometheus targets show red/DOWN status

**In Docker Compose:**
- Check that `prometheus.yml` has `targets: ["api:8000"]` (service name, not localhost)
- All services must be on the same Docker network
- Run `docker-compose ps` to verify all containers are running

**In Kubernetes:**
- Check that Prometheus ConfigMap has `targets: ["heart-disease-api:8000"]`
- Verify API pod is running: `kubectl get pods`
- Check API service exists: `kubectl get service heart-disease-api`
- Restart Prometheus: `kubectl rollout restart deployment/prometheus`

---

### No metrics appearing in Grafana

**Problem:** Grafana panels show "No data"

**Checklist:**
1. Is Prometheus scraping the API? (Check http://localhost:9090/targets)
2. Have you generated traffic? Run a `/predict` request to populate metrics
3. Is the datasource connected? (Grafana → Connections → Data sources → check Prometheus)
4. Wait 30-60 seconds — Prometheus scrapes every 15s, Grafana refreshes every 10s

**Fix:**
```bash
# Generate some traffic
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# Wait 30 seconds, then refresh Grafana
```

---

### Model Status shows "Not Loaded" or no data

**Problem:** Grafana "Model Status" panel is empty or red

**Cause:** The `MODEL_LOADED` gauge is not defined in `app/main.py`

**Solution:** Add the gauge as shown in Section 5 above, then:
```bash
# Docker Compose
docker-compose up -d --build

# OR Kubernetes
docker build -t heart-disease-api:latest .
kubectl rollout restart deployment/heart-disease-api
```

---

### Can't access Grafana / Prometheus from localhost

**In Kubernetes only:**

You must use `port-forward` to access services:
```bash
kubectl port-forward service/grafana 3000:3000
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/heart-disease-api 8000:80
```

Don't forget the `port-forward` — Kubernetes services are not exposed on localhost by default.

---

### Minikube docker-env not working

**Problem:** `docker build` inside Minikube fails

**Windows PowerShell:**
```powershell
# First check what minikube docker-env prints
minikube docker-env

# Then run it
minikube docker-env | Invoke-Expression

# Verify it worked
docker ps  # Should show Minikube containers
```

**Linux/macOS:**
```bash
eval $(minikube docker-env)
docker ps  # Should show Minikube containers
```

---

### Docker Compose and Kubernetes conflict

**Problem:** Ports 8000, 9090, 3000 already in use

**Root cause:** You're running both Docker Compose and Kubernetes at the same time.

**Solution:** Pick ONE approach per session:

```bash
# Stop Docker Compose
docker-compose down

# Then start Kubernetes
kubectl apply -f k8s/

# OR vice versa:
kubectl delete -f k8s/
docker-compose up --build
```

---

## Project Structure

```
heart-disease-mlops/
│
├── data/                    # Dataset (not committed — add heart_disease.csv here)
├── models/                  # Trained model artifact (not committed)
├── mlruns/                  # MLflow tracking data (not committed)
│
├── notebooks/
│   └── 01_eda.ipynb         # Task 1: EDA
│
├── src/
│   ├── config.py            # All paths and constants (single source of truth)
│   ├── data_preparation.py  # Load → clean → split → preprocess
│   └── train.py             # Train models, track with MLflow, register best
│
├── app/
│   ├── schemas.py           # Pydantic request/response models
│   └── main.py              # FastAPI app — /health, /predict, /metrics
│
├── tests/
│   ├── test_data_preparation.py
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml           # API Deployment (2 replicas with health probes)
│   ├── service.yaml              # API Service (LoadBalancer, port 80 → 8000)
│   ├── prometheus-deployment.yaml # Prometheus ConfigMap + Deployment + Service
│   └── grafana-deployment.yaml    # Grafana ConfigMaps + Deployment + Service
│
├── monitoring/
│   └── prometheus.yml       # Prometheus scrape config
│
├── .github/workflows/
│   └── ci.yml               # GitHub Actions: lint → test → docker build
│
├── Dockerfile               # Multi-stage build (builder + runtime)
├── docker-compose.yml       # api + prometheus + grafana
├── requirements.txt
├── environment.yml
└── .gitignore
```

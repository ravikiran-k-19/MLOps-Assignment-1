# Setup Guide — Heart Disease MLOps Project

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | |
| pip / conda | latest | |
| Docker | 24+ | Tasks 6, 8 |
| Minikube or Docker Desktop K8s | latest | Task 7 |
| Git | any | |

---

## 1. Clone and Install

```bash
git clone <your-repo-url>
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

Starts the API, Prometheus, and Grafana together:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| API + Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

**Configure Grafana:**
1. Open http://localhost:3000 → login admin / admin
2. Add data source → Prometheus → URL: `http://prometheus:9090`
3. Create dashboard → use metrics:
   - `api_request_total` — request count by endpoint + status
   - `api_request_latency_seconds` — response time histogram
   - `prediction_total` — prediction counts by label

---

## 9. Kubernetes Deployment (Task 7)

### Minikube (local)

```bash
# Start cluster
minikube start

# Point Docker CLI to Minikube's daemon (so kubectl can pull the image)
eval $(minikube docker-env)              # Linux / macOS
minikube docker-env | Invoke-Expression  # Windows PowerShell

# Build image inside Minikube
docker build -t heart-disease-api:latest .

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify
kubectl get pods
kubectl get services

# Open in browser
minikube service heart-disease-api
```

### Useful kubectl commands

```bash
kubectl describe deployment heart-disease-api   # deployment status
kubectl logs -l app=heart-disease-api           # pod logs
kubectl scale deployment heart-disease-api --replicas=3  # scale out
kubectl delete -f k8s/                          # tear down
```

---

## 10. CI/CD Pipeline (Task 5)

The `.github/workflows/ci.yml` workflow runs automatically on every push to `main` or `develop`:

1. **Lint** — `flake8` checks code style
2. **Test** — `pytest` runs unit tests + coverage report
3. **Build** — Docker image is built to validate `Dockerfile`

The build step is skipped if linting or tests fail — this enforces code quality gates.

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
│   ├── deployment.yaml      # 2-replica Deployment with health probes
│   └── service.yaml         # LoadBalancer Service (port 80 → 8000)
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

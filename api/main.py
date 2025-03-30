import os
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

MODEL_DIR = Path(os.path.dirname(__file__)) / "../build/models"

svc_model_path = MODEL_DIR / "svc_model.pkl"
print(f"Loading SVC model from {svc_model_path.resolve()}")
with svc_model_path.open("rb") as f:
    svc_engine = pickle.load(f)

knn_model_path = MODEL_DIR / "knn_model.pkl"
print(f"Loading KNN model from {knn_model_path.resolve()}")
with knn_model_path.open("rb") as f:
    knn_engine = pickle.load(f)

class QueryRequest(BaseModel):
    query: str

class RoutePrediction(BaseModel):
    id: str
    path: str

@app.post("/predict/svc")
def predict_svc(request: QueryRequest):
    predictions = svc_engine.predict(request.query)
    return {"predictions": [RoutePrediction(id=route.id, path=route.path) for route in predictions]}

@app.post("/predict/knn")
def predict_knn(request: QueryRequest):
    predictions = knn_engine.predict(request.query)
    return {"predictions": [RoutePrediction(id=route.id, path=route.path) for route in predictions]}

def run():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

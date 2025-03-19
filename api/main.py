import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MODEL_DIR = "../build/models"
with open(os.path.join(MODEL_DIR, "svc_model.pkl"), "rb") as f:
    svc_engine = pickle.load(f)
with open(os.path.join(MODEL_DIR, "knn_model.pkl"), "rb") as f:
    knn_engine = pickle.load(f)

class QueryRequest(BaseModel):
    query: str

@app.post("/predict/svc")
def predict_svc(request: QueryRequest):
    predictions = svc_engine.predict(request.query)
    return {"predictions": [route.dict() for route in predictions]}

@app.post("/predict/knn")
def predict_knn(request: QueryRequest):
    predictions = knn_engine.predict(request.query)
    return {"predictions": [route.dict() for route in predictions]}

def run():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

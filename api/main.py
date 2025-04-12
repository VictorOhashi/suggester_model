import uvicorn
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from api.models_loader import SearchModelLoader

app = FastAPI()
model_loader = SearchModelLoader()

class QueryRequest(BaseModel):
    query: str

class RoutePrediction(BaseModel):
    id: str
    path: str

@app.post("/predict/svc")
async def predict_svc(request: QueryRequest):
    model = await model_loader.load_svc()
    predictions = model.predict(request.query)
    return {"predictions": [RoutePrediction(id=route.id, path=route.path) for route in predictions]}

@app.post("/predict/knn")
async def predict_knn(request: QueryRequest):
    model = await model_loader.load_knn()
    predictions = model.predict(request.query)
    return {"predictions": [RoutePrediction(id=route.id, path=route.path) for route in predictions]}

def run():
    """Launched with `poetry run start` at root level"""
    asyncio.run(model_loader.preload_models())
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

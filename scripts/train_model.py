import os
import pickle
import asyncio
from notebooks.engines.sailor_data_engineer import RouteGenConfig, SailorDataEngineer
from sailor.vector_sailor_engine import SVCSailorEngine, KNNSailorEngine
from sailor.route_vectorizer import TfidfRouteVectorizer

async def train():
    _config = RouteGenConfig.from_env(dir="./build/cache")
    _enginner = SailorDataEngineer(
        _config,
        cache_key='validate_model',
        route_description="flight agency admin panel")

    route_context = await _enginner.generate_data(route_count=20, session_count=100)
    if route_context is None:
        raise ValueError("No data generated")

    model_dir = "./build/models"
    os.makedirs(model_dir, exist_ok=True)
    for file in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, file))

    # Train SVC Model
    svc_vectorizer = TfidfRouteVectorizer()
    svc_engine = SVCSailorEngine(svc_vectorizer)
    svc_engine.train(route_context.routes, route_context.sessions)
    with open(os.path.join(model_dir, "svc_model.pkl"), "wb") as f:
        pickle.dump(svc_engine, f)

    # Train KNN Model
    knn_vectorizer = TfidfRouteVectorizer()
    knn_engine = KNNSailorEngine(knn_vectorizer)
    knn_engine.train(route_context.routes, route_context.sessions)
    with open(os.path.join(model_dir, "knn_model.pkl"), "wb") as f:
        pickle.dump(knn_engine, f)

def run():
    """Launched with `poetry run train` at root level"""
    asyncio.run(train())

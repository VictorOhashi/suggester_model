import asyncio
import os
import pickle
import aiofiles
from pathlib import Path

_SVC_MODEL = "svc_model"
_KNN_MODEL = "knn_model"

_MODEL_DIR = Path(os.path.dirname(__file__)) / "../build/models"

class SearchModelLoader:
    def __init__(self):
        self.models = {}

    async def preload_models(self):
        print("Preloading models...")
        models_coroutine = (
            self._load(model_name)
            for model_name in [_SVC_MODEL, _KNN_MODEL]
        )
        await asyncio.gather(*models_coroutine)

    async def _load(self, model_name: str):
      if model_name in self.models:
        return self.models[model_name]

      model_path = _MODEL_DIR / f"{model_name}.pkl"
      print(f"Loading model from {model_path.resolve()}")
      async with aiofiles.open(model_path, "rb") as f:
          file = await f.read()
          self.models.update({model_name: pickle.loads(file)})

      return self.models[model_name]

    async def load_svc(self):
        return await self._load(_SVC_MODEL)

    async def load_knn(self):
        return await self._load(_KNN_MODEL)

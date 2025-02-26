import time
from typing import List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sailor import VectorSailorEngine, NavigationContext, SessionSpec, RouteGenConfig, SailorDataEngineer

class VectorTestEngine:
  def __init__(self, engine: VectorSailorEngine, key: str = "vector_test"):
    _config = RouteGenConfig.fromEnv()
    self._engineer = SailorDataEngineer(_config, cache_key=key)
    self.engine = engine
    self.route_context: Optional[NavigationContext] = None
    self.train_sessions: List[SessionSpec] = []
    self.test_sessions: List[SessionSpec] = []

  async def build(self, context: str):
    _route_context = await self._engineer.generate_data(route_context=context)
    if _route_context is None:
      raise ValueError("No data generated")

    self.route_context = _route_context
    self.train_sessions, self.test_sessions = train_test_split(_route_context.sessions, test_size=0.2, random_state=1)

    self.engine.train(self.route_context.routes, self.train_sessions)

  def _predict_query(self, query: str):
    start_time = time.time()
    results = self.engine.predict(query)
    return results, time.time() - start_time

  def evaluate(self):
    if self.test_sessions is None:
      raise ValueError("Test sessions not found, run build() first.")

    print(f"=== {self.engine.__class__.__name__} ===")

    predicted = []
    expected = []
    inference_times = []
    for session in self.test_sessions:
      query = session.context
      route_scores, latency = self._predict_query(query)

      if route := route_scores[0]:
        predicted.append(route.id)
        expected.append(session.target)
        inference_times.append(latency)

    accuracy = accuracy_score(expected, predicted)
    print(f"Accuracy: {accuracy:.2f}")

    precision = precision_score(expected, predicted, average="weighted", zero_division=0)
    print(f"Precision: {precision:.2f}")

    recall = recall_score(expected, predicted, average="weighted", zero_division=0)
    print(f"Recall: {recall:.2f}")

    f1_metric = f1_score(expected, predicted, average="weighted", zero_division=0)
    print(f"F1-Score: {f1_metric:.2f}")

    avg_inference_time = np.mean(inference_times)
    print(f"Inference time: {avg_inference_time:.4f} s\n")

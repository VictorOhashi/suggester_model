import time
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, ndcg_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from notebooks.engines.sailor_data_engineer import SailorDataEngineer
from sailor import VectorSailorEngine, SessionSpec
from sailor.types import RouteContextResult

RoutePrediction = List[Tuple[str, List[RouteContextResult]]]

class VectorTestEngine:
  def __init__(self, engine: VectorSailorEngine, engineer: SailorDataEngineer):
      self._engineer = engineer
      self.engine = engine
      self.test_sessions: Optional[List[SessionSpec]] = None

  async def build(self, route_count=20, session_count=100) -> None:
      route_context = await self._engineer.generate_data(route_count, session_count)
      if route_context is None:
        raise ValueError("No data generated")

      train_sessions, self.test_sessions = train_test_split(route_context.sessions, test_size=0.2, random_state=14)

      self.engine.train(route_context.routes, train_sessions)

  def _predict_query(self, query: str):
      start_time = time.time()
      results = self.engine.predict(query)
      return results, time.time() - start_time

  def evaluate(self, top_k: int = 5):
      if self.test_sessions is None:
        raise ValueError("Test sessions not found, run build() first.")

      print(f"=== {self.engine.__class__.__name__} ===")

      prediction: RoutePrediction = []
      inference_times: List[float] = []

      for session in self.test_sessions:
        route_scores, latency = self._predict_query(session.context)
        prediction.append((session.target, route_scores))
        inference_times.append(latency)

      self._evaluate_prediction(prediction)
      self._evaluate_k_prediction(prediction, k=top_k)

      avg_inference_time = np.mean(inference_times)
      print(f"Inference time: {avg_inference_time:.4f} s\n")

  def _evaluate_prediction(self, prediction: RoutePrediction):
      expected = [t for (t, _) in prediction]
      predicted = [routes[0].id for (_, routes) in prediction]

      accuracy = accuracy_score(expected, predicted)
      print(f"Accuracy: {accuracy:.2f}")

      precision = precision_score(expected, predicted, average="weighted")
      print(f"Precision: {precision:.2f}")

      recall = recall_score(expected, predicted, average="weighted")
      print(f"Recall: {recall:.2f}")

      f1_metric = f1_score(expected, predicted, average="weighted")
      print(f"F1-Score: {f1_metric:.2f}")

  def _evaluate_k_prediction(self, prediction: RoutePrediction, k: int):
      labels = list(self.engine.documentor.labels_)
      expected = [t for (t, _) in prediction]
      expected = np.array(expected)

      predicted = []
      for (_, routes) in prediction:
        route_scores = {route.id: route.score for route in routes}
        sample_scores = [route_scores.get(label, 0.0) for label in labels]
        predicted.append(sample_scores)
      predicted = np.array(predicted)

      top_k_accuracy = top_k_accuracy_score(expected, predicted, k=k, labels=labels)
      print(f"Top-{k} Accuracy: {top_k_accuracy:.2f}")

      expected_ordened = np.zeros((len(expected), len(labels)))
      for i, target in enumerate(expected):
        if target in labels:
            j = labels.index(target)
            expected_ordened[i, j] = 1

      ndcg = ndcg_score(expected_ordened, predicted, k=k)
      print(f"NDCG-{k}: {ndcg:.2f}")

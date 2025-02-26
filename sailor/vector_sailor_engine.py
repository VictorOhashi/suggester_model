import numpy as np
from typing import List

from sklearn.svm import NuSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sailor.route_vectorizer import RouteContext
from .route_specs import RouteSpec, SessionSpec
from .sailor_engine import VectorSailorEngine

class TfidfSailorEngine(VectorSailorEngine):
  def predict(self, query: str):
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = cosine_similarity(query_vec, self.vectorizer.route_vectors).flatten()
    return self.scored_routes(scores)

class SVCSailorEngine(VectorSailorEngine):
  def __init__(self):
    super().__init__()
    self.model = NuSVC(class_weight="balanced", max_iter=2000, probability=True)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    route_vectors, labels = super().train(routes, sessions)
    self.model.fit(route_vectors, labels)

  def predict(self, query: str):
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = self.model.predict_proba(query_vec)[0]
    return self.scored_routes(scores)

class KNNSailorEngine(VectorSailorEngine):
  def __init__(self):
    super().__init__()
    self.model = KNeighborsClassifier(weights='distance', algorithm='brute', n_neighbors=5)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    route_vectors, labels = super().train(routes, sessions)
    self.model.fit(route_vectors, labels)

  def predict(self, query: str):
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = self.model.predict_proba(query_vec)[0]
    return self.scored_routes(scores)


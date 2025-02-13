from ast import Tuple
import numpy as np
from typing import List

from sklearn.calibration import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sailor.route_vectorizer import RouteContext, RouteVectorizer
from .route_specs import NavigationContext, RouteSpec, SessionSpec
from .sailor_engine import SailorEngine

class VectorSailorEngine(SailorEngine):
  def __init__(self, routes: List[RouteSpec]):
    super().__init__(routes)
    self.vectorizer = RouteVectorizer()

  def train(self, sessions: List[SessionSpec]):
    self.train_context = NavigationContext(routes=self.routes, sessions=sessions)
    return self.vectorizer.fit(self.train_context)

  def predict(self, query: str) -> List[tuple[RouteContext, float]]: ...

class TfidfSailorEngine(VectorSailorEngine):
  def predict(self, query: str) -> List[tuple[RouteContext, float]]:
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = cosine_similarity(query_vec, self.vectorizer.route_vectors).flatten()
    sorted_indices = np.argsort(scores)[::-1]

    scored_routes: List[tuple[RouteContext, float]] = []
    for i in sorted_indices:
        route = self.vectorizer.inverse_transform(i)
        if route is not None:
          scored_routes.append((route, float(scores[i])))

    return scored_routes

class SVCSailorEngine(VectorSailorEngine):
  def __init__(self, routes: List[RouteSpec]):
    super().__init__(routes)
    self.model = LinearSVC(class_weight="balanced", max_iter=2000)

  def train(self, sessions: List[SessionSpec]):
    route_vectors, labels = super().train(sessions)
    self.model.fit(route_vectors, labels)

  def predict(self, query: str) -> List[tuple[RouteContext, float]]:
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = self.model.decision_function(query_vec)[0]
    scores = 1 / (1 + np.exp(-scores))

    sorted_indices = np.argsort(scores)[::-1]

    scored_routes: List[tuple[RouteContext, float]] = []
    for i in sorted_indices:
        route = self.vectorizer.inverse_transform(i)
        if route is not None:
          scored_routes.append((route, float(scores[i])))

    return scored_routes

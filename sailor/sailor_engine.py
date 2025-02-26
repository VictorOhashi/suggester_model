import numpy as np
from typing import List, Optional

from .route_vectorizer import RouteContext, RouteVectorizer
from .route_specs import NavigationContext, RouteSpec, SessionSpec

class SailorEngine:
  def __init__(self):
    self.train_context: Optional[NavigationContext] = None

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]): ...

  def predict(self, query: str): ...

class VectorSailorEngine(SailorEngine):
  def __init__(self):
    super().__init__()
    self.vectorizer = RouteVectorizer()

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    self.train_context = NavigationContext(routes=routes, sessions=sessions)
    return self.vectorizer.fit(self.train_context)

  def predict(self, query: str) -> List[tuple[RouteContext, float]]: ...

  def scored_routes(self, scores) -> List[tuple[RouteContext, float]]:
    sorted_index = np.argsort(scores)[::-1]
    scored_routes: List[tuple[RouteContext, float]] = []
    for i in sorted_index:
      route = self.vectorizer.inverse_transform(i)
      if route is not None:
        scored_routes.append((route, float(scores[i])))
    return scored_routes

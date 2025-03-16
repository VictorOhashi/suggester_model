from typing import List

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from .route_vectorizer import RouteVectorizer
from .route_documentor import RouteDocumentor
from .types import SailorEngine, RouteSpec, SessionSpec, NavigationContext, RouteContextResult

class VectorSailorEngine(SailorEngine):
  def __init__(self, vectorizer: RouteVectorizer):
    super().__init__()
    self._vectorizer = vectorizer
    self.documentor: RouteDocumentor

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    self.train_context = NavigationContext(routes=routes, sessions=sessions)
    self.documentor = RouteDocumentor(self.train_context)

    labels, _ = self.documentor.fit_transform()
    vectors = self._vectorizer.fit_transform(self.documentor.documents)

    return vectors, labels

  def transform(self, query: str):
      parsed_query = query.lower().strip()
      if not parsed_query:
          return None
      return self._vectorizer.transform(query=parsed_query)

  def scored_routes(self, scores) -> List[RouteContextResult]:
    sorted_index = np.argsort(scores)[::-1]
    scored_routes: List[RouteContextResult] = []
    for i in sorted_index:
      route = self.documentor.inverse_transform(i)
      if route is not None:
        route = route.copy_with_score(float(scores[i]))
        scored_routes.append(route)
    return scored_routes

class SVCSailorEngine(VectorSailorEngine):
  def __init__(self, vectorizer: RouteVectorizer):
    super().__init__(vectorizer)
    self.model = LinearSVC(class_weight="balanced", max_iter=2000)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    vectors, labels = super().train(routes, sessions)
    self.model.fit(vectors, labels)

  def predict(self, query: str):
    query_vec = self.transform(query)
    if query_vec is None:
        return []

    scores = self.model.decision_function(query_vec)[0]
    return self.scored_routes(scores)

class KNNSailorEngine(VectorSailorEngine):
  def __init__(self, vectorizer: RouteVectorizer, n_neighbors: int = 2):
    super().__init__(vectorizer=vectorizer)
    self.model = KNeighborsClassifier(weights='distance', n_neighbors=n_neighbors)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    vectors, labels = super().train(routes, sessions)
    self.model.fit(vectors, labels)

  def predict(self, query: str):
    query_vec = self.transform(query)
    if query_vec is None:
        return []

    scores = self.model.predict_proba(query_vec)[0]
    return self.scored_routes(scores)


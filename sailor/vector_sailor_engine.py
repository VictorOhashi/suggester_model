from typing import List, Optional

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .route_documentor import RouteDocumentor
from .types import SailorEngine, RouteSpec, SessionSpec, NavigationContext, RouteContextResult

class VectorSailorEngine(SailorEngine):
  def __init__(self, vectorizer: Optional[TfidfVectorizer] = None):
    super().__init__()
    self._vectorizer = vectorizer or TfidfVectorizer(max_features=1000, min_df=1, max_df=0.8)
    self.documentor: RouteDocumentor

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    self.train_context = NavigationContext(routes=routes, sessions=sessions)
    self.documentor = RouteDocumentor(self.train_context)

    labels, _ = self.documentor.fit()
    vectors = self._vectorizer.fit_transform(self.documentor.documents)

    return vectors, labels

  def predict_transform(self, query: str):
      parsed_query = query.lower().strip()
      if not parsed_query:
          return None
      return self._vectorizer.transform([parsed_query])

  def scored_routes(self, scores) -> List[RouteContextResult]:
    sorted_index = np.argsort(scores)[::-1]
    scored_routes: List[RouteContextResult] = []
    for i in sorted_index:
      route = self.documentor.inverse_transform(i)
      if route is not None:
        route = route.copy_with_score(float(scores[i]))
        scored_routes.append(route)
    return scored_routes

class TfidfSailorEngine(VectorSailorEngine):
  def __init__(self):
    super().__init__()
    self._route_vectors = None

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    vectors, labels = super().train(routes, sessions)
    self._route_vectors = vectors
    return vectors, labels

  def predict(self, query: str):
    query_vec = self.predict_transform(query)
    if query_vec is None:
        return []

    scores = cosine_similarity(query_vec, self._route_vectors).flatten()
    return self.scored_routes(scores)

class SVCSailorEngine(VectorSailorEngine):
  def __init__(self):
    super().__init__()
    self.model = LinearSVC(class_weight="balanced", max_iter=2000)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    vectors, labels = super().train(routes, sessions)
    self.model.fit(vectors, labels)

  def predict(self, query: str):
    query_vec = self.predict_transform(query)
    if query_vec is None:
        return []

    scores = self.model.decision_function(query_vec)[0]
    return self.scored_routes(scores)

class KNNSailorEngine(VectorSailorEngine):
  def __init__(self, n_neighbors: int = 2):
    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.8)
    super().__init__(vectorizer=vectorizer)
    self.model = KNeighborsClassifier(weights='distance', n_neighbors=n_neighbors)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    vectors, labels = super().train(routes, sessions)
    self.model.fit(vectors, labels)

  def predict(self, query: str):
    query_vec = self.predict_transform(query)
    if query_vec is None:
        return []

    scores = self.model.predict_proba(query_vec)[0]
    return self.scored_routes(scores)


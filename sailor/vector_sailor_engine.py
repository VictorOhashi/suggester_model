from typing import List

from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

from sailor.route_vectorizer import RouteVectorizer
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
    self.model = LinearSVC(class_weight="balanced", max_iter=2000)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    route_vectors, labels = super().train(routes, sessions)
    self.model.fit(route_vectors, labels)

  def predict(self, query: str):
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = self.model.decision_function(query_vec)[0]
    return self.scored_routes(scores)

class KNNSailorEngine(VectorSailorEngine):
  def __init__(self, n_neighbors: int = 2):
    vectorizer = RouteVectorizer(min_df=1)
    super().__init__(vectorizer=vectorizer)
    self.model = KNeighborsClassifier(weights='distance', n_neighbors=n_neighbors)

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
    route_vectors, labels = super().train(routes, sessions)
    self.model.fit(route_vectors, labels)

  def predict(self, query: str):
    query_vec = self.vectorizer.transform(query)
    if query_vec is None:
        return []

    scores = self.model.predict_proba(query_vec)[0]
    return self.scored_routes(scores)


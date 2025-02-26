import time
from typing import List, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sailor import VectorSailorEngine, NavigationContext, SessionSpec, RouteGenConfig, SailorDataEngineer

class VectorTestEngine:
  def __init__(self, engine: VectorSailorEngine):
    _config = RouteGenConfig.fromEnv()
    self._engineer = SailorDataEngineer(_config)
    self.engine = engine
    self.route_context: Optional[NavigationContext] = None
    self.train_sessions: List[SessionSpec] = []
    self.test_sessions: List[SessionSpec] = []

  async def build(self, context: str, cache_key: str):
    _route_context = await self._engineer.generate_data(route_context=context, cache_key=cache_key)
    if _route_context is None:
      raise ValueError("No data generated")

    self.route_context = _route_context
    self.train_sessions, self.test_sessions = train_test_split(_route_context.sessions, test_size=0.2, random_state=1)

    self.engine.train(self.route_context.routes, self.train_sessions)

  def _test_query(self, query: str):
    start_time = time.time()
    results = self.engine.predict(query)
    latency = (time.time() - start_time)*1000
    print(f"Results ({latency:.2f}ms):")

    for route, score in results[:5]:
        print(f"- {route.path} (score: {score:.3f})")

  def test(self):
    if self.route_context is None:
      raise ValueError("No route context found")

    for session in self.test_sessions:
      for route in self.route_context.routes:
        if route.id == session.route_id:
            break

      query = session.intention.context
      print(f"Query: '{query}'; Expected route: {route.path};")
      self._test_query(query)

  def analyze(self):
    if self.route_context is None:
      raise ValueError("No route context found")

    route_vectors = self.engine.vectorizer.route_vectors
    if route_vectors is None:
      raise ValueError("No route vectors found")

    similarity_matrix = cosine_similarity(route_vectors)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="Blues")
    plt.colorbar(label="Similarity Score")
    plt.title("Route Similarity Matrix")

    test_routes = [r.path for r in self.route_context.routes if r in self.test_sessions]

    plt.xticks(ticks=range(len(test_routes)), labels=test_routes, rotation=90)
    plt.yticks(ticks=range(len(test_routes)), labels=test_routes)

    plt.show()

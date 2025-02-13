from typing import List, Optional

from .route_vectorizer import RouteVectorizer
from .route_specs import NavigationContext, RouteSpec, SessionSpec

class SailorEngine:
  def __init__(self, routes: List[RouteSpec]):
    self.routes = routes
    self.train_context: Optional[NavigationContext] = None

  def train(self, sessions: List[SessionSpec]): ...

  def predict(self, query: str): ...


from typing import List, Optional

from .route_context import NavigationContext, RouteContextResult
from .route_specs import RouteSpec, SessionSpec

class SailorEngine:
  def __init__(self):
    self.train_context: Optional[NavigationContext] = None

  def train(self, routes: List[RouteSpec], sessions: List[SessionSpec]): ...

  def predict(self, query: str) -> List[RouteContextResult]: ...

from typing import List
from pydantic import BaseModel, Field
from .route_specs import RouteSpec, SessionSpec

class NavigationContext(BaseModel):
    routes: List[RouteSpec] = Field(..., description="Routes")
    sessions: List[SessionSpec] = Field(..., description="Sessions")

class RouteContext(BaseModel):
    id: str = Field(..., description="Route ID")
    path: str = Field(..., description="Route path")
    context: str = Field(..., description="Route merged context")

    def copy_with_score(self, score: float) -> 'RouteContextResult':
        return RouteContextResult(
            id=self.id,
            path=self.path,
            context=self.context,
            score=score
        )

class RouteContextResult(RouteContext):
    score: float = Field(..., description="Route score")

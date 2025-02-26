import string
from typing import List
from pydantic import BaseModel, Field
from .route_specs import RouteSpec, SessionSpec, NavigationContext

class RouteContext(BaseModel):
    id: str = Field(..., description="Route ID")
    path: str = Field(..., description="Route path")
    context: str = Field(..., description="Route merged context")

    @classmethod
    def from_route_spec(cls, route: RouteSpec) -> 'RouteContext':
        context: List[str] = []

        for path in route.path.split('/'):
            if path not in string.punctuation:
                context.append(path)

        for tag in route.tags:
            context.append(tag)

        return RouteContext(
            id=route.id,
            path=route.path,
            context=' '.join(context)
        )

    def copy_with_session(self, session: SessionSpec):
        self.context = f"{self.context} {session.context}"

    def copy_with_score(self, score: float) -> 'RouteContextResult':
        return RouteContextResult(
            id=self.id,
            path=self.path,
            context=self.context,
            score=score
        )

class RouteContextResult(RouteContext):
    score: float = Field(..., description="Route score")

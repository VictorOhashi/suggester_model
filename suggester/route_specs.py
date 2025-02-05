from pydantic import BaseModel, Field

from typing import List

class RouteSpec(BaseModel):
    id: str = Field(..., description="Route ID (format UUID)")
    path: str = Field(..., description="Route path (hierarchical route, ex: '/pedidos/historico')")
    tags: List[str] = Field(..., description="Route tags (related to the actual function)")
    last_date: str = Field(..., description="Last date of the route (format YYYY-MM-DD)")

class SessionSpec(BaseModel):
    id: str = Field(..., description="Session ID (format UUID)")
    route_id: str = Field(..., description="Route ID (reference to routes[].id)")
    last_date: str = Field(..., description="Last date of the session (format YYYY-MM-DD)")
    time_spent: int = Field(..., description="Time spent in the session (5-60 minutes in milliseconds)")
    intention_context: str = Field(..., description="User intention")

class NavigationContext(BaseModel):
    routes: List[RouteSpec] = Field(..., description="Routes")
    sessions: List[SessionSpec] = Field(..., description="Sessions")

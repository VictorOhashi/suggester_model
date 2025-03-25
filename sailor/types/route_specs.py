from typing import List
from pydantic import BaseModel, Field

class RouteSpec(BaseModel):
    id: str = Field(..., description="Route ID")
    path: str = Field(..., description="Route path")
    tags: List[str] = Field(..., description="Route tags")


class SessionSpec(BaseModel):
    id: str = Field(..., description="Session ID")
    route_id: str = Field(..., description="Route ID")
    context: str = Field(..., description="User intention context")

    @property
    def target(self):
        return self.route_id

class RouteResponse(BaseModel):
    routes: List[RouteSpec]

class SessionResponse(BaseModel):
    sessions: List[SessionSpec]

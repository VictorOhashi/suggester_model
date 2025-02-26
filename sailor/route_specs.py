from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class RouteSpec(BaseModel):
    id: str = Field(..., description="Route ID")
    path: str = Field(..., description="Route path")
    tags: List[str] = Field(..., description="Route tags")

class SessionIntentType(str, Enum):
    SEARCH = "search"
    NAVIGATION = "navigation"

class SessionIntentSpec(BaseModel):
    type: SessionIntentType = Field(..., description="Intent type")
    context: str = Field(..., description="User intention context")

class SessionSpec(BaseModel):
    id: str = Field(..., description="Session ID")
    route_id: str = Field(..., description="Route ID")
    last_date: str = Field(..., description="Last date that the route was accessed (yyyy-mm-dd)")
    intention: SessionIntentSpec = Field(..., description="User intention")

    @property
    def context(self):
        return self.intention.context

    @property
    def target(self):
        return self.route_id

class NavigationContext(BaseModel):
    routes: List[RouteSpec] = Field(..., description="Routes")
    sessions: List[SessionSpec] = Field(..., description="Sessions")

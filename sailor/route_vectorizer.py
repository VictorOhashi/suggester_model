import string
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from .route_specs import RouteSpec, SessionSpec, NavigationContext

class RouteContext(BaseModel):
    id: str = Field(..., description="Route ID")
    path: str = Field(..., description="Route path")
    context: str = Field(..., description="Route merged context")

    @classmethod
    def _from_route_spec(cls, route: RouteSpec) -> 'RouteContext':
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

    def _bind_session(self, session: SessionSpec):
        self.context = f"{self.context} {session.intention.context}"

class RouteVectorizer:
    def __init__(self, max_features: int = 1000):
      self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
      self.route_vectors = None
      self.label_encoder = LabelEncoder()
      self.label_encoded = None
      self._routes_cache: Dict[str, RouteContext] = {}

    def fit(self, navigation_context: NavigationContext):
        routes = []
        sessions_cache = navigation_context.sessions.copy()
        for route in navigation_context.routes:
            route_context = RouteContext._from_route_spec(route)

            for i, session in enumerate(sessions_cache):
                if session.route_id == route_context.id:
                    sessions_cache.pop(i)
                    route_context._bind_session(session)

            self._routes_cache.update({route_context.id: route_context})
            routes.append(route_context)

        self.route_vectors = self._vectorizer.fit_transform([route.context for route in routes])
        self.label_encoded = self.label_encoder.fit_transform([route.id for route in routes])

        return self.route_vectors, self.label_encoded

    def transform(self, query: str):
        parsed_query = query.lower().strip()
        if not parsed_query:
            return None

        return self._vectorizer.transform([parsed_query])

    def inverse_transform(self, label: int) -> Optional[RouteContext]:
        route_id = self.label_encoder.inverse_transform([label])[0]
        return self._routes_cache.get(route_id)

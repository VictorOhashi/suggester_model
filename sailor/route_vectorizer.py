from typing import Dict, Optional
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from .route_context import RouteContext, NavigationContext

class RouteVectorizer:
    def __init__(self, min_df: int = 2, max_df: float = 0.8, max_features: int = 1000):
      self._vectorizer = TfidfVectorizer(
          max_features=max_features,
          min_df=min_df,
          max_df=max_df)
      self.route_vectors = None
      self.label_encoder = LabelEncoder()
      self.label_encoded = None
      self._routes_cache: Dict[str, RouteContext] = {}

    def fit(self, navigation_context: NavigationContext):
        routes = []
        sessions_cache = navigation_context.sessions.copy()
        for route in navigation_context.routes:
            route_context = RouteContext.from_route_spec(route)

            for i, session in enumerate(sessions_cache):
                if session.target == route_context.id:
                    sessions_cache.pop(i)
                    route_context.copy_with_session(session)

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

    @property
    def labels(self):
        return self.label_encoder.classes_

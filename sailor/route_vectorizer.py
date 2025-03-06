import string
from typing import Dict, List, Optional
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from sailor.route_specs import RouteSpec, SessionSpec
from .route_context import RouteContext, NavigationContext

class RouteVectorizer:
    def __init__(self, min_df: int = 2, max_df: float = 0.8, max_features: int = 1000):
      self._tokenizer = spacy.load("en_core_web_lg")
      self._vectorizer = TfidfVectorizer(
          max_features=max_features,
          min_df=min_df,
          max_df=max_df)
      self.route_vectors = None
      self.label_encoder = LabelEncoder()
      self.label_encoded = None
      self._routes_cache: Dict[str, RouteContext] = {}

    def parse_route(self, route: RouteSpec, sessions: List[SessionSpec]) -> RouteContext:
        context: List[str] = []

        for path in route.path.split('/'):
            if path not in string.punctuation:
                context.append(path)

        for tag in route.tags:
            context.append(tag)

        session_context: List[str] = []
        for i, s in enumerate(sessions):
            if s.target == route.id:
                session_context.append(s.context)
                sessions.pop(i)

        docs = self._tokenizer.pipe(session_context)
        session_tokens = [t.text for d in docs for t in d if not t.is_stop and t.is_alpha]
        context.append(" ".join(session_tokens))

        route_context = RouteContext(id=route.id, path=route.path, context=" ".join(context))
        self._routes_cache.update({route_context.id: route_context})

        return route_context

    def fit(self, navigation_context: NavigationContext):
        sessions = navigation_context.sessions.copy()
        routes = [self.parse_route(r, sessions) for r in navigation_context.routes]

        self.route_vectors = self._vectorizer.fit_transform([r.context for r in routes])
        self.label_encoded = self.label_encoder.fit_transform([r.id for r in routes])

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

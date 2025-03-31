import spacy
import string
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from sailor.types import RouteSpec, SessionSpec, RouteContext

class RouteDocumentor:
    def __init__(self):
        self._label_encoder = LabelEncoder()
        self._tokenizer = spacy.load("en_core_web_lg")
        self._routes: Dict[str, RouteContext]

    @property
    def labels_(self):
        return self._label_encoder.classes_

    @property
    def documents(self):
        return [r.context for r in self._routes.values()]

    def get_route(self, route_id: str) -> RouteContext:
        route = self._routes.get(route_id)
        if route is None:
            raise ValueError(f"Route with id {route_id} not found.")
        return route

    def fit_transform(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
        self._fit(routes, sessions)
        route_labels = list(self._routes.keys())
        labels = self._label_encoder.fit_transform(route_labels)
        return np.array(labels)

    def _fit(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
        sessions = sessions.copy()
        parsed_routes = [self._parse_route(r, sessions) for r in routes]
        self._routes = {r.id: r for r in parsed_routes}

    def _parse_route(self, route: RouteSpec, sessions: List[SessionSpec]) -> RouteContext:
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

        return RouteContext(id=route.id, path=route.path, context=" ".join(context))

    def transform(self, labels: list[str]):
        return self._label_encoder.transform(labels)

    def inverse_transform(self, label: int) -> RouteContext:
        route_id = self._label_encoder.inverse_transform([label])[0]
        return self.get_route(route_id)

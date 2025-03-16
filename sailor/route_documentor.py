import spacy
import string
from typing import Dict, List, Optional
from sklearn.calibration import LabelEncoder

from sailor.types import RouteSpec, SessionSpec, NavigationContext, RouteContext

class RouteDocumentor:
    def __init__(self, navigation_context: NavigationContext):
        self._tokenizer = spacy.load("en_core_web_lg")
        self.label_encoder = LabelEncoder()

        self._routes: Dict[str, RouteContext] = self._prepare(navigation_context)
        self._labels = list(self._routes.keys())

        self.documents = [r.context for r in self._routes.values()]

    @property
    def labels_(self):
        return self.label_encoder.classes_

    def _prepare(self, navigation_context: NavigationContext):
        sessions = navigation_context.sessions.copy()
        routes = [self._prepare_route(r, sessions) for r in navigation_context.routes]
        return {r.id: r for r in routes}

    def _prepare_route(self, route: RouteSpec, sessions: List[SessionSpec]) -> RouteContext:
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

    def fit_transform(self):
         labels = self.label_encoder.fit_transform(self._labels)
         return labels, self.labels_

    def inverse_transform(self, label: int) -> Optional[RouteContext]:
        route_id = self.label_encoder.inverse_transform([label])[0]
        return self._routes.get(route_id)

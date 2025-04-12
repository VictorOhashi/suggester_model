from abc import ABC, abstractmethod
import os
import pickle
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from .route_documentor import RouteDocumentor
from .types import  RouteSpec, SessionSpec, RouteContextResult

class SailorEngine(ABC):
    def __init__(self):
        super().__init__()
        self.documentor = RouteDocumentor()
        self.pipeline: Pipeline

    def fit(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
        labels = self.documentor.fit_transform(routes, sessions)
        return self.pipeline.fit(self.documentor.documents, labels)

    @abstractmethod
    def predict(self, query: str) -> List[RouteContextResult]: ...

    def scored_routes(self, scores) -> List[RouteContextResult]:
        sorted_index = np.argsort(scores)[::-1]
        scored_routes: List[RouteContextResult] = []
        for i in sorted_index:
            route = self.documentor.inverse_transform(i)
            if route is not None:
                score = float(scores[i])
                route = route.copy_with_score(score)
                scored_routes.append(route)
        return scored_routes

    def save_model(self, model_name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        return model_path

class SVCSailorEngine(SailorEngine):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('svc', LinearSVC(class_weight='balanced', max_iter=2000)),
            ])

    def predict(self, query: str):
        if query is None: return []
        scores = self.pipeline.decision_function([query])[0]
        return self.scored_routes(scores)

class KNNSailorEngine(SailorEngine):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('knn', KNeighborsClassifier(weights='distance')),
            ])

    def predict(self, query: str):
        if query is None: return []
        scores = self.pipeline.predict_proba([query])[0]
        return self.scored_routes(scores)


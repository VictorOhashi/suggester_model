import os
import pickle
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from .route_documentor import RouteDocumentor
from .types import  RouteSpec, SessionSpec, RouteContextResult

class SailorEngine:
    def __init__(self):
        super().__init__()
        self.documentor = RouteDocumentor()
        self.pipeline: Pipeline

    def fit(self, routes: List[RouteSpec], sessions: List[SessionSpec]):
        labels = self.documentor.fit_transform(routes, sessions)
        return self.pipeline.fit(self.documentor.documents, labels)

    def validate(self, sessions: List[SessionSpec]):
        test_queries = [s.context for s in sessions]
        predictions = self.pipeline.predict(test_queries)

        targets = self.documentor.transform([s.target for s in sessions])
        target_names = [
            self.documentor.get_route(id).path
            for id in self.documentor.labels_
        ]
        return classification_report(targets, predictions, target_names=target_names)

    def scored_routes(self, scores) -> List[RouteContextResult]:
        sorted_index = np.argsort(scores)[::-1]
        scored_routes: List[RouteContextResult] = []
        for i in sorted_index:
            route = self.documentor.inverse_transform(i)
            if route is not None:
                route = route.copy_with_score(float(scores[i]))
                scored_routes.append(route)
        return scored_routes

    def save_model(self, model_name: str, model_dir: str):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)


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


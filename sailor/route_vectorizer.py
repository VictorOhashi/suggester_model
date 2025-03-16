from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class RouteVectorizer:
  def fit_transform(self, documents: List[str]) -> np.ndarray: ...

  def transform(self, query: str) -> np.ndarray: ...

class TfidfRouteVectorizer(RouteVectorizer):
  def __init__(self):
      super().__init__()
      self._vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.8)

  def fit_transform(self, documents: List[str]):
      return self._vectorizer.fit_transform(documents)

  def transform(self, query: str):
      return self._vectorizer.transform([query])

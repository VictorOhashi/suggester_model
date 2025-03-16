"""
Search Suggestion: AI-powered search suggestion.
"""

from .vector_sailor_engine import VectorSailorEngine, SVCSailorEngine, KNNSailorEngine
from .types import NavigationContext, RouteSpec, SessionSpec
from .route_documentor import RouteDocumentor
from .route_vectorizer import TfidfRouteVectorizer

__version__ = "0.0.1"

__all__ = [
    "VectorSailorEngine",
    "SVCSailorEngine",
    "KNNSailorEngine",
    "NavigationContext",
    "RouteSpec",
    "SessionSpec",
    "RouteDocumentor",
    "TfidfRouteVectorizer",
]

"""
Search Suggestion: AI-powered search suggestion.
"""

from .vector_sailor_engine import VectorSailorEngine, TfidfSailorEngine, SVCSailorEngine, KNNSailorEngine
from .types import NavigationContext, RouteSpec, SessionSpec
from .route_documentor import RouteDocumentor

__version__ = "0.0.1"

__all__ = [
    "VectorSailorEngine",
    "TfidfSailorEngine",
    "SVCSailorEngine",
    "KNNSailorEngine",
    "NavigationContext",
    "RouteSpec",
    "SessionSpec",
    "RouteDocumentor",
]

"""
Search Suggestion: AI-powered search suggestion.
"""

from .sailor_engine import VectorSailorEngine
from .vector_sailor_engine import TfidfSailorEngine, SVCSailorEngine, KNNSailorEngine
from .route_context import NavigationContext
from .route_specs import RouteSpec, SessionSpec

__version__ = "0.0.1"

__all__ = [
    "RouteSpec",
    "SessionSpec",
    "NavigationContext",
    "VectorSailorEngine",
    "TfidfSailorEngine",
    "SVCSailorEngine",
    "KNNSailorEngine",
]

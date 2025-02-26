"""
Search Suggestion: AI-powered search suggestion.
"""

from .sailor_data_engineer import SailorDataEngineer, RouteGenConfig
from .route_specs import RouteSpec, SessionSpec, NavigationContext
from .sailor_engine import VectorSailorEngine
from .vector_sailor_engine import TfidfSailorEngine, SVCSailorEngine, KNNSailorEngine

__version__ = "0.0.1"

__all__ = [
    "SailorDataEngineer",
    "RouteGenConfig",
    "RouteSpec",
    "SessionSpec",
    "NavigationContext",
    "VectorSailorEngine",
    "TfidfSailorEngine",
    "SVCSailorEngine",
    "KNNSailorEngine",
]

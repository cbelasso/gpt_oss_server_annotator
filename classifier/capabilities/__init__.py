"""
Capabilities system for extensible text classification.

This package provides a plugin-like architecture for adding new features
to the classification system. Each capability is a self-contained module
that can declare dependencies and be composed with other capabilities.
"""

from .alerts import AlertsCapability
from .base import Capability
from .classification import ClassificationCapability
from .orchestrator import CapabilityOrchestrator
from .recommendations import RecommendationsCapability
from .registry import CapabilityRegistry, create_default_registry
from .stem_polarity import StemPolarityCapability
from .stem_recommendations import StemRecommendationsCapability
from .stem_trend import StemTrendCapability
from .trend import TrendCapability

__all__ = [
    "Capability",
    "CapabilityRegistry",
    "CapabilityOrchestrator",
    "create_default_registry",
    # Standard capabilities
    "ClassificationCapability",
    "RecommendationsCapability",
    "AlertsCapability",
    "StemRecommendationsCapability",
    "StemPolarityCapability",
    "TrendCapability",
    "StemTrendCapability",
]

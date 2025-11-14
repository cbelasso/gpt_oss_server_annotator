"""
Recommendation detection capability.

Detects and extracts actionable recommendations from text.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..models import RecommendationsOutput
from ..prompts import recommendations_detection_prompt
from .base import Capability


class RecommendationsCapability(Capability):
    """
    Detects recommendations in text.

    Identifies actionable suggestions, advice, proposals, or requests for change
    and classifies them by direction (add, reduce, modify, etc.).
    """

    @property
    def name(self) -> str:
        return "recommendations"

    @property
    def schema(self) -> Type[BaseModel]:
        return RecommendationsOutput

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        return recommendations_detection_prompt(text)

    def format_for_export(self, result: Any) -> Any:
        """Extract just the recommendations list."""
        if result is None:
            return []

        formatted = super().format_for_export(result)

        if isinstance(formatted, dict):
            return formatted.get("recommendations", [])

        return []

"""
Global trend detection capability.

Detects temporal change patterns across entire text without dependency on classification.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..models import TrendsOutput
from ..prompts import trends_detection_prompt
from .base import Capability


class TrendCapability(Capability):
    """
    Detects temporal change patterns in text.

    Identifies whether the text describes temporal changes - such as improvements,
    declines, stability, or fluctuations over time. Can detect multiple distinct
    trends if the comment discusses multiple aspects changing over time.

    This is a global capability that analyzes the entire text without requiring
    classification results.
    """

    @property
    def name(self) -> str:
        return "trend"

    @property
    def schema(self) -> Type[BaseModel]:
        return TrendsOutput

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        return trends_detection_prompt(text)

    def get_result_key(self) -> str:
        """Store under 'trends' key."""
        return "trends"

    def format_for_export(self, result: Any) -> Any:
        """Format trend output for export."""
        if result is None:
            return {"has_trends": False, "trends": []}

        # Convert Pydantic model to dict
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "dict"):
            return result.dict()

        return result

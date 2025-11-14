"""
Hierarchical classification capability.

Wraps the existing HierarchicalClassifier as a capability for consistency.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..models import ClassificationOutput
from .base import Capability


class ClassificationCapability(Capability):
    """
    Performs hierarchical topic classification.

    This is a special capability that uses the existing HierarchicalClassifier
    and requires special handling in the orchestrator.
    """

    @property
    def name(self) -> str:
        return "classification"

    @property
    def schema(self) -> Type[BaseModel]:
        return ClassificationOutput

    def requires_hierarchy(self) -> bool:
        return True

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Classification uses its own internal prompt generation.
        This method should not be called directly.
        """
        raise NotImplementedError(
            "ClassificationCapability uses HierarchicalClassifier internally"
        )

    def get_result_key(self) -> str:
        """Store under 'classification_result' key."""
        return "classification_result"

    def format_for_export(self, result: Any) -> Any:
        """Format classification output for export."""
        if result is None:
            return None

        # Convert Pydantic model to dict
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "dict"):
            return result.dict()

        return result

"""
Base capability interface for extensible classification features.

This module provides the abstract base class that all capabilities must implement,
enabling a plugin-like architecture for adding new features.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from pydantic import BaseModel


class Capability(ABC):
    """
    Abstract base class for classification capabilities.

    A capability represents a discrete feature (e.g., classification, recommendations,
    alerts) that can process texts and produce structured outputs.

    Capabilities can declare dependencies on other capabilities, enabling complex
    workflows where one capability's output feeds into another.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this capability."""
        pass

    @property
    @abstractmethod
    def schema(self) -> Type[BaseModel]:
        """Pydantic schema for this capability's output."""
        pass

    @property
    def dependencies(self) -> List[str]:
        """
        List of capability names that must run before this one.

        Returns:
            List of capability names (empty list if no dependencies)
        """
        return []

    @abstractmethod
    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a prompt for processing the given text.

        Args:
            text: The text to process
            context: Optional context from dependent capabilities

        Returns:
            Formatted prompt string
        """
        pass

    def requires_hierarchy(self) -> bool:
        """
        Whether this capability requires access to the topic hierarchy.

        Returns:
            True if hierarchy is needed, False otherwise
        """
        return False

    def prepare_batch(
        self, texts: List[str], context: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Prepare a batch of prompts for processing.

        Args:
            texts: List of texts to process
            context: Optional per-text context from dependent capabilities

        Returns:
            List of prompts ready for LLM processing
        """
        if context is None:
            return [self.create_prompt(text) for text in texts]

        return [self.create_prompt(text, context.get(text, {})) for text in texts]

    def post_process(
        self, results: Dict[str, Any], context: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Post-process results after LLM inference.

        This hook allows capabilities to transform or enrich their results
        before they're merged into the final output.

        Args:
            results: Raw results from LLM processing
            context: Optional context from dependent capabilities

        Returns:
            Processed results
        """
        return results

    def get_result_key(self) -> str:
        """
        The key name to use when merging results into the output.

        Returns:
            String key (defaults to capability name)
        """
        return self.name

    def format_for_export(self, result: Any) -> Any:
        """
        Format a single result for export to JSON.

        Args:
            result: Raw result object (Pydantic model or dict)

        Returns:
            JSON-serializable representation
        """
        if result is None:
            return None

        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "dict"):
            return result.dict()

        return result

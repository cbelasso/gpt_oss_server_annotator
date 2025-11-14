"""
Stem-specific trend analysis capability.

Analyzes temporal change patterns for each complete classification stem.
"""

import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel

from ..models import StemTrendsOutput
from ..prompts import stem_trends_prompt
from .base import Capability


class StemTrendCapability(Capability):
    """
    Evaluates temporal change patterns for complete classification stems.

    For each complete stem (path that reaches a leaf node), determines whether
    the text describes temporal changes about that specific topic - such as
    improvements, declines, stability, or fluctuations over time.

    Can identify multiple distinct trends for the same topic path if the comment
    discusses multiple aspects changing over time.

    Requires classification results to identify complete stems.
    """

    def __init__(self, max_stem_definitions: int = None):
        """
        Initialize stem trend capability.

        Args:
            max_stem_definitions: Maximum number of node definitions to include
                                 from the end of the stem (None = all)
        """
        self.max_stem_definitions = max_stem_definitions

    @property
    def name(self) -> str:
        return "stem_trend"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemTrendsOutput

    @property
    def dependencies(self) -> List[str]:
        return ["classification"]

    def requires_hierarchy(self) -> bool:
        return True

    def extract_stem_definitions(
        self, hierarchy: Dict[str, Any], stem_path: str, separator: str = ">"
    ) -> List[Dict[str, str]]:
        """Extract definitions for each node in a stem path."""
        path_parts = stem_path.split(separator)
        definitions = []

        # Start from hierarchy root
        if isinstance(hierarchy, dict) and "children" in hierarchy:
            current_nodes = hierarchy["children"]
        elif isinstance(hierarchy, list):
            current_nodes = hierarchy
        else:
            return definitions

        # Traverse the path
        for part in path_parts:
            # Find the matching node
            found = None
            for node in current_nodes:
                if node.get("name") == part:
                    found = node
                    break

            if not found:
                break

            # Extract definition info
            node_info = {
                "name": found.get("name", ""),
                "definition": found.get("definition", ""),
                "description": found.get("description", ""),
                "keywords": found.get("keywords", []),
            }
            definitions.append(node_info)

            # Move to children for next level
            current_nodes = found.get("children", [])

        # Limit to last N definitions if specified
        if self.max_stem_definitions is not None and self.max_stem_definitions > 0:
            definitions = definitions[-self.max_stem_definitions :]

        return definitions

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        This shouldn't be called directly - use prepare_batch instead.
        Stem trend requires special batch preparation with encoded context.
        """
        raise NotImplementedError("StemTrendCapability requires batch preparation with context")

    def prepare_batch(
        self, texts: List[str], context: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Prepare batch with encoded (text, stem, definitions) tuples.

        Returns encoded JSON strings that will be decoded in post_process.
        """
        if context is None:
            raise ValueError("StemTrendCapability requires classification context")

        hierarchy = context.get("_hierarchy")
        if hierarchy is None:
            raise ValueError("StemTrendCapability requires hierarchy in context")

        # Build list of (text, stem) pairs with definitions
        encoded_pairs = []

        for text in texts:
            text_context = context.get(text, {})
            complete_stems = text_context.get("complete_stems", [])

            for stem in complete_stems:
                stem_definitions = self.extract_stem_definitions(hierarchy, stem, separator=">")

                # Encode as JSON to preserve all info
                encoded = json.dumps(
                    {
                        "text": text,
                        "stem": stem,
                        "definitions": stem_definitions,
                    },
                    ensure_ascii=False,
                )

                encoded_pairs.append(encoded)

        # Store mapping for post-processing
        self._text_stem_mapping = []
        for text in texts:
            text_context = context.get(text, {})
            complete_stems = text_context.get("complete_stems", [])
            for stem in complete_stems:
                self._text_stem_mapping.append((text, stem))

        # Create prompts from encoded pairs
        prompts = []
        for encoded in encoded_pairs:
            data = json.loads(encoded)
            prompt = stem_trends_prompt(data["text"], data["stem"], data["definitions"])
            prompts.append(prompt)

        return prompts

    def post_process(
        self, results: Dict[str, Any], context: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reorganize results by text and stem.

        Results come back as a flat dict keyed by prompts.
        We reorganize into: text -> stem -> list of trends
        """
        stem_trend_dict = {}

        # Group results by text
        for encoded_key, result in results.items():
            # Find corresponding (text, stem) from mapping
            idx = list(results.keys()).index(encoded_key)
            if idx < len(self._text_stem_mapping):
                text, stem = self._text_stem_mapping[idx]

                if text not in stem_trend_dict:
                    stem_trend_dict[text] = {}

                # Convert result to dict
                if hasattr(result, "model_dump"):
                    stem_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    stem_dict = result.dict()
                else:
                    stem_dict = result

                # Store list of trends (or empty list if no trends)
                if stem_dict and stem_dict.get("has_trends"):
                    stem_trend_dict[text][stem] = stem_dict.get("trends", [])
                else:
                    stem_trend_dict[text][stem] = []

        return stem_trend_dict

    def get_result_key(self) -> str:
        """Store under 'stem_trends' key."""
        return "stem_trends"

    def format_for_export(self, result: Any) -> Any:
        """Format stem trend output for export."""
        if result is None:
            return {}

        # Result is already a dict of stem -> list of trends
        # Just return as-is
        return result

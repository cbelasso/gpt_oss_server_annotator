"""
Stem polarity analysis capability.

Analyzes the sentiment/polarity of text towards complete classification stems.
"""

import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel

from ..models import StemPolarityOutput
from ..prompts import stem_polarity_prompt
from .base import Capability


class StemPolarityCapability(Capability):
    """
    Evaluates polarity (sentiment) for complete classification stems.

    For each complete stem, determines whether the text expresses positive,
    negative, neutral, or mixed sentiment towards that specific topic.

    Requires classification results to identify complete stems.
    """

    def __init__(self, max_stem_definitions: int = None):
        """
        Initialize stem polarity capability.

        Args:
            max_stem_definitions: Maximum number of node definitions to include
                                 from the end of the stem (None = all)
        """
        self.max_stem_definitions = max_stem_definitions

    @property
    def name(self) -> str:
        return "stem_polarity"

    @property
    def schema(self) -> Type[BaseModel]:
        return StemPolarityOutput

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
        """
        raise NotImplementedError(
            "StemPolarityCapability requires batch preparation with context"
        )

    def prepare_batch(
        self, texts: List[str], context: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Prepare batch with encoded (text, stem, definitions) tuples.
        """
        if context is None:
            raise ValueError("StemPolarityCapability requires classification context")

        hierarchy = context.get("_hierarchy")
        if hierarchy is None:
            raise ValueError("StemPolarityCapability requires hierarchy in context")

        # Build list of (text, stem) pairs with definitions
        encoded_pairs = []

        for text in texts:
            text_context = context.get(text, {})
            complete_stems = text_context.get("complete_stems", [])

            for stem in complete_stems:
                stem_definitions = self.extract_stem_definitions(hierarchy, stem, separator=">")

                # Encode as JSON
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
            prompt = stem_polarity_prompt(data["text"], data["stem"], data["definitions"])
            prompts.append(prompt)

        return prompts

    def post_process(
        self, results: Dict[str, Any], context: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reorganize results by text and stem.
        """
        stem_polarity_dict = {}

        # Group results by text
        for encoded_key, result in results.items():
            idx = list(results.keys()).index(encoded_key)
            if idx < len(self._text_stem_mapping):
                text, stem = self._text_stem_mapping[idx]

                if text not in stem_polarity_dict:
                    stem_polarity_dict[text] = {}

                # Convert result to dict
                if hasattr(result, "model_dump"):
                    stem_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    stem_dict = result.dict()
                else:
                    stem_dict = result

                # Store polarity result
                if stem_dict and stem_dict.get("has_polarity"):
                    stem_polarity_dict[text][stem] = stem_dict.get("polarity_result", {})
                else:
                    stem_polarity_dict[text][stem] = None

        return stem_polarity_dict

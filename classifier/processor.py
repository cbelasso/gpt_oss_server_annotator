"""
High-level interface for text classification processing.

This module provides a simple, user-friendly API that abstracts away the
complexity of managing LLM processors, policies, and prompts.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

from llm_parallelization.parallelization import (
    FlexibleSchemaProcessor,
)
from llm_parallelization.vllm_server_processor import VLLMServerProcessor
from pydantic import BaseModel

from .classifier import HierarchicalClassifier
from .hierarchy import load_topic_hierarchy
from .models import ClassificationOutput
from .policies import AcceptancePolicy, DefaultPolicy
from .prompts import standard_classification_prompt


class ClassificationProcessor:
    def __init__(
        self,
        config_path: str | Path,
        gpu_list: List[int] = None,
        llm: str = "openai/gpt-oss-120b",
        prompt_fn: Callable[[Dict[str, Any]], str] = None,
        policy: AcceptancePolicy = None,
        separator: str = ">",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 10240,
        multiplicity: int = 1,
        batch_size=25,
        backend: str = "local",
        server_urls: List[str] = None,
        max_concurrent: int = 5,
    ):
        self.batch_size = batch_size
        self.config_path = config_path
        self.separator = separator
        self.backend = backend

        # Load hierarchy
        self.topic_hierarchy = load_topic_hierarchy(config_path)
        if not self.topic_hierarchy:
            raise ValueError(f"Failed to load hierarchy from {config_path}")

        # Initialize LLM processor based on backend
        if backend == "local":
            self.llm_processor = FlexibleSchemaProcessor(
                gpu_list=gpu_list or [0],
                llm=llm,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                multiplicity=multiplicity,
            )
        elif backend == "vllm-server":
            self.llm_processor = VLLMServerProcessor(
                server_urls=server_urls,
                model_name=llm,
                max_concurrent=max_concurrent,
                gpu_list=gpu_list,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Initialize classifier (same regardless of backend!)
        self.classifier = HierarchicalClassifier(
            processor=self.llm_processor,
            prompt_fn=prompt_fn or standard_classification_prompt,
            policy=policy or DefaultPolicy(),
            separator=separator,
        )

    def classify_hierarchical(self, texts: List[str]) -> Dict[str, List[str]]:
        """Classify texts using node-by-node hierarchical traversal."""
        return self.classifier.classify_hierarchical(
            texts=texts, topic_hierarchy=self.topic_hierarchy
        )

    def classify_target_path(self, texts: List[str], target_path: List[str]) -> Dict[str, Any]:
        """Evaluate texts against a specific target path (leaf node only)."""
        return self.classifier.classify_target_path(
            texts=texts, topic_hierarchy=self.topic_hierarchy, target_path=target_path
        )

    def classify_full_path(
        self, texts: List[str], target_path: List[str]
    ) -> Dict[str, ClassificationOutput]:
        """Evaluate all nodes along a specific path for each text."""
        return self.classifier.classify_full_path(
            texts=texts, topic_hierarchy=self.topic_hierarchy, target_path=target_path
        )

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str | Path,
        mode: str = None,  # optional now
    ) -> None:
        """
        Export classification results to a JSON file.
        Automatically detects result type:
        - hierarchical: Dict[str, List[str]]
        - target_path: Dict[str, SingleClassificationResult]
        - full_path: Dict[str, ClassificationOutput]
        - multi_target: Dict[str, Dict[str, Any]] (nested per target path)

        Args:
            results: Classification results from any classify_* method
            output_path: Path to output JSON file
            mode: Optional. If provided, overrides auto-detection
        """
        output_path = Path(output_path)
        output_data = []
        root_name = self.topic_hierarchy.get("name", "[ROOT]")

        # Auto-detect mode if not explicitly provided
        if mode is None:
            if all(isinstance(v, list) for v in results.values()):
                mode = "hierarchical"
            elif all(hasattr(v, "dict") for v in results.values()):
                mode = "full_path"
            elif all(isinstance(v, dict) for v in results.values()):
                # Could be single target_path OR multi-target nested dict
                first_val = next(iter(results.values()))
                if all(
                    isinstance(subv, (dict, list, ClassificationOutput))
                    for subv in first_val.values()
                ):
                    mode = "multi_target"
                else:
                    mode = "target_path"
            else:
                raise ValueError("Unable to detect results mode automatically.")

        # Serialize based on detected mode
        if mode == "hierarchical":
            for text, paths in results.items():
                full_paths = [
                    p if p.startswith(root_name) else f"{root_name}{self.separator}{p}"
                    for p in paths
                ]
                output_data.append(
                    {
                        "text": text,
                        "classification_paths": full_paths,
                    }
                )

        elif mode == "target_path":
            for text, result in results.items():
                output_data.append(
                    {
                        "text": text,
                        "classification_result": result.dict()
                        if hasattr(result, "dict")
                        else result,
                    }
                )

        elif mode == "full_path":
            for text, classification_output in results.items():
                output_data.append(
                    {
                        "text": text,
                        "classification_results": {
                            name: res.dict()
                            for name, res in classification_output.node_results.items()
                        },
                        "classification_paths": classification_output.classification_paths,
                    }
                )

        elif mode == "multi_target":
            # results: Dict[str, Dict[str, Any]]  (key = target_path string, value = sub_results)
            for target_key, sub_results in results.items():
                for text, classification in sub_results.items():
                    entry = {"text": text, "target_path_key": target_key}
                    if hasattr(classification, "dict"):
                        entry["classification_result"] = classification.dict()
                    elif isinstance(classification, dict):
                        entry["classification_result"] = classification
                    elif isinstance(classification, list):
                        entry["classification_paths"] = classification
                    else:
                        entry["classification_result"] = str(classification)
                    output_data.append(entry)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def classify_with_custom_schema(
        self, texts: List[str], prompt_fn: Callable[[str], str], schema: Type[BaseModel]
    ) -> Dict[str, BaseModel]:
        """
        Classify texts using a custom prompt function and Pydantic schema.

        This method allows running arbitrary classification tasks beyond the
        standard hierarchical classification, useful for post-processing or
        additional analysis tasks.

        Args:
            texts: List of text strings to classify
            prompt_fn: Function that takes a text string and returns a prompt
            schema: Pydantic model class defining the expected output structure

        Returns:
            Dictionary mapping each text to its classification result (Pydantic model instance)
        """
        # Generate prompts for all texts
        prompts = [prompt_fn(text) for text in texts]

        # Process with the FlexibleSchemaProcessor using the custom schema
        self.llm_processor.process_with_schema(
            prompts=prompts, schema=schema, batch_size=self.batch_size, formatted=False
        )

        # Parse results with the schema
        parsed_results = self.llm_processor.parse_results_with_schema(
            schema=schema, validate=True
        )

        # Map results back to texts
        return {text: result for text, result in zip(texts, parsed_results)}

    def cleanup(self) -> None:
        """Clean up resources (terminate LLM processor)."""
        if self.llm_processor:
            self.llm_processor.terminate()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()

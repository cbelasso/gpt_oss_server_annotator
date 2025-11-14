"""
Capability orchestrator for executing capabilities in dependency order.

Handles the complex workflow of running multiple capabilities, managing context,
and merging results.
"""

import time
from typing import Any, Dict, List

from rich.console import Console

from ..processor import ClassificationProcessor
from .registry import CapabilityRegistry

console = Console()


def get_root_prefix(hierarchy, project_name=None):
    """Extract the root node name from hierarchy."""
    if project_name:
        return project_name.upper()

    if hierarchy is None:
        return "ROOT"

    if isinstance(hierarchy, dict):
        if "name" in hierarchy:
            return hierarchy["name"].upper()

    return "ROOT"


def extract_all_leaf_paths(hierarchy):
    """Extract all root-to-leaf paths from hierarchy."""
    paths = []

    def traverse(node, current_path):
        current_path = current_path + [node["name"]]
        children = node.get("children", [])

        if not children:
            paths.append(current_path)
        else:
            for child in children:
                traverse(child, current_path)

    if isinstance(hierarchy, list):
        for root in hierarchy:
            traverse(root, [])
    elif isinstance(hierarchy, dict) and "children" in hierarchy:
        for root in hierarchy["children"]:
            traverse(root, [])

    return paths


def get_leaf_paths_set(hierarchy):
    """Get set of complete leaf paths (without root)."""
    leaf_paths = extract_all_leaf_paths(hierarchy)
    return {">".join(path) for path in leaf_paths}


def identify_complete_stems(
    classification_paths: List[str], leaf_paths_set: set, root_prefix: str
) -> List[str]:
    """Identify which classification paths are complete stems."""
    complete_stems = []

    for path in classification_paths:
        # Remove root prefix
        if path.startswith(root_prefix + ">"):
            path_without_root = path[len(root_prefix) + 1 :]
        else:
            path_without_root = path

        # Check if this path is a complete leaf path
        if path_without_root in leaf_paths_set:
            complete_stems.append(path_without_root)

    return complete_stems


class CapabilityOrchestrator:
    """
    Orchestrates execution of multiple capabilities.

    Handles dependency resolution, context management, and result merging.
    """

    def __init__(
        self, processor: ClassificationProcessor, registry: CapabilityRegistry, verbose: int = 0
    ):
        """
        Initialize orchestrator.

        Args:
            processor: ClassificationProcessor instance
            registry: CapabilityRegistry with registered capabilities
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)
        """
        self.processor = processor
        self.registry = registry
        self.verbose = verbose
        self.capability_timings = {}  # Track timing per capability

    def execute_capabilities(
        self, texts: List[str], capability_names: List[str], project_name: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute specified capabilities for a batch of texts.

        Args:
            texts: List of texts to process
            capability_names: List of capability names to execute
            project_name: Optional project name for root prefix

        Returns:
            Dict mapping text to all capability results
        """
        # Reset timings for this execution
        self.capability_timings = {}

        # Resolve execution order
        execution_order = self.registry.get_execution_order(capability_names)

        if self.verbose > 0:
            console.print(f"[cyan]Execution order: {' → '.join(execution_order)}[/cyan]")

        # Initialize results structure
        results = {text: {"text": text} for text in texts}

        # Context for capabilities (stores intermediate results)
        context: Dict[str, Dict[str, Any]] = {}

        # Execute capabilities in order
        for cap_name in execution_order:
            capability = self.registry.get(cap_name)

            # Start timing
            cap_start = time.time()

            if self.verbose > 0:
                console.print(f"[cyan]Executing capability: {cap_name}[/cyan]")

            # Special handling for classification capability
            if cap_name == "classification":
                cap_results = self._execute_classification(texts)

                # Build context for dependent capabilities
                if any(
                    cap_name in ["stem_recommendations", "stem_polarity", "stem_trend"]
                    for cap_name in capability_names
                ):
                    context = self._build_classification_context(
                        texts, cap_results, project_name
                    )
            else:
                # Execute regular capability
                cap_results = self._execute_capability(capability, texts, context)

            # Merge results
            self._merge_results(results, cap_results, capability)

            # End timing
            cap_elapsed = time.time() - cap_start
            self.capability_timings[cap_name] = cap_elapsed

            if self.verbose > 0:
                console.print(f"[green]✓ {cap_name} complete ({cap_elapsed:.2f}s)[/green]")

        return results

    def get_timing_summary(self) -> Dict[str, float]:
        """
        Get timing summary for last execution.

        Returns:
            Dict mapping capability name to elapsed time in seconds
        """
        return self.capability_timings.copy()

    def format_timing_summary(self) -> str:
        """
        Format timing summary as a readable string.

        Returns:
            Formatted string with timing breakdown and percentages
        """
        if not self.capability_timings:
            return "No timing data available"

        lines = []
        total = sum(self.capability_timings.values())

        # Sort by time (descending)
        for cap_name, elapsed in sorted(
            self.capability_timings.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"  • {cap_name}: {elapsed:.2f}s ({percentage:.1f}%)")

        lines.append(f"  • Total: {total:.2f}s")
        return "\n".join(lines)

    def _execute_classification(self, texts: List[str]) -> Dict[str, Any]:
        """Execute hierarchical classification."""
        return self.processor.classify_hierarchical(texts=texts)

    def _execute_capability(
        self, capability, texts: List[str], context: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a single capability."""
        # Prepare prompts
        prompts = capability.prepare_batch(texts, context)

        # Create mapping: text -> prompt
        # For simple capabilities: len(prompts) == len(texts)
        # For complex capabilities (stem): handled in post_process
        text_to_prompt_map = None
        if len(prompts) == len(texts):
            text_to_prompt_map = {text: prompt for text, prompt in zip(texts, prompts)}

        # Execute with processor (results keyed by prompt)
        prompt_results = self.processor.classify_with_custom_schema(
            texts=prompts,
            prompt_fn=lambda x: x,  # Prompts already formatted
            schema=capability.schema,
        )

        # Post-process results
        processed_results = capability.post_process(prompt_results, context)

        # Remap from prompt keys to text keys if needed
        if text_to_prompt_map is not None:
            results = {}
            for text, prompt in text_to_prompt_map.items():
                if prompt in processed_results:
                    results[text] = processed_results[prompt]
                else:
                    # Handle missing results gracefully
                    if self.verbose >= 2:
                        console.print("[yellow]Warning: No result for prompt[/yellow]")
                    results[text] = None
            return results
        else:
            # Stem capabilities already return text-keyed results from post_process
            return processed_results

    def _build_classification_context(
        self, texts: List[str], classification_results: Dict[str, Any], project_name: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build context from classification results for dependent capabilities.

        Extracts complete stems and adds hierarchy reference.
        """
        hierarchy = self.processor.topic_hierarchy
        leaf_paths_set = get_leaf_paths_set(hierarchy)
        root_prefix = get_root_prefix(hierarchy, project_name=None)

        context = {"_hierarchy": hierarchy}

        for text in texts:
            if text in classification_results:
                result = classification_results[text]

                # Convert to dict if needed
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = result

                # Extract classification paths
                classification_paths = result_dict.get("classification_paths", [])

                # Identify complete stems
                complete_stems = identify_complete_stems(
                    classification_paths, leaf_paths_set, root_prefix
                )

                context[text] = {"complete_stems": complete_stems}

        return context

    def _merge_results(
        self, results: Dict[str, Dict[str, Any]], capability_results: Dict[str, Any], capability
    ) -> None:
        """
        Merge capability results into main results structure.

        Modifies results dict in-place.
        """
        result_key = capability.get_result_key()

        for text in results.keys():
            if text in capability_results and capability_results[text] is not None:
                result = capability_results[text]
                formatted_result = capability.format_for_export(result)
                results[text][result_key] = formatted_result
            else:
                # If no result for this text, add empty default
                if self.verbose >= 3:
                    console.print(
                        f"[yellow]Warning: No {result_key} for text: {text[:30]}...[/yellow]"
                    )
                # Use capability's format_for_export with None to get proper default
                results[text][result_key] = capability.format_for_export(None)

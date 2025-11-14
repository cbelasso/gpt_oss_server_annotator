"""
Core classification logic for hierarchical text classification.

This module contains the main classification algorithms, including both
node-by-node hierarchical traversal and full-path evaluation modes.

MODIFIED VERSION: classify_hierarchical now returns full ClassificationOutput
objects with node_results preserved, including excerpts.
"""

from typing import Any, Callable, Dict, List, Tuple

from llm_parallelization.parallelization import FlexibleSchemaProcessor

from .hierarchy import get_node_path
from .models import ClassificationOutput, SingleClassificationResult
from .policies import AcceptancePolicy, DefaultPolicy
from .prompts import add_text_to_prompt, standard_classification_prompt


class HierarchicalClassifier:
    """
    Main classifier for hierarchical text classification.

    This class manages the classification process, coordinating between the LLM
    processor, prompts, and acceptance policies.

    Args:
        processor: FlexibleSchemaProcessor for batch LLM generation
        prompt_fn: Function to generate prompts from node configs
        policy: Acceptance policy for filtering results
        separator: String to join hierarchical path components
    """

    def __init__(
        self,
        processor: FlexibleSchemaProcessor,
        prompt_fn: Callable[[Dict[str, Any]], str] = standard_classification_prompt,
        policy: AcceptancePolicy = None,
        separator: str = ">",
    ):
        self.processor = processor
        self.prompt_fn = prompt_fn
        self.policy = policy or DefaultPolicy()
        self.separator = separator

    def _batch_generate(self, prompts: List[str]) -> List[SingleClassificationResult]:
        """
        Execute batch LLM generation and parse results.

        Args:
            prompts: List of prompts to send to the LLM

        Returns:
            List of parsed classification results
        """
        self.processor.process_with_schema(prompts=prompts, schema=SingleClassificationResult)
        return self.processor.parse_results_with_schema(schema=SingleClassificationResult)

    def classify_hierarchical(
        self, texts: List[str], topic_hierarchy: Dict[str, Any]
    ) -> Dict[str, ClassificationOutput]:
        """
        Classify texts using node-by-node hierarchical traversal.

        **MODIFIED**: Now returns full ClassificationOutput objects with node_results
        preserved, including excerpts, reasoning, and confidence scores.

        This method explores the hierarchy dynamically, starting from the root's
        children and expanding only those branches where relevance is established.

        **Algorithm**:
        1. Initialize a queue with all children of the root node for each text
        2. For each round:
           a. Batch-evaluate all (text, node) pairs in the queue
           b. For each result that passes the acceptance policy:
              - Add the path to final results
              - Store the node result (including excerpt)
              - Queue this node's children for the next round
           c. Nodes that don't pass the policy stop their branch from expanding
        3. Continue until no more nodes are queued

        This provides efficient exploration: irrelevant branches are pruned early,
        avoiding unnecessary LLM calls.

        Args:
            texts: List of texts to classify
            topic_hierarchy: Root node of the topic hierarchy

        Returns:
            Dictionary mapping each text to ClassificationOutput containing:
            - classification_paths: List of valid hierarchical paths
            - node_results: Dictionary of node results with excerpts

        Example:
            >>> results = classifier.classify_hierarchical(
            ...     texts=["The CEO discussed our vision"],
            ...     topic_hierarchy=hierarchy
            ... )
            >>> output = results["The CEO discussed our vision"]
            >>> output.classification_paths
            ["Root>Leadership>CEO>Vision"]
            >>> output.node_results["CEO"].excerpt
            "The CEO discussed"
        """
        root_name = topic_hierarchy.get("name", "[ROOT]")

        # Initialize queue: for each text, start with root's children
        text_queue: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
            text: [(child, root_name) for child in topic_hierarchy.get("children", [])]
            for text in texts
        }

        # Initialize results with ClassificationOutput objects
        final_results: Dict[str, ClassificationOutput] = {
            text: ClassificationOutput(text=text, classification_paths=[], node_results={})
            for text in texts
        }

        # Process level by level
        while any(text_queue.values()):
            # Collect all (text, node, parent_path) tuples for this round
            batch_prompts = []
            text_node_map = []

            for text, nodes in text_queue.items():
                for node, parent_path in nodes:
                    prompt = add_text_to_prompt(self.prompt_fn(node), text)
                    batch_prompts.append(prompt)
                    text_node_map.append((text, node, parent_path))

            # Batch generate for all prompts
            results = self._batch_generate(batch_prompts)

            # Process results and build next queue
            next_text_queue: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
                text: [] for text in texts
            }

            for (text, node, parent_path), result in zip(text_node_map, results):
                # Store the node result regardless of acceptance
                # This preserves all classification attempts
                node_name = node["name"]
                final_results[text].node_results[node_name] = result

                if self.policy.accept(result, node):
                    # Build current path
                    current_path = f"{parent_path}{self.separator}{node['name']}"
                    final_results[text].classification_paths.append(current_path)

                    # Queue children for next round
                    for child in node.get("children", []):
                        next_text_queue[text].append((child, current_path))

            text_queue = next_text_queue

        return final_results

    def classify_target_path(
        self, texts: List[str], topic_hierarchy: Dict[str, Any], target_path: List[str]
    ) -> Dict[str, SingleClassificationResult]:
        """
        Evaluate texts against a specific target path (leaf node only).

        This method checks only the final node in the path, useful when you
        already know which specific topic to test for.

        Args:
            texts: List of texts to classify
            topic_hierarchy: Root node of the topic hierarchy
            target_path: Path to the target node (e.g., ["Root", "A", "B"])

        Returns:
            Dictionary mapping each text to its classification result
        """
        nodes = get_node_path(topic_hierarchy, target_path)
        if not nodes:
            raise ValueError(
                f"Target path {' > '.join(target_path)} does not exist in hierarchy."
            )

        last_node = nodes[-1]
        prompts = [add_text_to_prompt(self.prompt_fn(last_node), text) for text in texts]
        results = self._batch_generate(prompts)

        return {text: result for text, result in zip(texts, results)}

    def classify_full_path(
        self, texts: List[str], topic_hierarchy: Dict[str, Any], target_path: List[str]
    ) -> Dict[str, ClassificationOutput]:
        """
        Evaluate all nodes along a specific path for each text.

        Unlike classify_target_path (which only checks the leaf), this method
        evaluates every node from root to leaf along the specified path.

        **Algorithm**:
        1. Extract all nodes along the target path
        2. For each text, generate prompts for every node in the path
        3. Batch-evaluate all (text, node) pairs
        4. Organize results by text and node
        5. Build classification paths by checking consecutive relevance:
           - Start from root
           - Continue adding nodes while they pass the acceptance policy
           - Stop at the first node that doesn't pass

        This is useful when you want detailed per-node analysis along a known path,
        or to understand where in the hierarchy the text stops being relevant.

        Args:
            texts: List of texts to classify
            topic_hierarchy: Root node of the topic hierarchy
            target_path: Path to evaluate (e.g., ["Root", "A", "B"])

        Returns:
            Dictionary mapping each text to a ClassificationOutput containing:
            - node_results: Classification result for each node
            - classification_paths: Valid paths based on consecutive relevance

        Example:
            >>> results = classifier.classify_full_path(
            ...     texts=["Discussion about leadership"],
            ...     topic_hierarchy=hierarchy,
            ...     target_path=["Root", "Leadership", "CEO"]
            ... )
            >>> results["Discussion about leadership"].node_results
            {"Root": ..., "Leadership": ..., "CEO": ...}
        """
        root_name = topic_hierarchy.get("name", "[ROOT]")

        # Skip root in path if it's explicitly included
        path_to_use = (
            target_path[1:] if target_path and target_path[0] == root_name else target_path[:]
        )

        nodes = get_node_path(topic_hierarchy, target_path)
        if not nodes:
            raise ValueError(
                f"Target path {' > '.join(target_path)} does not exist in hierarchy."
            )

        # Prepare batch prompts for all (text, node) combinations
        batch_prompts = []
        text_node_map = []

        for text in texts:
            for node in nodes:
                prompt = add_text_to_prompt(self.prompt_fn(node), text)
                batch_prompts.append(prompt)
                text_node_map.append((text, node["name"]))

        # Batch generate
        results_raw = self._batch_generate(batch_prompts)

        # Organize results per text
        results: Dict[str, ClassificationOutput] = {}
        for text in texts:
            results[text] = ClassificationOutput(
                text=text, node_results={}, classification_paths=[]
            )

        for (text, node_name), result in zip(text_node_map, results_raw):
            results[text].node_results[node_name] = result

        # Build classification paths based on consecutive relevance
        for text, output in results.items():
            path_accum = [root_name]

            for node in nodes:
                node_name = node["name"]
                result = output.node_results[node_name]

                if self.policy.accept(result, node):
                    path_accum.append(node_name)
                else:
                    # Stop at first node that doesn't pass policy
                    break

            # Only add path if we got beyond the root
            if len(path_accum) > 1:
                output.classification_paths.append(self.separator.join(path_accum))

        return results

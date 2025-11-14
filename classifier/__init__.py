"""
Hierarchical Text Classification Package

A modular, extensible system for classifying text according to hierarchical
topic structures using LLMs.

Main components:
- models: Data models for classification results
- policies: Acceptance policies for filtering results
- prompts: Customizable prompt templates
- hierarchy: Utilities for working with topic hierarchies
- classifier: Core classification algorithms
- processor: High-level processing interface
- cli: Command-line interface

Example usage:
    >>> from classifier import ClassificationProcessor
    >>> from classifier.policies import ConfidenceThresholdPolicy
    >>> 
    >>> with ClassificationProcessor(
    ...     config_path="topics.json",
    ...     policy=ConfidenceThresholdPolicy(min_confidence=4)
    ... ) as processor:
    ...     results = processor.classify_hierarchical(texts)
    ...     processor.export_results(results, "output.json")
"""

from .models import (
    SingleClassificationResult,
    NodeConfig,
    ClassificationOutput,
    BatchClassificationResult
)

from .policies import (
    AcceptancePolicy,
    DefaultPolicy,
    ConfidenceThresholdPolicy,
    KeywordInReasoningPolicy,
    ExcerptRequiredPolicy,
    MinimumReasoningLengthPolicy,
    CompositePolicy,
    AnyPolicy
)

from .prompts import (
    standard_classification_prompt,
    hierarchical_path_prompt,
    sentiment_aware_classification_prompt,
    keyword_focused_prompt,
    add_text_to_prompt
)

from .hierarchy import (
    load_topic_hierarchy,
    get_node_path,
    get_all_leaf_paths,
    build_tree_from_paths,
    format_tree_as_string,
    validate_hierarchy
)

from .classifier import HierarchicalClassifier

from .processor import ClassificationProcessor


__version__ = "1.0.0"

__all__ = [
    # Models
    "SingleClassificationResult",
    "NodeConfig",
    "ClassificationOutput",
    "BatchClassificationResult",
    
    # Policies
    "AcceptancePolicy",
    "DefaultPolicy",
    "ConfidenceThresholdPolicy",
    "KeywordInReasoningPolicy",
    "ExcerptRequiredPolicy",
    "MinimumReasoningLengthPolicy",
    "CompositePolicy",
    "AnyPolicy",
    
    # Prompts
    "standard_classification_prompt",
    "hierarchical_path_prompt",
    "sentiment_aware_classification_prompt",
    "keyword_focused_prompt",
    "add_text_to_prompt",
    
    # Hierarchy
    "load_topic_hierarchy",
    "get_node_path",
    "get_all_leaf_paths",
    "build_tree_from_paths",
    "format_tree_as_string",
    "validate_hierarchy",
    
    # Core
    "HierarchicalClassifier",
    "ClassificationProcessor",
]

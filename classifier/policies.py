"""
Acceptance policies for classification results.

This module provides a flexible policy system for determining whether a classification
result should be accepted. Policies can be combined and customized to implement
complex acceptance criteria.
"""

from typing import Protocol, List, Dict, Any
from .models import SingleClassificationResult


class AcceptancePolicy(Protocol):
    """
    Protocol defining the interface for acceptance policies.
    
    Policies determine whether a classification result meets acceptance criteria.
    """
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        """
        Determine if a classification result should be accepted.
        
        Args:
            result: The classification result to evaluate
            node: Optional node configuration for context-aware decisions
            
        Returns:
            True if the result meets acceptance criteria, False otherwise
        """
        ...


class DefaultPolicy:
    """
    Accept only when is_relevant==True (ignores confidence).
    
    This is the most permissive policy, accepting any result marked as relevant
    regardless of confidence level.
    """
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        return bool(result and result.is_relevant)


class ConfidenceThresholdPolicy:
    """
    Require a minimum confidence level for acceptance.
    
    This policy ensures that only high-confidence classifications are accepted,
    filtering out uncertain or ambiguous results.
    
    Args:
        min_confidence: Minimum confidence score (1-5) required for acceptance
    """
    
    def __init__(self, min_confidence: int = 3):
        if not 1 <= min_confidence <= 5:
            raise ValueError("min_confidence must be between 1 and 5")
        self.min_confidence = min_confidence
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        return bool(
            result 
            and result.is_relevant 
            and result.confidence >= self.min_confidence
        )


class KeywordInReasoningPolicy:
    """
    Require that reasoning mentions at least one of the provided keywords.
    
    This policy ensures that the LLM's reasoning explicitly references
    expected concepts, providing an additional validation layer.
    
    Args:
        keywords: List of keywords to check for in reasoning
    """
    
    def __init__(self, keywords: List[str]):
        self.keywords = [k.lower() for k in keywords]
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        if not result or not result.is_relevant:
            return False
        reasoning = (result.reasoning or "").lower()
        return any(kw in reasoning for kw in self.keywords)


class ExcerptRequiredPolicy:
    """
    Require that a non-empty excerpt is provided.
    
    This ensures the model can point to specific evidence in the text,
    reducing false positives.
    """
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        return bool(
            result 
            and result.is_relevant 
            and result.excerpt 
            and result.excerpt.strip()
        )


class MinimumReasoningLengthPolicy:
    """
    Require reasoning to be at least a certain number of words.
    
    This helps ensure the model provides substantive explanations
    rather than superficial responses.
    
    Args:
        min_words: Minimum number of words required in reasoning
    """
    
    def __init__(self, min_words: int = 5):
        self.min_words = min_words
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        if not result or not result.is_relevant:
            return False
        word_count = len((result.reasoning or "").split())
        return word_count >= self.min_words


class CompositePolicy:
    """
    Combine multiple policies using logical AND.
    
    All constituent policies must accept for the composite policy to accept.
    This allows building complex acceptance criteria from simple building blocks.
    
    Args:
        *policies: Variable number of policies to combine
        
    Example:
        >>> policy = CompositePolicy(
        ...     ConfidenceThresholdPolicy(min_confidence=4),
        ...     ExcerptRequiredPolicy()
        ... )
    """
    
    def __init__(self, *policies: AcceptancePolicy):
        self.policies = policies
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        return all(p.accept(result, node) for p in self.policies)


class AnyPolicy:
    """
    Combine multiple policies using logical OR.
    
    At least one constituent policy must accept for this policy to accept.
    
    Args:
        *policies: Variable number of policies to combine
    """
    
    def __init__(self, *policies: AcceptancePolicy):
        self.policies = policies
    
    def accept(
        self, 
        result: SingleClassificationResult, 
        node: Dict[str, Any] | None = None
    ) -> bool:
        return any(p.accept(result, node) for p in self.policies)

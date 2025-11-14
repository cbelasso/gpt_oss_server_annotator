"""
Data models for hierarchical text classification.

This module defines the core data structures used throughout the classification system.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class SingleClassificationResult(BaseModel):
    """
    Result of classifying a single text against a single node.

    Attributes:
        is_relevant: Whether the text is relevant to the node's topic
        confidence: Confidence score from 1-5 (1=very uncertain, 5=very certain)
        reasoning: Explanation of the classification decision (1-2 sentences)
        excerpt: Exact text span supporting the classification (empty if not relevant)
    """

    is_relevant: bool
    confidence: int
    reasoning: str
    excerpt: str


class NodeConfig(BaseModel):
    """
    Configuration for a single node in the topic hierarchy.

    Attributes:
        name: Node identifier
        description: Detailed description of the topic
        keywords: List of relevant keywords
        scope: Scope definition for the topic
        children: Child nodes in the hierarchy
    """

    name: str
    description: str = "[No Description]"
    keywords: List[str] = []
    scope: str = "[None]"
    children: List["NodeConfig"] = []

    class Config:
        # Allow arbitrary types for flexibility
        arbitrary_types_allowed = True


class ClassificationOutput(BaseModel):
    """
    Final output for a single text's classification.

    Attributes:
        text: The original text that was classified
        classification_paths: List of hierarchical paths (e.g., ["Root>A>B", "Root>C"])
        node_results: Optional detailed results per node (for target path evaluation)
    """

    text: str
    classification_paths: List[str]
    node_results: Dict[str, SingleClassificationResult] | None = None


class BatchClassificationResult(BaseModel):
    """
    Results for a batch of texts.

    Attributes:
        results: List of classification outputs, one per input text
    """

    results: List[ClassificationOutput]


class RecommendationSpan(BaseModel):
    excerpt: str
    reasoning: str = ""
    paraphrased_recommendation: str = ""
    qualifier: Literal[
        "add_or_increase",
        "reduce_or_remove",
        "introduce_or_start",
        "eliminate_or_stop",
        "modify_or_improve",
        "maintain_or_continue",
        "unspecified_or_general",
    ] = "unspecified_or_general"


class RecommendationsOutput(BaseModel):
    has_recommendations: bool
    recommendations: List[RecommendationSpan] = []


class AlertSpan(BaseModel):
    excerpt: str
    reasoning: str = ""
    alert_type: Literal[
        "discrimination",
        "sexual_harassment",
        "workplace_violence",
        "safety_concern",
        "ethical_violation",
        "hostile_environment",
        "retaliation",
        "bullying",
        "substance_abuse",
        "mental_health_crisis",
        "data_breach",
        "fraud",
        "other_serious_concern",
    ]
    severity: Literal["low", "moderate", "high", "critical"] = "moderate"


class AlertsOutput(BaseModel):
    has_alerts: bool
    alerts: List[AlertSpan] = []


class StemRecommendationType(BaseModel):
    recommendation_type: Literal["start", "stop", "do_more", "do_less", "continue", "change"]
    excerpt: str
    reasoning: str = ""
    paraphrased_recommendation: str = ""


class StemRecommendationsOutput(BaseModel):
    has_recommendations: bool
    recommendations: List[StemRecommendationType] = []


class StemPolarityResult(BaseModel):
    polarity: Literal["Positive", "Negative", "Neutral", "Mixed"]
    confidence: int  # 1-5
    reasoning: str = ""
    excerpt: str = ""


class StemPolarityOutput(BaseModel):
    has_polarity: bool
    polarity_result: Optional[StemPolarityResult] = None


class TrendSpan(BaseModel):
    """
    Individual trend detected in text.

    Attributes:
        excerpt: The exact text span from the comment containing the trend
        reasoning: One sentence explaining why this qualifies as a trend
        subject: What aspect is changing (e.g., "manager support", "training quality")
        direction: Type of change (increasing, decreasing, improving, deteriorating, stable, fluctuating)
        valence: Whether the trend is viewed positively or negatively
        confidence: Confidence score from 1-5 (1=very uncertain, 5=very certain)
    """

    excerpt: str
    reasoning: str = ""
    subject: str = ""
    direction: Literal[
        "increasing",
        "decreasing",
        "improving",
        "deteriorating",
        "stable_positive",
        "stable_negative",
        "fluctuating",
    ]
    valence: Literal["positive", "negative", "neutral", "mixed"] = "neutral"
    confidence: int


class TrendsOutput(BaseModel):
    """
    Output for global trend detection capability.

    Detects temporal change patterns across the entire text.
    Can contain multiple trends if the comment discusses multiple aspects changing over time.
    """

    has_trends: bool
    trends: List[TrendSpan] = []


class StemTrendSpan(BaseModel):
    """
    Individual trend detected for a specific classification stem.

    Attributes:
        excerpt: The exact text span from the comment containing the trend
        reasoning: One sentence explaining why this qualifies as a trend for this topic path
        subject: What aspect of the topic is changing (aligned with the topic path)
        direction: Type of change (increasing, decreasing, improving, deteriorating, stable, fluctuating)
        valence: Whether the trend is viewed positively or negatively
        confidence: Confidence score from 1-5 (1=very uncertain, 5=very certain)
    """

    excerpt: str
    reasoning: str = ""
    subject: str = ""
    direction: Literal[
        "increasing",
        "decreasing",
        "improving",
        "deteriorating",
        "stable_positive",
        "stable_negative",
        "fluctuating",
    ]
    valence: Literal["positive", "negative", "neutral", "mixed"] = "neutral"
    confidence: int


class StemTrendsOutput(BaseModel):
    """
    Output for stem-specific trend analysis capability.

    Analyzes temporal change patterns for a specific classification stem.
    Can contain multiple trends if the comment discusses multiple aspects of the topic changing over time.
    """

    has_trends: bool
    trends: List[StemTrendSpan] = []

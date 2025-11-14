# Adding New Capabilities to the Classification Server

This guide documents the complete process for adding new capabilities to the hierarchical text classification system.

## Table of Contents
1. [Overview](#overview)
2. [Types of Capabilities](#types-of-capabilities)
3. [Implementation Steps](#implementation-steps)
4. [Example: Adding Trend Capabilities](#example-adding-trend-capabilities)
5. [Testing Your Capability](#testing-your-capability)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The capabilities system provides a plugin-like architecture for extending the classification server with new features. Each capability is a self-contained module that can:

- Process text independently or depend on other capabilities
- Define its own Pydantic schema for structured outputs
- Access the topic hierarchy when needed
- Be composed with other capabilities in a single request

**Core Design Principles:**
- **Modularity**: Each capability is independent and reusable
- **Composability**: Capabilities can be combined in requests
- **Dependency Management**: Automatic resolution of capability dependencies
- **Type Safety**: Pydantic schemas ensure structured, validated outputs

---

## Types of Capabilities

There are two main patterns for capabilities:

### 1. Global Capabilities
**Characteristics:**
- Analyze entire text without requiring classification
- No dependencies on other capabilities
- Simple, single-pass processing
- Fast execution

**Examples:**
- `RecommendationsCapability` - Detects all recommendations in text
- `AlertsCapability` - Identifies safety/HR concerns
- `TrendCapability` - Detects temporal change patterns

**Use Cases:**
- Quick feature extraction
- Independent analysis that doesn't need topic classification
- Features that apply to the whole text

### 2. Stem-Specific Capabilities
**Characteristics:**
- Depend on classification results
- Analyze each complete classification path (stem) separately
- Require access to topic hierarchy
- More complex batch preparation

**Examples:**
- `StemRecommendationsCapability` - Analyzes recommendations per topic
- `StemPolarityCapability` - Analyzes sentiment per topic
- `StemTrendCapability` - Analyzes temporal patterns per topic

**Use Cases:**
- Granular, topic-specific analysis
- When different topics in the same text need different assessments
- Features that depend on classification context

---

## Implementation Steps

### Step 1: Define Data Models

Add your output models to `classifier/models.py`.

**For Global Capabilities:**
```python
class YourFeatureResult(BaseModel):
    """Result structure for your feature."""
    field1: str
    field2: int
    confidence: int  # 1-5
    reasoning: str = ""
    excerpt: str = ""

class YourFeatureOutput(BaseModel):
    """Output wrapper."""
    has_feature: bool
    feature_result: Optional[YourFeatureResult] = None
```

**For Stem-Specific Capabilities:**
```python
class StemYourFeatureResult(BaseModel):
    """Result for a specific stem."""
    field1: str
    field2: int
    confidence: int  # 1-5
    reasoning: str = ""
    excerpt: str = ""

class StemYourFeatureOutput(BaseModel):
    """Output wrapper for stem analysis."""
    has_feature: bool
    feature_result: Optional[StemYourFeatureResult] = None
```

**Key Guidelines:**
- Use descriptive field names
- Include confidence scores (1-5) for uncertain outputs
- Always include reasoning and excerpt fields
- Use Literal types for categorical fields
- Make optional fields have defaults

### Step 2: Create Prompt Functions

Add your prompt generation functions to `classifier/prompts.py`.

**For Global Capabilities:**
```python
def your_feature_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect your feature.
    
    Args:
        text: The text to analyze
        
    Returns:
        Formatted prompt string
    """
    return f"""
You are an expert text analyzer analyzing an employee comment.

**Comment**: {text}

---

# Your Task
[Clear description of what to detect/analyze]

# Key Rules
- Use only explicit information
- Provide specific excerpts
- Explain reasoning clearly

# Output Format
Return JSON with:
- has_feature: Boolean
- feature_result: Object with your fields

# Examples
[Provide 3-5 clear examples]
"""
```

**For Stem-Specific Capabilities:**
```python
def stem_your_feature_prompt(
    text: str, 
    stem_path: str, 
    stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate prompt for stem-specific analysis.
    
    Args:
        text: The text to analyze
        stem_path: Classification path (e.g., "Topic>Subtopic")
        stem_definitions: Node definitions from hierarchy
        
    Returns:
        Formatted prompt string
    """
    # Format definitions section
    definitions_section = ""
    if stem_definitions:
        definitions_section = "\n**Topic Path Definitions:**\n\n"
        for node_info in stem_definitions:
            definitions_section += f"**{node_info['name']}**\n"
            if node_info.get("definition"):
                definitions_section += f"- Definition: {node_info['definition']}\n"
            # ... add description, keywords, etc.
    
    return f"""
You are analyzing a comment classified under a specific topic.

**Comment**: {text}
**Topic Path**: {stem_path}

{definitions_section}---

# Your Task
Analyze this specific topic for [your feature].
Focus only on content related to this topic path.

[Rest of prompt similar to global version]
"""
```

**Prompt Best Practices:**
- Be extremely explicit about what to detect
- Provide clear decision criteria
- Include 3-5 diverse examples
- Use formatting for readability
- Specify exact output format
- Include edge cases in examples

### Step 3: Implement Capability Class

Create your capability class in `classifier/capabilities/`.

**For Global Capabilities:**

Create `classifier/capabilities/your_feature.py`:

```python
"""
Your feature detection capability.

[Description of what this capability does]
"""

from typing import Any, Dict, Type
from pydantic import BaseModel

from ..models import YourFeatureOutput
from ..prompts import your_feature_detection_prompt
from .base import Capability


class YourFeatureCapability(Capability):
    """
    [Docstring describing the capability]
    """
    
    @property
    def name(self) -> str:
        return "your_feature"
    
    @property
    def schema(self) -> Type[BaseModel]:
        return YourFeatureOutput
    
    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        return your_feature_detection_prompt(text)
    
    def get_result_key(self) -> str:
        """Key for storing results in output."""
        return "your_feature"
    
    def format_for_export(self, result: Any) -> Any:
        """Format output for JSON export."""
        if result is None:
            return {"has_feature": False, "feature_result": None}
        
        # Convert Pydantic to dict
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "dict"):
            return result.dict()
        
        return result
```

**For Stem-Specific Capabilities:**

Create `classifier/capabilities/stem_your_feature.py`:

```python
"""
Stem-specific [your feature] capability.

[Description]
"""

import json
from typing import Any, Dict, List, Type
from pydantic import BaseModel

from ..models import StemYourFeatureOutput
from ..prompts import stem_your_feature_prompt
from .base import Capability


class StemYourFeatureCapability(Capability):
    """
    [Docstring]
    """
    
    def __init__(self, max_stem_definitions: int = None):
        """
        Initialize capability.
        
        Args:
            max_stem_definitions: Max node definitions to include
        """
        self.max_stem_definitions = max_stem_definitions
    
    @property
    def name(self) -> str:
        return "stem_your_feature"
    
    @property
    def schema(self) -> Type[BaseModel]:
        return StemYourFeatureOutput
    
    @property
    def dependencies(self) -> List[str]:
        return ["classification"]  # Requires classification first
    
    def requires_hierarchy(self) -> bool:
        return True
    
    def extract_stem_definitions(
        self, hierarchy: Dict[str, Any], stem_path: str, separator: str = ">"
    ) -> List[Dict[str, str]]:
        """Extract node definitions from hierarchy for a stem path."""
        path_parts = stem_path.split(separator)
        definitions = []
        
        # Navigate hierarchy
        if isinstance(hierarchy, dict) and "children" in hierarchy:
            current_nodes = hierarchy["children"]
        elif isinstance(hierarchy, list):
            current_nodes = hierarchy
        else:
            return definitions
        
        # Traverse path
        for part in path_parts:
            found = None
            for node in current_nodes:
                if node.get("name") == part:
                    found = node
                    break
            
            if not found:
                break
            
            # Extract definition
            node_info = {
                "name": found.get("name", ""),
                "definition": found.get("definition", ""),
                "description": found.get("description", ""),
                "keywords": found.get("keywords", []),
            }
            definitions.append(node_info)
            current_nodes = found.get("children", [])
        
        # Optionally limit to last N definitions
        if self.max_stem_definitions and self.max_stem_definitions > 0:
            definitions = definitions[-self.max_stem_definitions:]
        
        return definitions
    
    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """Don't call directly - use prepare_batch."""
        raise NotImplementedError(
            f"{self.__class__.__name__} requires batch preparation with context"
        )
    
    def prepare_batch(
        self, texts: List[str], context: Dict[str, Dict[str, Any]] = None
    ) -> List[str]:
        """
        Prepare prompts for all (text, stem) combinations.
        """
        if context is None:
            raise ValueError(f"{self.__class__.__name__} requires classification context")
        
        hierarchy = context.get("_hierarchy")
        if hierarchy is None:
            raise ValueError(f"{self.__class__.__name__} requires hierarchy")
        
        # Build (text, stem) pairs
        encoded_pairs = []
        
        for text in texts:
            text_context = context.get(text, {})
            complete_stems = text_context.get("complete_stems", [])
            
            for stem in complete_stems:
                stem_defs = self.extract_stem_definitions(hierarchy, stem, ">")
                
                encoded = json.dumps({
                    "text": text,
                    "stem": stem,
                    "definitions": stem_defs,
                }, ensure_ascii=False)
                
                encoded_pairs.append(encoded)
        
        # Store mapping for post-processing
        self._text_stem_mapping = []
        for text in texts:
            text_context = context.get(text, {})
            complete_stems = text_context.get("complete_stems", [])
            for stem in complete_stems:
                self._text_stem_mapping.append((text, stem))
        
        # Create prompts
        prompts = []
        for encoded in encoded_pairs:
            data = json.loads(encoded)
            prompt = stem_your_feature_prompt(
                data["text"], data["stem"], data["definitions"]
            )
            prompts.append(prompt)
        
        return prompts
    
    def post_process(
        self, results: Dict[str, Any], context: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reorganize results by text -> stem -> result.
        """
        stem_results_dict = {}
        
        # Group by text
        for idx, (encoded_key, result) in enumerate(results.items()):
            if idx < len(self._text_stem_mapping):
                text, stem = self._text_stem_mapping[idx]
                
                if text not in stem_results_dict:
                    stem_results_dict[text] = {}
                
                # Convert to dict
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = result
                
                # Store result
                if result_dict and result_dict.get("has_feature"):
                    stem_results_dict[text][stem] = result_dict.get("feature_result", {})
                else:
                    stem_results_dict[text][stem] = None
        
        return stem_results_dict
    
    def get_result_key(self) -> str:
        """Key for storing in output."""
        return "stem_your_features"
    
    def format_for_export(self, result: Any) -> Any:
        """Format for export."""
        if result is None:
            return {}
        return result
```

### Step 4: Register the Capability

**Update `classifier/capabilities/__init__.py`:**
```python
from .your_feature import YourFeatureCapability
from .stem_your_feature import StemYourFeatureCapability

__all__ = [
    # ... existing exports ...
    "YourFeatureCapability",
    "StemYourFeatureCapability",
]
```

**Update `classifier/capabilities/registry.py`:**
```python
def create_default_registry() -> CapabilityRegistry:
    from .your_feature import YourFeatureCapability
    
    registry = CapabilityRegistry()
    # ... existing registrations ...
    registry.register(YourFeatureCapability())  # Global capabilities go here
    
    # Stem capabilities registered in server with parameters
    return registry
```

**Update `classifier_server.py`:**
```python
from classifier.capabilities import (
    # ... existing imports ...
    StemYourFeatureCapability,
)

def serve(...):
    # ... existing code ...
    
    state.registry = create_default_registry()
    # Register stem capabilities
    state.registry.register(
        StemYourFeatureCapability(max_stem_definitions=max_stem_definitions)
    )
```

---

## Example: Adding Trend Capabilities

Let's walk through the complete implementation of both trend capabilities.

### Models (`models.py`)

```python
class TrendResult(BaseModel):
    trend_category: Literal["current_state", "improved", "declined", "no_change"]
    confidence: int  # 1-5
    reasoning: str = ""
    excerpt: str = ""

class TrendOutput(BaseModel):
    has_trend: bool
    trend_result: Optional[TrendResult] = None

class StemTrendOutput(BaseModel):
    has_trend: bool
    trend_result: Optional[StemTrendResult] = None
```

### Global Trend Capability

**File: `classifier/capabilities/trend.py`**
- Detects temporal change language in entire text
- No dependencies
- Simple implementation following `RecommendationsCapability` pattern

### Stem Trend Capability  

**File: `classifier/capabilities/stem_trend.py`**
- Analyzes trends per classification stem
- Depends on classification
- Complex implementation following `StemRecommendationsCapability` pattern

---

## Testing Your Capability

### 1. Unit Testing

Create `tests/test_your_capability.py`:

```python
def test_your_capability():
    from classifier.capabilities import YourFeatureCapability
    from classifier.models import YourFeatureOutput
    
    cap = YourFeatureCapability()
    
    # Test name
    assert cap.name == "your_feature"
    
    # Test schema
    assert cap.schema == YourFeatureOutput
    
    # Test prompt generation
    prompt = cap.create_prompt("test text")
    assert "test text" in prompt
```

### 2. Integration Testing

Start the server and make requests:

```bash
# Start server
python classifier_server.py --config topics.json

# Test global capability
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Your test text"],
    "capabilities": ["your_feature"]
  }'

# Test stem capability
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Your test text"],
    "capabilities": ["classification", "stem_your_feature"]
  }'
```

### 3. Validation Checklist

- [ ] Capability appears in `/capabilities` endpoint
- [ ] Can execute capability alone
- [ ] Can execute with other capabilities
- [ ] Dependencies resolve correctly
- [ ] Output format matches schema
- [ ] Prompts generate valid LLM responses
- [ ] Results export to JSON correctly

---

## Troubleshooting

### Common Issues

**1. "Unknown capability" error**
- Check capability is registered in `create_default_registry()` or server startup
- Verify imports in `__init__.py`
- Restart server after changes

**2. "Capability requires classification context" error**
- Stem capabilities need classification to run first
- Ensure dependencies list includes `"classification"`
- Check orchestrator executes capabilities in order

**3. Schema validation errors**
- LLM output doesn't match your Pydantic schema
- Improve prompt clarity and examples
- Test prompt with LLM separately
- Consider making fields optional

**4. Empty results**
- Check `prepare_batch()` returns non-empty list
- Verify `post_process()` correctly reorganizes results
- Add debug logging to track data flow

**5. Performance issues**
- Batch size may be too large
- Consider limiting `max_stem_definitions`
- Profile with `/stats` endpoint

### Debug Tips

1. **Add logging:**
```python
from rich.console import Console
console = Console()

console.print(f"[cyan]Processing {len(texts)} texts[/cyan]")
```

2. **Test prompts independently:**
```python
prompt = your_capability.create_prompt("test text")
print(prompt)
# Copy to LLM interface and verify output
```

3. **Validate schemas:**
```python
from pydantic import ValidationError
try:
    result = YourOutput(**llm_response)
except ValidationError as e:
    print(e)
```

4. **Check capability execution order:**
```bash
# Watch server logs for execution order
# Look for: "Execution order: classification → your_capability"
```

---

## Best Practices

### Design
- Start simple, add complexity only if needed
- Prefer global capabilities unless stem-specific analysis is essential
- Write clear, testable prompts with diverse examples
- Use confidence scores for uncertain outputs

### Implementation
- Follow existing capability patterns closely
- Reuse helper functions (e.g., `extract_stem_definitions`)
- Add comprehensive docstrings
- Include type hints everywhere

### Testing
- Test with varied real-world examples
- Verify edge cases (empty text, no classifications, etc.)
- Check performance with large batches
- Validate JSON output structure

### Documentation
- Document capability purpose and use cases
- Explain key configuration parameters
- Provide example API requests
- Note any limitations or constraints

---

## Additional Resources

- **Base Capability Class**: `classifier/capabilities/base.py`
- **Example Capabilities**: 
  - Simple: `recommendations.py`, `alerts.py`
  - Complex: `stem_recommendations.py`, `stem_polarity.py`
- **Server Manager**: `classifier_server_manager.py` (orchestration logic)
- **API Documentation**: Check `/docs` endpoint when server is running

---

**Questions or Issues?**

Check existing capabilities for reference implementations, or refer to the server logs for detailed execution traces.

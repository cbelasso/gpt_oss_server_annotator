"""
Prompt templates for hierarchical text classification.

This module contains various prompt generation functions that can be easily
swapped or customized for different classification tasks.
"""

from typing import Any, Dict, List


def standard_classification_prompt(node_config: Dict[str, Any]) -> str:
    """
    Generate a standard single-node classification prompt.

    This prompt focuses on binary classification of whether a text discusses
    the target topic as a main subject (not just in passing).

    Args:
        node_config: Dictionary containing node configuration with keys:
            - name: Topic name
            - description: Detailed topic description
            - keywords: List of relevant keywords
            - scope: Scope definition

    Returns:
        Formatted prompt string for the LLM
    """
    name = node_config.get("name", "[No Name]")
    description = node_config.get("description", "[No Description]")
    keywords = ", ".join(node_config.get("keywords", [])) or "[None]"
    scope = node_config.get("scope", "[None]")

    return f"""
You are an expert binary classifier. You are classifying an employee comment from a post-training survey.
The goal is to determine if a specific target topic — {name} — appears as one of the **main topics** in the comment.

**Topic**: {name}
**Description**: {description}
**Keywords**: {keywords}
**Scope**: {scope}

# Key Rules
- Use only the information **explicitly present** in the comment and the topic definition above.
- DO NOT infer intent, sentiment, or unstated implications.
- DO NOT guess based on general knowledge, stereotypes, or assumptions.
- A "main topic" means the comment directly and substantially discusses it, not just a brief or indirect mention.
- If evidence is insufficient, classify as False.

# Output
Return a JSON object with the following fields:
- **classification**: True or False
- **reasoning**: One or two sentences. It must:
    * Directly reference or quote the relevant part(s) of the comment.
    * Only use explicit evidence from the text and topic definition.
    * Avoid general statements not grounded in the comment.
- **confidence**: Integer 1–5, where:
    1 = Very uncertain, weak or no evidence
    2 = Low confidence, weak evidence
    3 = Moderate confidence, ambiguous or partial evidence
    4 = High confidence, clear direct evidence
    5 = Very certain, strong direct evidence
- **excerpt**: Exact span of the comment that supports the classification if True, otherwise empty string.
"""


def hierarchical_path_prompt(node_path: List[Dict[str, Any]]) -> str:
    """
    Generate a prompt for evaluating multiple nodes along a hierarchical path.

    This prompt asks the LLM to evaluate relevance for each node in a path
    simultaneously, which can be more efficient for full-path evaluation.

    Args:
        node_path: List of node configurations from root to leaf

    Returns:
        Formatted prompt string for multi-node evaluation
    """
    path_str = " > ".join([n.get("name", "[No Name]") for n in node_path])

    nodes_info = []
    for n in node_path:
        node_name = n.get("name", "[No Name]")
        description = n.get("description", "[No Description]")
        keywords = ", ".join(n.get("keywords", [])) or "[None]"
        nodes_info.append(f"- {node_name} (Keywords: {keywords})\n  Description: {description}")

    nodes_block = "\n".join(nodes_info)

    return f"""
You are an expert multi-level classifier. Evaluate the following text and determine whether it is relevant to each of the nodes along this hierarchical path:

Hierarchy path: {path_str}
Nodes:
{nodes_block}

Instructions:
- A node is considered relevant if the text addresses it as a **main topic**, not just in passing.
- Multiple nodes along the path may be relevant.
- Return a JSON object with one entry per node containing:
  - "name": node name
  - "is_relevant": true/false
  - "confidence": 1-5 (1=very uncertain, 5=very certain)
  - "reasoning": 1-2 sentences explaining the decision
  - "excerpt": text snippet supporting the decision, empty string if not relevant

Example JSON:
[
    {{"name": "Course Component", "is_relevant": true, "confidence": 5, "reasoning": "Text discusses lessons and modules.", "excerpt": "The course modules included ..."}},
    {{"name": "Course Material & Structure", "is_relevant": true, "confidence": 4, "reasoning": "Mentions organization of materials.", "excerpt": "The lesson plans were ..."}}
]
"""


def sentiment_aware_classification_prompt(node_config: Dict[str, Any]) -> str:
    """
    Generate a classification prompt that also considers sentiment.

    This variant asks the LLM to note whether the discussion is positive,
    negative, or neutral, which can be useful for feedback analysis.

    Args:
        node_config: Dictionary containing node configuration

    Returns:
        Formatted prompt string with sentiment analysis
    """
    name = node_config.get("name", "[No Name]")
    description = node_config.get("description", "[No Description]")
    keywords = ", ".join(node_config.get("keywords", [])) or "[None]"

    return f"""
You are an expert binary classifier analyzing employee feedback.
Determine if the topic "{name}" is a main subject of discussion, and if so, assess the sentiment.

**Topic**: {name}
**Description**: {description}
**Keywords**: {keywords}

# Key Rules
- Use only explicit information in the comment
- A "main topic" means substantial, direct discussion
- Sentiment should be: positive, negative, neutral, or mixed

# Output
Return a JSON object with:
- **classification**: True or False
- **reasoning**: Brief explanation referencing the text
- **confidence**: Integer 1–5
- **excerpt**: Supporting text span
- **sentiment**: (Only if relevant) One of: positive, negative, neutral, mixed
"""


def keyword_focused_prompt(node_config: Dict[str, Any]) -> str:
    """
    Generate a prompt that emphasizes keyword matching.

    This variant is stricter about requiring explicit keyword presence,
    useful when you want high precision.

    Args:
        node_config: Dictionary containing node configuration

    Returns:
        Formatted prompt string emphasizing keywords
    """
    name = node_config.get("name", "[No Name]")
    description = node_config.get("description", "[No Description]")
    keywords = ", ".join(node_config.get("keywords", [])) or "[None]"

    return f"""
You are a precise keyword-based classifier.
Determine if the text discusses "{name}" based on explicit keyword presence and topic relevance.

**Topic**: {name}
**Description**: {description}
**Required Keywords**: {keywords}

# Key Rules
- The text should mention at least one keyword OR clearly discuss the topic using synonyms
- Brief mentions don't count unless they're central to the comment
- Be conservative: when in doubt, classify as False

# Output
Return a JSON object with:
- **classification**: True or False
- **reasoning**: Explain which keywords were found or why topic is present
- **confidence**: Integer 1–5
- **excerpt**: Exact span containing evidence
"""


def add_text_to_prompt(prompt: str, text: str) -> str:
    """
    Append the text to classify to a prompt template.

    Args:
        prompt: The base prompt template
        text: The text to be classified

    Returns:
        Complete prompt with text appended
    """
    return f"{prompt}\n\nText:\n{text}"


def recommendations_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect, extract, paraphrase, and qualify recommendations from text.

    This version extends the prior version by adding a 'qualifier' field that
    categorizes the type or direction of the recommendation (e.g., add, reduce, modify, maintain, etc.).
    """
    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey.
The goal is to identify whether the comment contains **recommendations** — actionable suggestions, advice, proposals, or requests for change that the author is making — and to determine the **type of recommendation** being made.

**Comment**: {text}

---

# What Counts as a Recommendation
A recommendation is:
- An actionable suggestion for improvement, change, or future action
- Advice on what should or could be done differently
- A proposal for a specific approach, policy, or practice
- A request for something to be added, removed, or modified
- Constructive feedback that implies "we/you/they should do X"

Examples of recommendations:
- "The training should include more hands-on exercises"
- "I suggest we cover advanced topics in a separate session"
- "It would be helpful to provide reference materials afterward"
- "Consider adding more time for Q&A"

---

# What Does NOT Count as a Recommendation
- General positive/negative statements without actionable content ("It was great", "I didn't like it")
- Pure descriptions of current state without suggesting change
- Questions without implied suggestions
- Purely personal preferences without actionable proposals

---

# Key Rules
- Use only the information **explicitly present** in the comment.
- DO NOT infer unstated recommendations or read between the lines.
- DO NOT count vague dissatisfaction as a recommendation unless it includes a clear actionable suggestion.
- Identify **all distinct** recommendations in the comment.
- Each recommendation must correspond to a specific identifiable excerpt.
- If multiple recommendations appear in one sentence, extract them as separate items if they suggest different actions.
- If the comment is purely descriptive, observational, or expresses sentiment without actionable content, return has_recommendations: false.

---

# Qualifier Types (Direction of Change)

For each recommendation, assign one **qualifier** from the list below based on the intended action:

| Qualifier | Meaning | Example Phrases | Example |
|------------|----------|----------------|----------|
| **add_or_increase** | Suggests adding or increasing something | "Add", "Include", "More", "Increase" | "Add more examples" |
| **reduce_or_remove** | Suggests reducing or removing something | "Reduce", "Fewer", "Too much", "Cut down" | "Reduce lecture time" |
| **introduce_or_start** | Suggests starting something new | "Start", "Implement", "Offer", "Provide" | "Start offering online sessions" |
| **eliminate_or_stop** | Suggests stopping or discontinuing something | "Stop", "Eliminate", "Don't" | "Stop using group exercises" |
| **modify_or_improve** | Suggests changing or improving something existing | "Make it more", "Change", "Improve", "Adjust" | "Make the slides clearer" |
| **maintain_or_continue** | Suggests keeping something as is or reinforcing it | "Keep", "Continue", "Maintain" | "Keep the interactive activities" |
| **unspecified_or_general** | Suggests improvement without clear direction | "Should be better", "Needs improvement" | "The course should improve overall" |

---

# Output Format
Return a JSON object with:
- **has_recommendations**: Boolean (true if at least one recommendation is present, false otherwise)
- **recommendations**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the recommendation.
    * **reasoning**: One sentence explaining why this qualifies as a recommendation and what action is being suggested.
    * **paraphrased_recommendation**: A clear, concise restatement of the recommendation in natural language.
    * **qualifier**: One of the following literals:
      "add_or_increase", "reduce_or_remove", "introduce_or_start",
      "eliminate_or_stop", "modify_or_improve", "maintain_or_continue",
      or "unspecified_or_general".

---

# Examples

**Example 1:**
Comment: "The session was too short. We need more time to practice the concepts."
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "excerpt": "We need more time to practice the concepts",
      "reasoning": "Suggests extending the duration of training to allow for more practice.",
      "paraphrased_recommendation": "Extend the training sessions to provide additional practice time.",
      "qualifier": "add_or_increase"
    }}
  ]
}}

---

**Example 2:**
Comment: "I really enjoyed the training. The instructor was excellent."
Output:
{{
  "has_recommendations": false,
  "recommendations": []
}}

---

**Example 3:**
Comment: "The material was confusing and moved too quickly. It would help to slow down and add more examples, especially for the advanced topics."
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "excerpt": "It would help to slow down",
      "reasoning": "Recommends reducing the pace of content delivery.",
      "paraphrased_recommendation": "Deliver the material at a slower pace to improve comprehension.",
      "qualifier": "reduce_or_remove"
    }},
    {{
      "excerpt": "add more examples, especially for the advanced topics",
      "reasoning": "Suggests adding examples to enhance understanding, particularly for complex sections.",
      "paraphrased_recommendation": "Include more examples, especially for advanced topics.",
      "qualifier": "add_or_increase"
    }}
  ]
}}

---

**Example 4:**
Comment: "We should have more breakout sessions for group work. Also, it would be great to get the slides beforehand, and maybe include a follow-up session next month."
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "excerpt": "We should have more breakout sessions for group work",
      "reasoning": "Recommends increasing opportunities for collaborative learning.",
      "paraphrased_recommendation": "Add more breakout sessions to encourage group collaboration.",
      "qualifier": "add_or_increase"
    }},
    {{
      "excerpt": "it would be great to get the slides beforehand",
      "reasoning": "Suggests providing learning materials before the training begins.",
      "paraphrased_recommendation": "Distribute slides in advance of the session.",
      "qualifier": "introduce_or_start"
    }},
    {{
      "excerpt": "include a follow-up session next month",
      "reasoning": "Proposes scheduling a continuation session for reinforcement.",
      "paraphrased_recommendation": "Hold a follow-up session next month for continued learning.",
      "qualifier": "introduce_or_start"
    }}
  ]
}}
"""


def alert_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect and categorize alerts from employee comments.

    Alerts are serious concerns, complaints, or reports of problematic situations
    that require attention from HR, management, or compliance teams.
    """
    return f"""
You are an expert text analyzer specializing in workplace safety and compliance. You are analyzing an employee comment from a post-training survey.
The goal is to identify whether the comment contains **alerts** — serious concerns, complaints, or reports of problematic workplace situations that require immediate attention or follow-up from HR, management, or compliance teams.

**Comment**: {text}

---

# What Counts as an Alert
An alert is:
- A report or complaint about discrimination, harassment, or hostile behavior
- A concern about safety, violence, or threatening situations
- A description of ethical violations, fraud, or policy breaches
- A disclosure of mental health crises or substance abuse issues
- A report of retaliation for protected activities
- Any serious workplace concern that could expose the organization to legal, safety, or reputational risk

Examples of alerts:
- "My manager makes inappropriate comments about my appearance"
- "I witnessed someone being excluded from meetings because of their race"
- "There's a safety hazard in the warehouse that hasn't been addressed"
- "I was told not to report the incident or I'd lose my job"
- "A colleague threatened me after I disagreed with them"

---

# What Does NOT Count as an Alert
- General complaints about workload, stress, or job satisfaction (unless they indicate serious mental health crisis)
- Constructive feedback or improvement suggestions (these are recommendations, not alerts)
- Minor interpersonal conflicts without harassment or discrimination elements
- General dissatisfaction with training, management style, or policies (unless indicating serious violations)
- Requests for resources or process improvements

---

# Key Rules
- Use only the information **explicitly present** in the comment.
- DO NOT infer alerts from vague dissatisfaction — there must be clear indication of a serious concern.
- DO NOT escalate routine complaints into alerts unless they describe genuinely problematic situations.
- Identify **all distinct** alerts in the comment.
- Each alert must correspond to a specific identifiable excerpt.
- If multiple distinct concerns appear, extract them as separate alert items.
- If the comment contains only routine feedback or minor complaints, return has_alerts: false.
- **Be especially vigilant** for subtle indicators of harassment, discrimination, or retaliation that may be worded indirectly.

---

# Alert Types

For each alert, assign one **alert_type** from the list below:

| Alert Type | Description | Example Indicators |
|------------|-------------|-------------------|
| **discrimination** | Unfair treatment based on protected characteristics (race, gender, age, religion, disability, etc.) | "treated differently because of", "excluded due to", "overlooked for promotion because I'm" |
| **sexual_harassment** | Unwanted sexual advances, requests for sexual favors, or other verbal/physical harassment of a sexual nature | "inappropriate comments about appearance", "unwanted touching", "sexual jokes", "asked me out repeatedly" |
| **workplace_violence** | Physical violence, threats of violence, or intimidating behavior | "threatened", "physical altercation", "afraid for my safety", "aggressive behavior" |
| **safety_concern** | Hazardous conditions, unsafe practices, or violations of safety protocols | "unsafe equipment", "no safety training", "hazardous materials", "accident waiting to happen" |
| **ethical_violation** | Fraud, corruption, misuse of company resources, or violations of ethical standards | "falsified records", "misused funds", "asked to lie", "cover up" |
| **hostile_environment** | Pervasive pattern of intimidation, ridicule, or offensive conduct that creates an abusive atmosphere | "constantly belittled", "toxic atmosphere", "everyone gangs up on", "culture of fear" |
| **retaliation** | Adverse action taken against someone for reporting concerns or exercising legal rights | "punished for speaking up", "demoted after complaining", "told not to report" |
| **bullying** | Repeated mistreatment, verbal abuse, or deliberate sabotage by colleagues or supervisors | "constantly criticized in front of others", "undermined my work", "publicly humiliated" |
| **substance_abuse** | Concerns about drug or alcohol use affecting workplace safety or performance | "came to work intoxicated", "using drugs on site", "impaired while operating equipment" |
| **mental_health_crisis** | Indicators of severe mental distress, self-harm risk, or crisis requiring immediate support | "thinking about ending it all", "can't go on", "severe depression", "suicidal thoughts" |
| **data_breach** | Unauthorized access to, disclosure of, or loss of confidential information | "customer data exposed", "security protocols violated", "leaked confidential information" |
| **fraud** | Intentional deception for financial or personal gain | "stealing from company", "fake invoices", "embezzlement", "kickback scheme" |
| **other_serious_concern** | Any other serious workplace issue requiring urgent attention that doesn't fit other categories | Various situations requiring escalation |

---

# Severity Levels

For each alert, assign a **severity** level:

| Severity | When to Use |
|----------|-------------|
| **low** | Minor concerns that should be documented but don't require immediate action |
| **moderate** | Concerns requiring timely follow-up and investigation |
| **high** | Serious issues requiring prompt attention and intervention |
| **critical** | Imminent threats to safety, ongoing criminal activity, or situations requiring immediate response |

Guidelines:
- **sexual_harassment**, **discrimination**, **workplace_violence**, **retaliation**: Usually moderate to critical depending on specifics
- **mental_health_crisis** with self-harm indicators: Critical
- **safety_concern** with imminent danger: High to critical
- **bullying**, **hostile_environment**: Typically moderate to high
- Use context clues about severity, frequency, and impact

---

# Output Format
Return a JSON object with:
- **has_alerts**: Boolean (true if at least one alert is present, false otherwise)
- **alerts**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the alert.
    * **reasoning**: One sentence explaining why this qualifies as an alert and what concern it raises.
    * **alert_type**: One of the alert type literals listed above.
    * **severity**: One of: "low", "moderate", "high", or "critical".

---

# Examples

**Example 1:**
Comment: "My supervisor constantly makes jokes about my accent and asks where I'm 'really' from. It makes me uncomfortable."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "My supervisor constantly makes jokes about my accent and asks where I'm 'really' from",
      "reasoning": "Describes repeated comments targeting national origin or ethnicity, indicating potential discrimination.",
      "alert_type": "discrimination",
      "severity": "high"
    }}
  ]
}}

---

**Example 2:**
Comment: "The training was too long and the room was cold. I think we need better scheduling."
Output:
{{
  "has_alerts": false,
  "alerts": []
}}

---

**Example 3:**
Comment: "I reported safety concerns about the machinery to my manager, but nothing was done. Now I'm being assigned to less desirable shifts and I think it's because I spoke up."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "I reported safety concerns about the machinery to my manager, but nothing was done",
      "reasoning": "Reports unaddressed safety hazards that could lead to workplace injuries.",
      "alert_type": "safety_concern",
      "severity": "high"
    }},
    {{
      "excerpt": "Now I'm being assigned to less desirable shifts and I think it's because I spoke up",
      "reasoning": "Describes potential retaliation for reporting safety concerns, a protected activity.",
      "alert_type": "retaliation",
      "severity": "high"
    }}
  ]
}}

---

**Example 4:**
Comment: "One of my coworkers came to work clearly intoxicated yesterday and was operating heavy equipment. I'm worried someone will get hurt."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "One of my coworkers came to work clearly intoxicated yesterday and was operating heavy equipment",
      "reasoning": "Reports substance abuse combined with operation of dangerous equipment, creating immediate safety risk.",
      "alert_type": "substance_abuse",
      "severity": "critical"
    }}
  ]
}}

---

**Example 5:**
Comment: "The training didn't cover the new software features we need. I recommend adding a module on the advanced tools."
Output:
{{
  "has_alerts": false,
  "alerts": []
}}

---

**Example 6:**
Comment: "My team leader yells at me in front of everyone when I make mistakes and has said I'm too stupid to do this job. This happens almost daily."
Output:
{{
  "has_alerts": true,
  "alerts": [
    {{
      "excerpt": "My team leader yells at me in front of everyone when I make mistakes and has said I'm too stupid to do this job. This happens almost daily.",
      "reasoning": "Describes repeated public humiliation and verbal abuse creating a hostile work environment.",
      "alert_type": "bullying",
      "severity": "high"
    }}
  ]
}}

---

Now analyze the comment and return your response.
"""


def stem_recommendations_prompt(
    text: str, stem_path: str, stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate a prompt to evaluate what types of recommendations apply to a specific classification stem.

    Args:
        text: The original text being analyzed
        stem_path: The classification path/stem (e.g., "Teaching Effectiveness>Teaching Style>Explanations")
        stem_definitions: Optional list of definition dicts for each node in the stem path
    """

    # Format definitions section if provided
    definitions_section = ""
    if stem_definitions:
        definitions_section = "\n**Topic Path Definitions:**\n\n"
        for node_info in stem_definitions:
            definitions_section += f"**{node_info['name']}**\n"
            if node_info.get("definition"):
                definitions_section += f"- Definition: {node_info['definition']}\n"
            if node_info.get("description"):
                definitions_section += f"- Description: {node_info['description']}\n"
            if node_info.get("keywords"):
                keywords_str = ", ".join(
                    node_info["keywords"][:10]
                )  # Limit to first 10 keywords
                definitions_section += f"- Keywords: {keywords_str}\n"
            definitions_section += "\n"

    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey that has been classified under a specific topic path.

**Comment**: {text}

**Topic Path**: {stem_path}

{definitions_section}---

# Your Task
Determine which types of recommendations apply to this specific topic path based on the comment. The comment has already been classified as relevant to this topic. Now, identify what kind of recommendation (if any) the author is making about this specific topic.

Use the topic definitions above to understand what each level of the path represents and ensure your recommendations are aligned with the actual meaning of the topic.

---

# Recommendation Types

Evaluate whether the comment suggests any of the following actions regarding the topic:

| Type | Meaning | Example Phrases |
|------|---------|----------------|
| **start** | Begin doing something new that isn't currently done | "Should start", "Need to begin", "Introduce" |
| **stop** | Cease or discontinue something currently being done | "Should stop", "Eliminate", "Don't do", "Remove" |
| **do_more** | Increase or expand something that's already being done | "More of", "Increase", "Expand", "Add more" |
| **do_less** | Reduce or decrease something that's currently being done | "Less of", "Reduce", "Too much", "Cut back" |
| **continue** | Keep doing something that's working well | "Keep doing", "Continue", "Maintain", "Don't change" |
| **change** | Modify or alter how something is currently done | "Change how", "Do differently", "Improve", "Adjust" |

---

# Key Rules
- A comment may contain **multiple recommendation types** for the same topic path
- Only identify recommendations that are **explicitly stated** in the comment
- Each recommendation type must have a corresponding excerpt that supports it
- The excerpt must be directly relevant to the topic path
- Focus on what is being recommended **for this specific topic**, not other topics
- If no recommendations are present for this topic, return has_recommendations: false

---

# Output Format
Return a JSON object with:
- **has_recommendations**: Boolean (true if at least one recommendation type applies, false otherwise)
- **recommendations**: List of objects, each containing:
    * **recommendation_type**: One of: "start", "stop", "do_more", "do_less", "continue", "change"
    * **excerpt**: The exact text span from the comment that indicates this recommendation type
    * **reasoning**: One sentence explaining why this recommendation type applies to this topic path
    * **paraphrased_recommendation**: A clear, concise restatement of the recommendation in natural language.

---

# Examples

**Example 1:**
Comment: "The instructor should provide more real-world examples during lectures."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "recommendation_type": "do_more",
      "excerpt": "should provide more real-world examples",
      "reasoning": "Suggests increasing the use of real-world examples in the teaching approach.",
      "paraphrased_recommendation": "Increase the use of real-world examples in lectures."
    }}
  ]
}}

---

**Example 2:**
Comment: "The training was excellent and very helpful."
Topic Path: "Course Content>Relevance"
Output:
{{
  "has_recommendations": false,
  "recommendations": []
}}

---

**Example 3:**
Comment: "We should stop using PowerPoint slides and start incorporating more interactive activities. The current lecture format needs to be more engaging."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "recommendation_type": "stop",
      "excerpt": "should stop using PowerPoint slides",
      "reasoning": "Recommends discontinuing the use of PowerPoint slides in teaching.",
      "paraphrased_recommendation": "Discontinue the use of PowerPoint slides in teaching."
    }},
    {{
      "recommendation_type": "start",
      "excerpt": "start incorporating more interactive activities",
      "reasoning": "Suggests beginning to use interactive activities that aren't currently part of the teaching style.",
      "paraphrased_recommendation": "Begin incorporating interactive activities into the teaching approach."
    }},
    {{
      "recommendation_type": "change",
      "excerpt": "current lecture format needs to be more engaging",
      "reasoning": "Indicates the teaching approach should be modified to increase engagement.",
      "paraphrased_recommendation": "Modify the lecture format to make it more engaging."
    }}
  ]
}}

---

**Example 4:**
Comment: "Keep the hands-on exercises exactly as they are - they're perfect and really help with learning."
Topic Path: "Teaching Effectiveness>Practical Application"
Output:
{{
  "has_recommendations": true,
  "recommendations": [
    {{
      "recommendation_type": "continue",
      "excerpt": "Keep the hands-on exercises exactly as they are",
      "reasoning": "Recommends maintaining the current approach to hands-on exercises without changes.",
      "paraphrased_recommendation": "Maintain the current hands-on exercises without modification."
    }}
  ]
}}
"""


def stem_polarity_prompt(
    text: str, stem_path: str, stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate a prompt to evaluate the polarity/sentiment of a comment towards a specific classification stem.

    Args:
        text: The original text being analyzed
        stem_path: The classification path/stem (e.g., "Teaching Effectiveness>Teaching Style>Explanations")
        stem_definitions: Optional list of definition dicts for each node in the stem path
    """

    # Format definitions section if provided
    definitions_section = ""
    if stem_definitions:
        definitions_section = "\n**Topic Path Definitions:**\n\n"
        for node_info in stem_definitions:
            definitions_section += f"**{node_info['name']}**\n"
            if node_info.get("definition"):
                definitions_section += f"- Definition: {node_info['definition']}\n"
            if node_info.get("description"):
                definitions_section += f"- Description: {node_info['description']}\n"
            if node_info.get("keywords"):
                keywords_str = ", ".join(node_info["keywords"][:10])
                definitions_section += f"- Keywords: {keywords_str}\n"
            definitions_section += "\n"

    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey that has been classified under a specific topic path.

**Comment**: {text}

**Topic Path**: {stem_path}

{definitions_section}---

# Your Task
Determine the overall polarity (sentiment) of the comment specifically toward this topic path. The comment has already been classified as relevant to this topic. Now, identify whether the author's stance toward this specific topic is positive, negative, neutral, or mixed.

Use the topic definitions above to understand what each level of the path represents and ensure your polarity assessment is aligned with the actual meaning of the topic.

---

# Polarity Types

| Polarity | Meaning | Examples |
|----------|---------|----------|
| **Positive** | Expresses satisfaction, praise, approval, or positive sentiment toward the topic | "Great explanations", "Very helpful", "Loved this aspect", "Excellent approach" |
| **Negative** | Expresses dissatisfaction, criticism, disapproval, or negative sentiment toward the topic | "Poor explanations", "Confusing", "Needs improvement", "Didn't like this" |
| **Neutral** | Factual observation or description without clear positive or negative sentiment | "The class covered this topic", "Explanations were provided", "This was included" |
| **Mixed** | Contains both positive and negative sentiments toward the same topic | "Good examples but too fast", "Clear but too technical", "Helpful but needs more" |

---

# Key Rules
- Assess polarity **only for this specific topic path**, not the overall comment
- Distinguish between constructive criticism (Negative) and balanced feedback (Mixed)
- If the comment mentions both strengths and weaknesses about the topic, classify as Mixed
- Pure suggestions for improvement without negative sentiment should be Negative (constructive criticism)
- Expressions of gratitude or appreciation are Positive even if brief
- If the comment is purely descriptive without evaluative language, classify as Neutral
- Provide a confidence score from 1 (low) to 5 (high) for your assessment
- Extract the specific excerpt that best supports your polarity determination

---

# Output Format
Return a JSON object with:
- **has_polarity**: Boolean (true if polarity can be determined, false otherwise)
- **polarity_result**: Object containing (only if has_polarity is true):
    * **polarity**: One of: "Positive", "Negative", "Neutral", "Mixed"
    * **confidence**: Integer from 1 to 5 indicating confidence in the assessment
    * **reasoning**: One sentence explaining why this polarity was assigned to this topic
    * **excerpt**: The text span from the comment that best indicates this polarity

---

# Examples

**Example 1:**
Comment: "The instructor provided excellent explanations that made complex topics easy to understand."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Positive",
    "confidence": 5,
    "reasoning": "The comment expresses clear praise for the explanations, describing them as 'excellent' and effective at making topics 'easy to understand'.",
    "excerpt": "excellent explanations that made complex topics easy to understand"
  }}
}}

---

**Example 2:**
Comment: "The explanations were confusing and didn't help me understand the material at all."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Negative",
    "confidence": 5,
    "reasoning": "The comment expresses clear dissatisfaction with the explanations, describing them as 'confusing' and ineffective.",
    "excerpt": "explanations were confusing and didn't help me understand the material"
  }}
}}

---

**Example 3:**
Comment: "The explanations were clear and well-structured, but they moved too quickly for me to keep up."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Mixed",
    "confidence": 4,
    "reasoning": "The comment contains both positive sentiment ('clear and well-structured') and negative sentiment ('moved too quickly') about the explanations.",
    "excerpt": "explanations were clear and well-structured, but they moved too quickly"
  }}
}}

---

**Example 4:**
Comment: "The course covered various teaching methods."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Neutral",
    "confidence": 5,
    "reasoning": "The comment is purely descriptive without any evaluative or emotional language regarding the teaching style.",
    "excerpt": "The course covered various teaching methods"
  }}
}}

---

**Example 5:**
Comment: "The professor should provide more examples to help clarify concepts."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_polarity": true,
  "polarity_result": {{
    "polarity": "Negative",
    "confidence": 4,
    "reasoning": "The suggestion for improvement implies current explanations are insufficient, indicating constructive criticism (negative sentiment).",
    "excerpt": "should provide more examples to help clarify concepts"
  }}
}}
"""


def trends_detection_prompt(text: str) -> str:
    """
    Generate a prompt to detect, extract, and classify trends in feedback text.

    A trend represents a temporal comparison indicating change (or stability) over time,
    with both a direction component and a valence (positive/negative sentiment).
    """
    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey.
The goal is to identify whether the comment contains **trends** — statements that indicate change or stability over time — and to characterize both the **direction** and **valence** (positive/negative nature) of each trend.

**Comment**: {text}

---

# What Counts as a Trend
A trend is a statement that indicates:
- A temporal comparison between past and present states
- An observation of change (or consistency) over time
- An evolution or trajectory of some aspect

**Trends require temporal markers** such as:
- Explicit time references: "last year", "in 2023", "this quarter", "recently"
- Comparative language: "used to be", "no longer", "has become", "still"
- Change verbs: "improved", "declined", "increased", "decreased", "remains"
- Sequential language: "now", "anymore", "these days", "currently"

Examples of trends:
- "My manager used to be caring, but not anymore" (declining)
- "The quality has improved significantly since last year" (improving)
- "Training remains excellent as always" (stable/positive)
- "Support has been inconsistent lately" (fluctuating)

---

# What Does NOT Count as a Trend
- **Static descriptions** without temporal comparison: "The training is good"
- **Comparisons to expectations** (not temporal): "This was worse than I expected"
- **Recommendations for future**: "We should improve this" (use recommendations detection instead)
- **One-time events**: "The last session was cancelled" (no pattern/trajectory)

---

# Key Rules
- Use only the information **explicitly present** in the comment.
- A trend **requires** at least a moderate temporal marker (explicit or clearly implicit).
- DO NOT infer trends from purely static statements.
- Identify **all distinct** trends in the comment.
- Each trend must correspond to a specific identifiable excerpt.
- If multiple trends appear, extract them as separate items if they describe different aspects or directions.

---

# Direction Types

For each trend, assign one **direction** from the list below:

| Direction | Meaning | Example Temporal Markers | Example |
|-----------|---------|-------------------------|---------|
| **increasing** | Something is growing, rising, or expanding | "has increased", "more than before", "growing" | "Workload has increased significantly" |
| **decreasing** | Something is shrinking, falling, or contracting | "has decreased", "less than before", "declining" | "Support has declined over time" |
| **improving** | Something is getting better in quality/effectiveness | "has improved", "better than", "getting better" | "Communication has improved recently" |
| **deteriorating** | Something is getting worse in quality/effectiveness | "has worsened", "worse than", "getting worse" | "Morale has deteriorated since the changes" |
| **stable_positive** | Something good remains consistent | "still excellent", "remains strong", "consistently good" | "Quality remains excellent" |
| **stable_negative** | Something bad remains consistent | "still poor", "remains weak", "consistently bad" | "Response time is still slow" |
| **fluctuating** | Something varies or is inconsistent over time | "inconsistent", "varies", "sometimes...sometimes" | "Feedback has been hit or miss lately" |

---

# Valence Types

For each trend, assign one **valence** indicating whether the change/state is viewed positively or negatively:

| Valence | Meaning | When to Use |
|---------|---------|-------------|
| **positive** | The trend is favorable, beneficial, or desirable | Improvements, positive stability, decrease in negative things |
| **negative** | The trend is unfavorable, problematic, or undesirable | Deterioration, negative stability, increase in negative things |
| **neutral** | The trend is neither clearly good nor bad | Factual observations without evaluative content |
| **mixed** | The trend has both positive and negative aspects | Fluctuating quality, or context suggests ambivalence |

**Important**: Valence should be determined from the **commenter's perspective** as expressed in the text.

---

# Confidence Scoring

For each trend, assign a **confidence score** from 1 to 5:

| Score | Meaning | When to Use |
|-------|---------|-------------|
| **5** | Very certain | Explicit temporal markers, clear change, unambiguous direction |
| **4** | Quite certain | Strong temporal indicators, clear trend with minor ambiguity |
| **3** | Moderately certain | Temporal markers present but trend requires some interpretation |
| **2** | Somewhat uncertain | Weak temporal markers or ambiguous direction/valence |
| **1** | Very uncertain | Minimal temporal indicators, highly ambiguous interpretation |

**Confidence Guidelines**:
- Higher confidence for explicit time references ("since 2023", "compared to last year")
- Lower confidence for vague temporal markers ("recently", "these days")
- Higher confidence when direction and valence are both clear
- Lower confidence when trend is implied rather than stated

---

# Output Format
Return a JSON object with:
- **has_trends**: Boolean (true if at least one trend is present, false otherwise)
- **trends**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the trend.
    * **reasoning**: One sentence explaining why this qualifies as a trend and what is changing.
    * **subject**: What aspect is changing (e.g., "manager support", "training quality", "workload"). Keep this concise.
    * **direction**: One of: "increasing", "decreasing", "improving", "deteriorating", "stable_positive", "stable_negative", "fluctuating"
    * **valence**: One of: "positive", "negative", "neutral", "mixed"
    * **confidence**: Integer from 1 to 5

---

# Examples

**Example 1:**
Comment: "My manager used to be caring, but not anymore."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "My manager used to be caring, but not anymore",
      "reasoning": "Indicates a decline in manager's caring behavior over time.",
      "subject": "manager caring behavior",
      "direction": "deteriorating",
      "valence": "negative",
      "confidence": 5
    }}
  ]
}}

---

**Example 2:**
Comment: "The training quality has improved significantly since last year."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The training quality has improved significantly since last year",
      "reasoning": "Shows an improvement in training quality with explicit temporal reference.",
      "subject": "training quality",
      "direction": "improving",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

---

**Example 3:**
Comment: "The workload has increased dramatically, and bureaucracy has decreased."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The workload has increased dramatically",
      "reasoning": "Indicates an increase in workload over time.",
      "subject": "workload",
      "direction": "increasing",
      "valence": "negative",
      "confidence": 4
    }},
    {{
      "excerpt": "bureaucracy has decreased",
      "reasoning": "Shows a reduction in bureaucracy over time.",
      "subject": "bureaucracy",
      "direction": "decreasing",
      "valence": "positive",
      "confidence": 4
    }}
  ]
}}

---

**Example 4:**
Comment: "The instructor remains excellent, just like in previous sessions."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The instructor remains excellent, just like in previous sessions",
      "reasoning": "Indicates consistent high quality over time with explicit temporal comparison.",
      "subject": "instructor quality",
      "direction": "stable_positive",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

---

**Example 5:**
Comment: "Support has been really inconsistent lately - sometimes responsive, sometimes not."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Support has been really inconsistent lately - sometimes responsive, sometimes not",
      "reasoning": "Describes varying quality of support over recent time period.",
      "subject": "support responsiveness",
      "direction": "fluctuating",
      "valence": "mixed",
      "confidence": 4
    }}
  ]
}}

---

**Example 6:**
Comment: "The training was excellent and very helpful."
Output:
{{
  "has_trends": false,
  "trends": []
}}

---

**Example 7:**
Comment: "Things seem to be getting better, I think."
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Things seem to be getting better",
      "reasoning": "Suggests improvement over time, though the subject and certainty are vague.",
      "subject": "general situation",
      "direction": "improving",
      "valence": "positive",
      "confidence": 2
    }}
  ]
}}

---

Now analyze the comment and return the JSON output.
"""


def stem_trends_prompt(
    text: str, stem_path: str, stem_definitions: List[Dict[str, str]] = None
) -> str:
    """
    Generate a prompt to evaluate what types of trends apply to a specific classification stem.

    Args:
        text: The original text being analyzed
        stem_path: The classification path/stem (e.g., "Teaching Effectiveness>Teaching Style>Explanations")
        stem_definitions: Optional list of definition dicts for each node in the stem path
    """

    # Format definitions section if provided
    definitions_section = ""
    if stem_definitions:
        definitions_section = "\n**Topic Path Definitions:**\n\n"
        for node_info in stem_definitions:
            definitions_section += f"**{node_info['name']}**\n"
            if node_info.get("definition"):
                definitions_section += f"- Definition: {node_info['definition']}\n"
            if node_info.get("description"):
                definitions_section += f"- Description: {node_info['description']}\n"
            if node_info.get("keywords"):
                keywords_str = ", ".join(
                    node_info["keywords"][:10]
                )  # Limit to first 10 keywords
                definitions_section += f"- Keywords: {keywords_str}\n"
            definitions_section += "\n"

    return f"""
You are an expert text analyzer. You are analyzing an employee comment from a post-training survey that has been classified under a specific topic path.

**Comment**: {text}

**Topic Path**: {stem_path}

{definitions_section}---

# Your Task
Determine which trends apply to this specific topic path based on the comment. The comment has already been classified as relevant to this topic. Now, identify what temporal changes or patterns (if any) the author is describing about this specific topic.

Use the topic definitions above to understand what each level of the path represents and ensure your trend analysis is aligned with the actual meaning of the topic.

---

# What Counts as a Trend
A trend for this topic path is a statement that indicates:
- A temporal comparison between past and present states **of this topic**
- An observation of change (or consistency) over time **regarding this topic**
- An evolution or trajectory **of some aspect of this topic**

**Trends require temporal markers** such as:
- Explicit time references: "last year", "in 2023", "this quarter", "recently"
- Comparative language: "used to be", "no longer", "has become", "still"
- Change verbs: "improved", "declined", "increased", "decreased", "remains"
- Sequential language: "now", "anymore", "these days", "currently"

---

# Direction Types

| Direction | Meaning | Example |
|-----------|---------|---------|
| **increasing** | Growing, rising, or expanding | "Workload has increased" |
| **decreasing** | Shrinking, falling, or contracting | "Support has decreased" |
| **improving** | Getting better in quality/effectiveness | "Teaching has improved" |
| **deteriorating** | Getting worse in quality/effectiveness | "Quality has worsened" |
| **stable_positive** | Something good remains consistent | "Still excellent as always" |
| **stable_negative** | Something bad remains consistent | "Still inadequate" |
| **fluctuating** | Varying or inconsistent over time | "Sometimes good, sometimes not" |

---

# Valence Types

| Valence | Meaning |
|---------|---------|
| **positive** | The trend is favorable, beneficial, or desirable |
| **negative** | The trend is unfavorable, problematic, or undesirable |
| **neutral** | The trend is neither clearly good nor bad |
| **mixed** | The trend has both positive and negative aspects |

**Important**: Valence should be determined from the **commenter's perspective** as expressed in the text.

---

# Confidence Scoring

For each trend, assign a **confidence score** from 1 to 5:

| Score | Meaning | When to Use |
|-------|---------|-------------|
| **5** | Very certain | Explicit temporal markers, clear change, unambiguous direction, clearly relevant to topic |
| **4** | Quite certain | Strong temporal indicators, clear trend with minor ambiguity about topic relevance |
| **3** | Moderately certain | Temporal markers present but trend or topic relevance requires interpretation |
| **2** | Somewhat uncertain | Weak temporal markers or ambiguous direction/valence or topic relevance |
| **1** | Very uncertain | Minimal temporal indicators, highly ambiguous interpretation or topic relevance |

**Confidence Guidelines**:
- Higher confidence for explicit time references ("since 2023", "compared to last year")
- Higher confidence when trend is clearly about this specific topic path
- Lower confidence for vague temporal markers ("recently", "these days")
- Lower confidence when it's unclear if trend applies to this specific topic vs. related topics
- Higher confidence when direction and valence are both clear

---

# Key Rules
- A comment may contain **multiple trends** for the same topic path
- Only identify trends that are **explicitly stated** in the comment with temporal markers
- Each trend must have a corresponding excerpt that supports it
- The excerpt must be directly relevant to the topic path
- Focus on trends **about this specific topic**, not other topics
- If no trends are present for this topic, return has_trends: false
- DO NOT infer trends from static descriptions without temporal comparison

---

# Output Format
Return a JSON object with:
- **has_trends**: Boolean (true if at least one trend applies, false otherwise)
- **trends**: List of objects, each containing:
    * **excerpt**: The exact text span from the comment containing the trend.
    * **reasoning**: One sentence explaining why this qualifies as a trend for this topic path.
    * **subject**: What aspect of the topic is changing (keep concise and aligned with the topic path).
    * **direction**: One of: "increasing", "decreasing", "improving", "deteriorating", "stable_positive", "stable_negative", "fluctuating"
    * **valence**: One of: "positive", "negative", "neutral", "mixed"
    * **confidence**: Integer from 1 to 5

---

# Examples

**Example 1:**
Comment: "The instructor's explanations used to be clear, but have become more confusing recently."
Topic Path: "Teaching Effectiveness>Teaching Style>Explanations"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The instructor's explanations used to be clear, but have become more confusing recently",
      "reasoning": "Indicates a decline in explanation clarity over time, directly relevant to the Explanations topic.",
      "subject": "explanation clarity",
      "direction": "deteriorating",
      "valence": "negative",
      "confidence": 5
    }}
  ]
}}

---

**Example 2:**
Comment: "The course content was excellent and very relevant to my work."
Topic Path: "Course Content>Relevance"
Output:
{{
  "has_trends": false,
  "trends": []
}}

---

**Example 3:**
Comment: "The amount of hands-on practice has increased significantly compared to last year, which is great. However, the quality of feedback on those exercises has declined."
Topic Path: "Teaching Effectiveness>Practical Application"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The amount of hands-on practice has increased significantly compared to last year",
      "reasoning": "Shows an increase in the quantity of practical application opportunities with explicit temporal reference.",
      "subject": "hands-on practice quantity",
      "direction": "increasing",
      "valence": "positive",
      "confidence": 5
    }},
    {{
      "excerpt": "the quality of feedback on those exercises has declined",
      "reasoning": "Indicates deterioration in feedback quality for practical exercises over time.",
      "subject": "feedback quality",
      "direction": "deteriorating",
      "valence": "negative",
      "confidence": 4
    }}
  ]
}}

---

**Example 4:**
Comment: "The instructor remains consistently engaging and motivating, just like in previous sessions I attended."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "The instructor remains consistently engaging and motivating, just like in previous sessions I attended",
      "reasoning": "Indicates stable positive quality in teaching style with explicit temporal comparison to past sessions.",
      "subject": "instructor engagement",
      "direction": "stable_positive",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

---

**Example 5:**
Comment: "Support from the teaching assistants has been really hit or miss lately - sometimes very helpful, sometimes unavailable."
Topic Path: "Course Support>Teaching Assistant Support"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Support from the teaching assistants has been really hit or miss lately - sometimes very helpful, sometimes unavailable",
      "reasoning": "Describes inconsistent quality of TA support over recent time period, directly relevant to the Teaching Assistant Support topic.",
      "subject": "TA support consistency",
      "direction": "fluctuating",
      "valence": "mixed",
      "confidence": 4
    }}
  ]
}}

---

**Example 6:**
Comment: "The workload has increased dramatically this semester, making it hard to keep up. Meanwhile, the teaching quality has remained excellent throughout."
Topic Path: "Teaching Effectiveness>Teaching Style"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "the teaching quality has remained excellent throughout",
      "reasoning": "Indicates consistent high quality in teaching style over the semester period, directly relevant to Teaching Style.",
      "subject": "teaching quality",
      "direction": "stable_positive",
      "valence": "positive",
      "confidence": 5
    }}
  ]
}}

Note: The workload trend is not included because it's not relevant to the "Teaching Effectiveness>Teaching Style" topic path.

---

**Example 7:**
Comment: "Things have gotten somewhat better recently, I think."
Topic Path: "Course Content>Organization"
Output:
{{
  "has_trends": true,
  "trends": [
    {{
      "excerpt": "Things have gotten somewhat better recently",
      "reasoning": "Suggests improvement over time, though the specifics and certainty are vague.",
      "subject": "course organization",
      "direction": "improving",
      "valence": "positive",
      "confidence": 2
    }}
  ]
}}
"""

"""
Command-line interface for hierarchical text classification.

This module provides a CLI for running classification tasks from the terminal.
"""

import sys
import ast
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.panel import Panel

from .processor import ClassificationProcessor
from .policies import (
    DefaultPolicy, 
    ConfidenceThresholdPolicy, 
    CompositePolicy,
    ExcerptRequiredPolicy
)
from .prompts import (
    standard_classification_prompt,
    keyword_focused_prompt,
    sentiment_aware_classification_prompt
)
from .hierarchy import build_tree_from_paths, format_tree_as_string


console = Console()


def parse_target_path(path_str: str) -> List[str]:
    """
    Parse a target path string into a list.
    
    Args:
        path_str: String representation of path (e.g., "['CEO','Vision']")
        
    Returns:
        List of path components
    """
    try:
        path_list = ast.literal_eval(path_str)
        if not isinstance(path_list, list):
            raise ValueError
        return path_list
    except Exception:
        raise ValueError(
            f"Invalid target-path format: {path_str}. "
            "Use a list like ['CEO','Vision']."
        )


def create_policy(
    confidence: int = None,
    require_excerpt: bool = False
) -> 'AcceptancePolicy':
    """
    Create an acceptance policy based on CLI options.
    
    Args:
        confidence: Minimum confidence threshold (1-5)
        require_excerpt: Whether to require non-empty excerpts
        
    Returns:
        Configured acceptance policy
    """
    policies = []
    
    if confidence is not None:
        policies.append(ConfidenceThresholdPolicy(min_confidence=confidence))
    
    if require_excerpt:
        policies.append(ExcerptRequiredPolicy())
    
    if policies:
        return CompositePolicy(*policies)
    else:
        return DefaultPolicy()


def select_prompt_function(prompt_type: str):
    """
    Select a prompt function based on type string.
    
    Args:
        prompt_type: One of "standard", "keyword", "sentiment"
        
    Returns:
        Prompt generation function
    """
    prompt_map = {
        "standard": standard_classification_prompt,
        "keyword": keyword_focused_prompt,
        "sentiment": sentiment_aware_classification_prompt
    }
    
    return prompt_map.get(prompt_type, standard_classification_prompt)


def display_hierarchical_results(
    results: dict,
    verbose: bool = False
) -> None:
    """
    Display hierarchical classification results in the console.
    
    Args:
        results: Dictionary mapping texts to classification paths
        verbose: Whether to show tree visualization
    """
    if not verbose:
        return
    
    for text, paths in results.items():
        if not paths:
            console.print(
                Panel(
                    "No classifications found",
                    title=f"Text: {text[:50]}...",
                    border_style="yellow"
                )
            )
            continue
        
        tree = build_tree_from_paths(paths)
        tree_str = format_tree_as_string(tree)
        panel_content = f"{tree_str}\n\n[bold]Paths:[/bold]\n" + "\n".join(
            f"  • {path}" for path in paths
        )
        
        console.print(
            Panel(
                panel_content,
                title=f"Text: {text[:50]}...",
                border_style="blue"
            )
        )


def display_target_path_results(
    results: dict,
    target_path: List[str]
) -> None:
    """
    Display target path classification results in the console.
    
    Args:
        results: Dictionary mapping texts to classification results
        target_path: The target path that was evaluated
    """
    path_str = " > ".join(target_path)
    
    for text, result in results.items():
        panel_lines = [
            f"[bold]Relevant:[/bold] {result.is_relevant}",
            f"[bold]Confidence:[/bold] {result.confidence}/5",
            f"[bold]Reasoning:[/bold] {result.reasoning}",
            f"[bold]Excerpt:[/bold] {result.excerpt or '[none]'}"
        ]
        
        border_style = "green" if result.is_relevant else "red"
        
        console.print(
            Panel(
                "\n".join(panel_lines),
                title=f"Text: {text[:50]}...\nPath: {path_str}",
                border_style=border_style
            )
        )


def display_full_path_results(
    results: dict,
    target_path: List[str]
) -> None:
    """
    Display full path classification results in the console.
    
    Args:
        results: Dictionary mapping texts to ClassificationOutput objects
        target_path: The target path that was evaluated
    """
    path_str = " > ".join(target_path)
    
    for text, output in results.items():
        panel_lines = []
        
        # Show per-node results
        panel_lines.append("[bold]Node Results:[/bold]")
        for node_name, result in output.node_results.items():
            status = "✓" if result.is_relevant else "✗"
            panel_lines.append(
                f"  {status} {node_name}: "
                f"Relevant={result.is_relevant}, "
                f"Confidence={result.confidence}/5"
            )
            panel_lines.append(f"    Reasoning: {result.reasoning}")
            if result.excerpt:
                panel_lines.append(f"    Excerpt: {result.excerpt}")
        
        # Show final classification paths
        if output.classification_paths:
            panel_lines.append(
                f"\n[bold]Classification Path(s):[/bold]\n  " +
                "\n  ".join(output.classification_paths)
            )
        else:
            panel_lines.append(
                "\n[bold]No valid classification path[/bold] "
                "(relevance stopped at some node)"
            )
        
        console.print(
            Panel(
                "\n".join(panel_lines),
                title=f"Text: {text[:50]}...\nEvaluated Path: {path_str}",
                border_style="blue"
            )
        )


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to topic hierarchy JSON file"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional JSON output file"
)
@click.option(
    "--target-path",
    type=str,
    default=None,
    help="Evaluate a specific target path, e.g. '[\"CEO\",\"Vision\"]'"
)
@click.option(
    "--full-path-mode",
    is_flag=True,
    help="Evaluate all nodes along the path (not just the leaf)"
)
@click.option(
    "--min-confidence",
    type=click.IntRange(1, 5),
    default=None,
    help="Minimum confidence threshold (1-5)"
)
@click.option(
    "--require-excerpt",
    is_flag=True,
    help="Require non-empty excerpts for acceptance"
)
@click.option(
    "--prompt-type",
    type=click.Choice(["standard", "keyword", "sentiment"]),
    default="standard",
    help="Type of prompt to use"
)
@click.option(
    "--gpu-list",
    type=str,
    default="0,1,2,3,4,5,6,7",
    help="Comma-separated list of GPU IDs"
)
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (-v, -vv, -vvv)"
)
def main(
    config: Path,
    output: Path,
    target_path: str,
    full_path_mode: bool,
    min_confidence: int,
    require_excerpt: bool,
    prompt_type: str,
    gpu_list: str,
    verbose: int
):
    """
    Hierarchical text classification using LLMs.
    
    Reads texts from stdin (one per line) and classifies them according to
    a hierarchical topic structure defined in the config file.
    
    Examples:
    
        # Basic hierarchical classification
        echo "The CEO discussed our vision" | python -m classifier.cli --config topics.json -v
        
        # Target a specific path
        echo "The CEO discussed our vision" | python -m classifier.cli \\
            --config topics.json --target-path '["Leadership","CEO"]' -v
        
        # Full path evaluation with confidence threshold
        echo "The CEO discussed our vision" | python -m classifier.cli \\
            --config topics.json --target-path '["Leadership","CEO"]' \\
            --full-path-mode --min-confidence 4 -v
    """
    # Read texts from stdin
    lines = [line.strip() for line in sys.stdin if line.strip()]
    
    if not lines:
        console.print("[yellow]No input text provided. Reading from stdin...[/yellow]")
        return
    
    # Parse options
    target_list = parse_target_path(target_path) if target_path else None
    gpu_ids = [int(x.strip()) for x in gpu_list.split(",")]
    policy = create_policy(confidence=min_confidence, require_excerpt=require_excerpt)
    prompt_fn = select_prompt_function(prompt_type)
    
    # Initialize processor
    with ClassificationProcessor(
        config_path=config,
        gpu_list=gpu_ids,
        prompt_fn=prompt_fn,
        policy=policy
    ) as processor:
        
        # Run classification based on mode
        if target_list and full_path_mode:
            # Full path evaluation
            console.print(
                f"[cyan]Evaluating full path: {' > '.join(target_list)}[/cyan]"
            )
            results = processor.classify_full_path(
                texts=lines,
                target_path=target_list
            )
            
            if verbose:
                display_full_path_results(results, target_list)
            
            if output:
                processor.export_results(results, output, mode="full_path")
                console.print(f"[green]Results saved to {output}[/green]")
        
        elif target_list:
            # Target path (leaf only)
            console.print(
                f"[cyan]Evaluating target path: {' > '.join(target_list)}[/cyan]"
            )
            results = processor.classify_target_path(
                texts=lines,
                target_path=target_list
            )
            
            if verbose:
                display_target_path_results(results, target_list)
            
            if output:
                processor.export_results(results, output, mode="target_path")
                console.print(f"[green]Results saved to {output}[/green]")
        
        else:
            # Hierarchical traversal
            console.print("[cyan]Running hierarchical classification[/cyan]")
            results = processor.classify_hierarchical(texts=lines)
            
            if verbose:
                display_hierarchical_results(results, verbose=True)
            
            if output:
                processor.export_results(results, output, mode="hierarchical")
                console.print(f"[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    main()

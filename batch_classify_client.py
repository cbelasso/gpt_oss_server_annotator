#!/usr/bin/env python3
"""
Batch Classification Client - Uses Classifier Server

Mirrors the interface of batch_classify.py but sends requests to a running
classifier server instead of loading models locally.

Usage examples:

# Standard classification
python batch_classify_client.py \\
    --input-file data.csv \\
    --input-column comment \\
    --config topics.json \\
    --server-url http://localhost:9000 \\
    --save-path results.json

# With multiple capabilities
python batch_classify_client.py \\
    --input-file data.csv \\
    --input-column comment \\
    --config topics.json \\
    --server-url http://localhost:9000 \\
    --enable-recommendations --enable-alerts \\
    --save-path results.json

# Chunked output with progress
python batch_classify_client.py \\
    --input-file data.csv \\
    --input-column comment \\
    --config topics.json \\
    --server-url http://localhost:9000 \\
    --output-dir results/ \\
    --chunk-size 100 \\
    --enable-stem-recommendations --enable-stem-polarity \\
    -v

# Without config (using server's default)
python batch_classify_client.py \\
    --input-file data.csv \\
    --server-url http://localhost:9000 \\
    --enable-recommendations-only \\
    --save-path results.json
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import click
import pandas as pd
import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def load_input_texts(
    input_file: Path = None, input_column: str = None, verbose: int = 0
) -> List[str]:
    """Load texts from file or stdin."""
    if input_file is None:
        if verbose > 0:
            console.print("[cyan]Reading texts from stdin...[/cyan]")
        return [line.strip() for line in sys.stdin if line.strip()]

    if verbose > 0:
        console.print(f"[cyan]Reading input file:[/cyan] {input_file}")

    suffix = input_file.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(input_file)
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(input_file)
        elif suffix == ".json":
            df = pd.read_json(input_file)
        elif suffix == ".pkl":
            df = pd.read_pickle(input_file)
        elif suffix == ".txt":
            with open(input_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as e:
        console.print(f"[red]Failed to read input file: {e}[/red]")
        sys.exit(1)

    # Infer column if not provided
    if input_column is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            console.print("[red]No text columns found. Specify --input-column.[/red]")
            sys.exit(1)
        input_column = text_cols[0]
        if verbose > 0:
            console.print(f"[yellow]Using column:[/yellow] {input_column}")

    if input_column not in df.columns:
        console.print(f"[red]Column '{input_column}' not found.[/red]")
        sys.exit(1)

    lines = df[input_column].dropna().astype(str).tolist()

    if verbose > 0:
        console.print(f"[green]✓ Loaded {len(lines)} texts[/green]")

    return lines


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def save_chunk_results(output_dir: Path, chunk_idx: int, chunk_data: List[Dict], verbose: int):
    """Save chunk to numbered JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"chunk_{chunk_idx:04d}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    if verbose > 0:
        console.print(f"[green]✓ Saved chunk {chunk_idx} to: {output_file}[/green]")


def save_results(save_path: Path, results: List[Dict], verbose: int):
    """Save results to single JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if verbose > 0:
        console.print(f"[green]✓ Results saved to: {save_path}[/green]")


def classify_batch(
    server_url: str,
    texts: List[str],
    capabilities: List[str],
    config_path: Optional[Path] = None,
    project_name: Optional[str] = None,
    timeout: int = 300,
) -> Dict:
    """
    Send classification request to server.

    Args:
        server_url: Server base URL (e.g., http://localhost:9000)
        texts: List of texts to classify
        capabilities: List of capability names
        config_path: Optional config file path
        project_name: Optional project name
        timeout: Request timeout in seconds

    Returns:
        Response JSON dict

    Raises:
        requests.RequestException: If request fails
    """
    endpoint = f"{server_url.rstrip('/')}/classify"

    payload = {
        "texts": texts,
        "capabilities": capabilities,
    }

    if config_path:
        payload["config_path"] = str(config_path)

    if project_name:
        payload["project_name"] = project_name

    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )

    response.raise_for_status()
    return response.json()


@click.command()
@click.option(
    "--server-url",
    type=str,
    required=True,
    help="Classification server URL (e.g., http://localhost:9000)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to topic hierarchy JSON file (optional if server has default)",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Input file (.txt, .csv, .json, .xlsx, .pkl). Reads from stdin if not provided.",
)
@click.option(
    "--input-column",
    type=str,
    default=None,
    help="Column name for text data (auto-detected if not provided)",
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path (single file mode)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for chunked files",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Texts per chunk (requires --output-dir)",
)
@click.option(
    "--project-name",
    type=str,
    default=None,
    help="Project name for root prefix (e.g., 'sce', 'eec')",
)
@click.option(
    "--request-timeout",
    type=int,
    default=300,
    help="Request timeout in seconds (default: 300)",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level (-v, -vv, -vvv)",
)
# Capability flags
@click.option(
    "--enable-classification",
    is_flag=True,
    default=True,
    help="Enable hierarchical classification (default: enabled)",
)
@click.option(
    "--enable-recommendations",
    is_flag=True,
    help="Enable recommendation detection",
)
@click.option(
    "--enable-alerts",
    is_flag=True,
    help="Enable alert detection",
)
@click.option(
    "--enable-stem-recommendations",
    is_flag=True,
    help="Enable stem recommendation analysis (requires classification)",
)
@click.option(
    "--enable-stem-polarity",
    is_flag=True,
    help="Enable stem polarity analysis (requires classification)",
)
@click.option(
    "--enable-stem-trends",
    is_flag=True,
    help="Enable stem trend analysis (requires classification)",
)
@click.option(
    "--enable-trends",
    is_flag=True,
    help="Enable global trend analysis",
)
@click.option(
    "--max-stem-definitions",
    type=int,
    default=None,
    help="Max definitions for stem analysis (NOTE: This is configured on the server side, not per-request)",
)
@click.option(
    "--recommendations-only",
    is_flag=True,
    help="Only detect recommendations (no classification)",
)
@click.option(
    "--alerts-only",
    is_flag=True,
    help="Only detect alerts (no classification)",
)
def main(
    server_url: str,
    config: Optional[Path],
    input_file: Optional[Path],
    input_column: Optional[str],
    save_path: Optional[Path],
    output_dir: Optional[Path],
    chunk_size: Optional[int],
    project_name: Optional[str],
    request_timeout: int,
    verbose: int,
    enable_classification: bool,
    enable_recommendations: bool,
    enable_alerts: bool,
    enable_stem_recommendations: bool,
    enable_stem_polarity: bool,
    enable_stem_trends: bool,
    enable_trends: bool,
    max_stem_definitions: Optional[int],
    recommendations_only: bool,
    alerts_only: bool,
):
    """
    Batch classification client - sends requests to classifier server.

    This script mirrors the interface of batch_classify.py but uses a running
    classifier server instead of loading models locally. This allows you to:
    - Use different configs per batch job without restarting the server
    - Process multiple datasets concurrently against the same server
    - Avoid GPU memory management on the client side
    """
    # Note about max_stem_definitions
    if max_stem_definitions is not None:
        console.print(
            "[yellow]Note: --max-stem-definitions is configured on the server side. "
            "This parameter has no effect in client mode.[/yellow]"
        )

    # Validate output options
    if output_dir and chunk_size is None:
        console.print("[red]Error: --chunk-size required with --output-dir[/red]")
        sys.exit(1)

    if chunk_size and output_dir is None:
        console.print("[red]Error: --output-dir required with --chunk-size[/red]")
        sys.exit(1)

    if save_path and output_dir:
        console.print("[red]Error: Cannot use both --save-path and --output-dir[/red]")
        sys.exit(1)

    # Handle standalone modes
    standalone_mode = recommendations_only or alerts_only

    if standalone_mode:
        if recommendations_only:
            enable_recommendations = True
        if alerts_only:
            enable_alerts = True
        enable_classification = False
        enable_stem_recommendations = False
        enable_stem_polarity = False
        enable_stem_trends = False

        if verbose > 0:
            mode_desc = "standalone"
            console.print(
                f"[yellow]Running {mode_desc} detection mode (no classification)[/yellow]"
            )

    # Load texts
    lines = load_input_texts(input_file, input_column, verbose)

    if not lines:
        console.print("[red]No input text provided[/red]")
        sys.exit(1)

    if save_path is None and output_dir is None and verbose == 0:
        console.print("[red]Error: Specify --save-path, --output-dir, or -v[/red]")
        sys.exit(1)

    # Setup chunking
    use_chunked_output = output_dir is not None and chunk_size is not None

    # For server requests, use smaller chunks to show progress
    # Even if not saving to chunked output
    request_chunk_size = chunk_size if chunk_size else 50
    chunks = chunk_list(lines, request_chunk_size)

    # Display configuration
    if verbose > 0:
        console.print("[cyan]Configuration:[/cyan]")
        console.print(f"  • Server: {server_url}")
        if config:
            console.print(f"  • Config: {config}")
        else:
            console.print("  • Config: Using server default")
        console.print(f"  • Texts: {len(lines)}")
        console.print(f"  • Request chunks: {len(chunks)}")

        if use_chunked_output:
            console.print(f"  • Output: Chunked ({chunk_size} per file)")
            console.print(f"  • Output dir: {output_dir}")
        elif save_path:
            console.print("  • Output: Single file")
            console.print(f"  • Output path: {save_path}")

        # Display enabled capabilities
        enabled_caps = []
        if enable_classification:
            enabled_caps.append("classification")
        if enable_recommendations:
            enabled_caps.append("recommendations")
        if enable_alerts:
            enabled_caps.append("alerts")
        if enable_stem_recommendations:
            enabled_caps.append("stem_recommendations")
        if enable_stem_polarity:
            enabled_caps.append("stem_polarity")
        if enable_stem_trends:
            enabled_caps.append("stem_trend")
        if enable_trends:
            enabled_caps.append("trend")

        console.print(f"  • Capabilities: {', '.join(enabled_caps)}")

    # Build capability list
    capability_names = []
    if enable_classification:
        capability_names.append("classification")
    if enable_recommendations:
        capability_names.append("recommendations")
    if enable_alerts:
        capability_names.append("alerts")
    if enable_stem_recommendations:
        capability_names.append("stem_recommendations")
    if enable_stem_polarity:
        capability_names.append("stem_polarity")
    if enable_stem_trends:
        capability_names.append("stem_trend")
    if enable_trends:
        capability_names.append("trend")

    # Check server health
    try:
        health_response = requests.get(f"{server_url.rstrip('/')}/health", timeout=10)
        health_response.raise_for_status()
        health_data = health_response.json()

        if verbose > 0:
            console.print("\n[green]✓ Server is healthy[/green]")
            if config:
                console.print(f"[cyan]Using config: {config}[/cyan]")
            elif health_data.get("default_config"):
                console.print(
                    f"[cyan]Using server's default config: {health_data['default_config']}[/cyan]"
                )
            else:
                console.print("[cyan]No config specified (standalone capabilities only)[/cyan]")
    except Exception as e:
        console.print(f"[red]Error: Cannot connect to server at {server_url}[/red]")
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    start_time = time.time()
    all_results = []

    # Process chunks with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Processing {len(chunks)} batches...", total=len(chunks)
        )

        for chunk_idx, chunk_texts in enumerate(chunks):
            try:
                # Send request to server
                response_data = classify_batch(
                    server_url=server_url,
                    texts=chunk_texts,
                    capabilities=capability_names,
                    config_path=config,
                    project_name=project_name,
                    timeout=request_timeout,
                )

                # Extract results
                chunk_results = response_data.get("results", [])
                all_results.extend(chunk_results)

                # Save chunk if using chunked output
                if use_chunked_output:
                    save_chunk_results(output_dir, chunk_idx, chunk_results, verbose=0)

                # Update progress
                progress.update(task, advance=1)

                # Show timing and config used (at verbose level 1 or higher)
                if verbose >= 1:
                    proc_time = response_data.get("processing_time", 0)
                    config_used = response_data.get("config_used", "none")
                    console.print(
                        f"[dim]  Chunk {chunk_idx}: {len(chunk_texts)} texts, "
                        f"{proc_time:.2f}s, config: {config_used}[/dim]"
                    )

            except requests.RequestException as e:
                console.print(f"\n[red]Error processing chunk {chunk_idx}: {e}[/red]")
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        console.print(f"[red]Server error: {error_detail}[/red]")
                    except:
                        console.print(f"[red]Server response: {e.response.text[:500]}[/red]")
                sys.exit(1)

    # Save results if single file mode
    if save_path:
        save_results(save_path, all_results, verbose)

    # Final summary
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format elapsed time
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        time_str = f"{hours}h {minutes}m {seconds:.2f}s"

    # Final summary
    if use_chunked_output:
        console.print(
            f"\n[green]✓ Processed {len(chunks)} batches ({len(lines)} texts)[/green]"
        )
        console.print(f"[cyan]Results saved to: {output_dir}/[/cyan]")
    elif save_path:
        console.print(f"\n[green]✓ Processed {len(lines)} texts[/green]")

    console.print(f"[cyan]⏱  Total Time: {time_str}[/cyan]")

    if verbose > 0:
        texts_per_second = len(lines) / elapsed_time if elapsed_time > 0 else 0
        console.print(f"[cyan]⚡ Throughput: {texts_per_second:.2f} texts/second[/cyan]")


if __name__ == "__main__":
    main()

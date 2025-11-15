#!/usr/bin/env python3
"""
Multi-Server Load Balancing Test Suite

Tests that requests are properly distributed across multiple VLLM servers.
"""

import json
from pathlib import Path
import time
from typing import Dict, List

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class MultiServerTester:
    """Test multi-server load balancing and performance."""

    def __init__(
        self,
        classifier_url: str = "http://localhost:9000",
        output_dir: Path = Path("test_results"),
    ):
        self.classifier_url = classifier_url
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def get_server_metrics(self) -> Dict:
        """Get current metrics from the classification server."""
        try:
            response = requests.get(f"{self.classifier_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            console.print(f"[red]Failed to get metrics: {e}[/red]")
        return {}

    def get_processor_metrics(self) -> Dict:
        """
        Get detailed processor metrics (requires custom endpoint).

        Note: You'll need to add a /processor-metrics endpoint to expose
        VLLMServerProcessor.get_metrics_summary()
        """
        try:
            response = requests.get(f"{self.classifier_url}/processor-metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            return {}

    def run_classification_test(
        self, texts: List[str], capabilities: List[str], test_name: str
    ) -> Dict:
        """Run a classification test and return results with timing."""
        console.print(f"\n[cyan]Running test: {test_name}[/cyan]")
        console.print(f"  Texts: {len(texts)}")
        console.print(f"  Capabilities: {', '.join(capabilities)}")

        # Get metrics before
        metrics_before = self.get_processor_metrics()

        # Make request
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.classifier_url}/classify",
                json={"texts": texts, "capabilities": capabilities},
                headers={"Content-Type": "application/json"},
                timeout=300,  # 5 minute timeout
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                # Get metrics after
                metrics_after = self.get_processor_metrics()

                # Save results
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = self.output_dir / f"{test_name}_{timestamp}.json"

                test_result = {
                    "test_name": test_name,
                    "timestamp": timestamp,
                    "num_texts": len(texts),
                    "capabilities": capabilities,
                    "elapsed_time": elapsed_time,
                    "processing_time": result.get("processing_time"),
                    "metrics_before": metrics_before,
                    "metrics_after": metrics_after,
                    "results": result.get("results", []),
                }

                with open(output_file, "w") as f:
                    json.dump(test_result, f, indent=2, ensure_ascii=False)

                console.print(f"[green]✓ Test completed in {elapsed_time:.2f}s[/green]")
                console.print(f"  Results saved to: {output_file}")

                return test_result
            else:
                console.print(f"[red]✗ Request failed: {response.status_code}[/red]")
                console.print(f"  {response.text}")
                return None

        except Exception as e:
            console.print(f"[red]✗ Test failed: {e}[/red]")
            return None

    def analyze_load_distribution(self, metrics_before: Dict, metrics_after: Dict):
        """Analyze how requests were distributed across servers."""
        if not metrics_before or not metrics_after:
            console.print("[yellow]No metrics available for load analysis[/yellow]")
            return

        console.print("\n[cyan]Load Distribution Analysis:[/cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Server", style="cyan")
        table.add_column("Requests Before", justify="right")
        table.add_column("Requests After", justify="right")
        table.add_column("New Requests", justify="right", style="green")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Latency", justify="right")

        total_new_requests = 0

        for server_url in metrics_after.keys():
            before = metrics_before.get(server_url, {})
            after = metrics_after.get(server_url, {})

            req_before = before.get("total_requests", 0)
            req_after = after.get("total_requests", 0)
            new_requests = req_after - req_before
            total_new_requests += new_requests

            success_rate = after.get("success_rate", "N/A")
            avg_latency = after.get("avg_latency", "N/A")

            table.add_row(
                server_url.split("//")[1].split("/")[0],  # Extract host:port
                str(req_before),
                str(req_after),
                str(new_requests),
                success_rate,
                avg_latency,
            )

        console.print(table)

        # Check if load is balanced
        if total_new_requests > 0:
            console.print(f"\n[cyan]Total new requests: {total_new_requests}[/cyan]")

            # Calculate if distribution is roughly equal
            num_servers = len(metrics_after)
            expected_per_server = total_new_requests / num_servers

            balanced = True
            for server_url in metrics_after.keys():
                before = metrics_before.get(server_url, {})
                after = metrics_after.get(server_url, {})
                new_requests = after.get("total_requests", 0) - before.get("total_requests", 0)

                # Allow 20% deviation from perfect balance
                if abs(new_requests - expected_per_server) > expected_per_server * 0.2:
                    balanced = False
                    break

            if balanced:
                console.print("[green]✓ Load is well balanced across servers[/green]")
            else:
                console.print("[yellow]⚠ Load distribution is uneven[/yellow]")


def main():
    """Run test suite."""
    console.print(
        Panel.fit(
            "[bold cyan]Multi-Server Load Balancing Test Suite[/bold cyan]", border_style="cyan"
        )
    )

    tester = MultiServerTester()

    # Test data
    single_text = [
        "The instructor's teaching style has improved significantly over the past year. "
        "They now provide more examples and take time to answer questions thoroughly."
    ]

    multiple_texts = [
        "The course content was excellent but the pacing was too fast.",
        "I loved the hands-on exercises and practical examples provided.",
        "The instructor needs to improve their explanation of complex topics.",
        "Great use of real-world case studies throughout the course.",
        "The online platform was difficult to navigate and had technical issues.",
        "More time should be allocated for Q&A sessions.",
        "The teaching assistant was very helpful and responsive.",
        "I wish there were more resources provided for self-study.",
        "The grading criteria were unclear and inconsistent.",
        "Overall, this was one of the best courses I've taken.",
    ]

    # Test 1: Single text
    console.print("\n" + "=" * 60)
    console.print("[bold]Test 1: Single Text Classification[/bold]")
    console.print("=" * 60)

    result1 = tester.run_classification_test(
        texts=single_text,
        capabilities=["classification", "recommendations"],
        test_name="single_text",
    )

    if result1:
        tester.analyze_load_distribution(
            result1.get("metrics_before", {}), result1.get("metrics_after", {})
        )

    # Wait a bit between tests
    console.print("\n[cyan]Waiting 2 seconds before next test...[/cyan]")
    time.sleep(2)

    # Test 2: Multiple texts
    console.print("\n" + "=" * 60)
    console.print("[bold]Test 2: Multiple Texts (10) Classification[/bold]")
    console.print("=" * 60)

    result2 = tester.run_classification_test(
        texts=multiple_texts,
        capabilities=["classification", "recommendations", "stem_polarity"],
        test_name="multiple_texts",
    )

    if result2:
        tester.analyze_load_distribution(
            result2.get("metrics_before", {}), result2.get("metrics_after", {})
        )

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Test Suite Complete![/bold green]")
    console.print("=" * 60)

    if result1 and result2:
        console.print("\n[cyan]Performance Summary:[/cyan]")
        console.print(f"  Single text: {result1['elapsed_time']:.2f}s")
        console.print(f"  10 texts: {result2['elapsed_time']:.2f}s")
        console.print(f"  Avg per text (10 texts): {result2['elapsed_time'] / 10:.2f}s")

        speedup = (result1["elapsed_time"] * 10) / result2["elapsed_time"]
        console.print(f"  Speedup factor: {speedup:.2f}x")

        if speedup > 1.5:
            console.print("[green]✓ Good parallelization![/green]")
        else:
            console.print("[yellow]⚠ Limited parallelization benefit[/yellow]")

    console.print(f"\n[cyan]Results saved to: {tester.output_dir}/[/cyan]")


if __name__ == "__main__":
    main()

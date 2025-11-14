"""
Stress test and benchmarking script for Classification Server

Demonstrates the efficiency of intelligent batching under various load patterns.
"""

import asyncio
import time
from typing import List
import statistics

import aiohttp
from rich.console import Console
from rich.table import Table

console = Console()


class LoadTester:
    """Load testing utility for classification server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def single_request(
        self, texts: List[str], capabilities: List[str], session: aiohttp.ClientSession
    ):
        """Make a single classification request."""
        payload = {"texts": texts, "capabilities": capabilities}

        start = time.time()
        async with session.post(
            f"{self.base_url}/classify",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            result = await response.json()
            latency = time.time() - start
            return result, latency

    async def concurrent_requests(
        self, num_requests: int, texts_per_request: int, capabilities: List[str]
    ):
        """Execute multiple concurrent requests."""
        console.print(
            f"[cyan]Running {num_requests} concurrent requests "
            f"({texts_per_request} texts each)[/cyan]"
        )

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                texts = [f"Test text {i}-{j}" for j in range(texts_per_request)]
                tasks.append(self.single_request(texts, capabilities, session))

            start = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start

            latencies = [latency for _, latency in results]
            total_texts = num_requests * texts_per_request

            return {
                "total_time": total_time,
                "total_texts": total_texts,
                "throughput": total_texts / total_time,
                "avg_latency": statistics.mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p50_latency": statistics.median(latencies),
                "p95_latency": statistics.quantiles(latencies, n=20)[18],
            }

    async def benchmark_batching_efficiency(self):
        """
        Benchmark showing batching efficiency with different request patterns.
        """
        console.print("\n[bold cyan]Batching Efficiency Benchmark[/bold cyan]\n")

        scenarios = [
            {
                "name": "Single Requests",
                "num_requests": 10,
                "texts_per_request": 1,
                "capabilities": ["classification"],
            },
            {
                "name": "Batch Requests",
                "num_requests": 2,
                "texts_per_request": 5,
                "capabilities": ["classification"],
            },
            {
                "name": "Mixed Capabilities (Sequential)",
                "num_requests": 5,
                "texts_per_request": 2,
                "capabilities": ["classification", "recommendations"],
            },
            {
                "name": "Mixed Capabilities (Concurrent)",
                "num_requests": 10,
                "texts_per_request": 1,
                "capabilities": ["classification", "recommendations"],
            },
        ]

        results = []
        for scenario in scenarios:
            console.print(f"Testing: {scenario['name']}")
            result = await self.concurrent_requests(
                num_requests=scenario["num_requests"],
                texts_per_request=scenario["texts_per_request"],
                capabilities=scenario["capabilities"],
            )
            result["scenario"] = scenario["name"]
            results.append(result)
            console.print(f"✓ Complete\n")
            await asyncio.sleep(0.5)  # Small delay between tests

        # Display results
        table = Table(title="Benchmark Results")
        table.add_column("Scenario", style="cyan")
        table.add_column("Total Texts", justify="right")
        table.add_column("Total Time", justify="right")
        table.add_column("Throughput\n(texts/sec)", justify="right", style="green")
        table.add_column("Avg Latency", justify="right")
        table.add_column("P95 Latency", justify="right")

        for result in results:
            table.add_row(
                result["scenario"],
                str(result["total_texts"]),
                f"{result['total_time']:.2f}s",
                f"{result['throughput']:.1f}",
                f"{result['avg_latency']:.3f}s",
                f"{result['p95_latency']:.3f}s",
            )

        console.print("\n")
        console.print(table)

    async def stress_test(self, duration_seconds: int = 10):
        """
        Run continuous load for specified duration.
        """
        console.print(f"\n[bold cyan]Stress Test ({duration_seconds}s)[/bold cyan]\n")

        start = time.time()
        request_count = 0
        text_count = 0
        latencies = []

        async with aiohttp.ClientSession() as session:
            while time.time() - start < duration_seconds:
                # Make request
                texts = [f"Stress test text {request_count}"]
                result, latency = await self.single_request(
                    texts, ["classification"], session
                )

                request_count += 1
                text_count += len(texts)
                latencies.append(latency)

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.01)

        total_time = time.time() - start

        console.print(f"[green]✓ Stress test complete[/green]\n")
        console.print(f"Total requests: {request_count}")
        console.print(f"Total texts: {text_count}")
        console.print(f"Duration: {total_time:.1f}s")
        console.print(f"Throughput: {text_count / total_time:.1f} texts/sec")
        console.print(f"Average latency: {statistics.mean(latencies):.3f}s")
        console.print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.3f}s")

    async def capability_comparison(self):
        """
        Compare performance across different capability combinations.
        """
        console.print("\n[bold cyan]Capability Performance Comparison[/bold cyan]\n")

        capability_sets = [
            ["classification"],
            ["classification", "recommendations"],
            ["classification", "alerts"],
            ["classification", "recommendations", "alerts"],
        ]

        results = []
        for capabilities in capability_sets:
            console.print(f"Testing: {', '.join(capabilities)}")
            result = await self.concurrent_requests(
                num_requests=5, texts_per_request=2, capabilities=capabilities
            )
            result["capabilities"] = ", ".join(capabilities)
            results.append(result)
            console.print(f"✓ Complete\n")
            await asyncio.sleep(0.5)

        # Display results
        table = Table(title="Capability Performance")
        table.add_column("Capabilities", style="cyan")
        table.add_column("Throughput", justify="right", style="green")
        table.add_column("Avg Latency", justify="right")

        for result in results:
            table.add_row(
                result["capabilities"],
                f"{result['throughput']:.1f} texts/sec",
                f"{result['avg_latency']:.3f}s",
            )

        console.print("\n")
        console.print(table)


async def main():
    """Run all benchmarks."""
    tester = LoadTester()

    # Check server health
    console.print("[cyan]Checking server health...[/cyan]")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{tester.base_url}/health") as response:
                health = await response.json()
                console.print(f"[green]✓ Server is {health['status']}[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Server not responding: {e}[/red]")
        console.print("\nStart the server first:")
        console.print("  python classifier_server.py --config topics.json")
        return

    # Run benchmarks
    await tester.benchmark_batching_efficiency()
    await tester.capability_comparison()
    await tester.stress_test(duration_seconds=10)

    # Show final server stats
    console.print("\n[bold cyan]Final Server Statistics[/bold cyan]\n")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{tester.base_url}/stats") as response:
            stats = await response.json()
            console.print(f"Total requests processed: {stats['total_requests']}")
            console.print(f"Total texts processed: {stats['total_texts_processed']}")
            console.print(f"Average processing time: {stats['average_processing_time']:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())

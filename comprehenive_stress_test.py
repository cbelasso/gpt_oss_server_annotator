"""
Comprehensive Concurrent Stress Test

Tests server with 20 realistic requests, different capability combinations,
and saves all results to JSON files for inspection.

Includes realistic scenarios:
- Regular feedback
- Recommendations
- Alert-worthy concerns
- Mixed capabilities

Results saved to: test_results/
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


# ============================================================================
# CONFIGURATION
# ============================================================================

SERVER_URL = "http://localhost:9000"  # <-- CHANGE THIS to match your server port
OUTPUT_DIR = Path("test_results")


# ============================================================================
# TEST SCENARIOS - 20 Realistic Cases
# ============================================================================

TEST_SCENARIOS = [
    # ===== SCENARIO 1-5: Regular Feedback (Classification Only) =====
    {
        "id": "regular_01",
        "text": "The training session was well-organized and covered all the essential topics. The instructor was knowledgeable and answered questions thoroughly.",
        "capabilities": ["classification"],
        "expected": "positive teaching feedback",
    },
    {
        "id": "regular_02",
        "text": "I found the course materials to be comprehensive. The slides were clear and the handouts were useful reference materials.",
        "capabilities": ["classification"],
        "expected": "positive course content feedback",
    },
    {
        "id": "regular_03",
        "text": "The online platform worked smoothly throughout the training. No technical issues encountered.",
        "capabilities": ["classification"],
        "expected": "positive technology feedback",
    },
    {
        "id": "regular_04",
        "text": "The pace of the training felt rushed in the morning session but improved in the afternoon.",
        "capabilities": ["classification"],
        "expected": "mixed pacing feedback",
    },
    {
        "id": "regular_05",
        "text": "Good mix of theory and practice. The group discussions were particularly valuable for understanding different perspectives.",
        "capabilities": ["classification"],
        "expected": "positive teaching methods",
    },
    # ===== SCENARIO 6-10: Constructive Feedback (Classification + Recommendations) =====
    {
        "id": "recommendations_01",
        "text": "The training would benefit from more hands-on exercises. We spent too much time on slides and not enough time practicing the concepts ourselves.",
        "capabilities": ["classification", "recommendations"],
        "expected": "recommendation for more practice",
    },
    {
        "id": "recommendations_02",
        "text": "I suggest providing pre-reading materials before the session so we can dive deeper during class time. Also, it would help to have follow-up resources for continued learning.",
        "capabilities": ["classification", "recommendations"],
        "expected": "multiple recommendations",
    },
    {
        "id": "recommendations_03",
        "text": "The examples used were outdated. Please update them to reflect current industry practices and technologies we're actually using in our daily work.",
        "capabilities": ["classification", "recommendations"],
        "expected": "recommendation to update content",
    },
    {
        "id": "recommendations_04",
        "text": "Consider breaking this into two separate sessions - one for beginners and one for advanced users. The current format tries to cover too much ground for mixed skill levels.",
        "capabilities": ["classification", "recommendations"],
        "expected": "recommendation for restructuring",
    },
    {
        "id": "recommendations_05",
        "text": "The instructor should allow more time for questions. Several people had their hands up but we had to move on due to time constraints. Maybe reduce the content or extend the session.",
        "capabilities": ["classification", "recommendations"],
        "expected": "recommendation for more Q&A time",
    },
    # ===== SCENARIO 11-14: Alert-Worthy Concerns (Classification + Alerts) =====
    {
        "id": "alert_01",
        "text": "During the training, the facilitator made several comments about my accent and asked where I'm 'really from' in front of the whole group. It made me very uncomfortable and I felt singled out.",
        "capabilities": ["classification", "alerts"],
        "expected": "discrimination alert",
    },
    {
        "id": "alert_02",
        "text": "I reported safety concerns about the equipment to my manager last month, but nothing has been done. Now I'm being assigned to less desirable shifts and I feel like it's retaliation for speaking up.",
        "capabilities": ["classification", "alerts"],
        "expected": "retaliation + safety concern alerts",
    },
    {
        "id": "alert_03",
        "text": "The team leader repeatedly makes inappropriate jokes of a sexual nature during our training sessions. When I asked him to stop, he laughed it off and said I need to 'lighten up'. Other colleagues have mentioned this makes them uncomfortable too.",
        "capabilities": ["classification", "alerts"],
        "expected": "sexual harassment alert",
    },
    {
        "id": "alert_04",
        "text": "My supervisor constantly belittles my work in front of others during team meetings, calling my contributions 'stupid' and 'worthless'. This happens almost every week and has severely impacted my confidence and mental health.",
        "capabilities": ["classification", "alerts"],
        "expected": "bullying + hostile environment alert",
    },
    # ===== SCENARIO 15-17: Everything Combined (Classification + Recommendations + Alerts) =====
    {
        "id": "combined_01",
        "text": "The training content needs to be updated with more current examples. However, I'm more concerned about the fact that our instructor made derogatory comments about women in tech throughout the session, suggesting we're not suited for certain roles. This needs to be addressed immediately.",
        "capabilities": ["classification", "recommendations", "alerts"],
        "expected": "recommendation + discrimination alert",
    },
    {
        "id": "combined_02",
        "text": "I think we should include more interactive activities in future sessions - the current format is too lecture-heavy. On a separate note, I witnessed a coworker using racial slurs during the break, and when I reported it to HR, I was told I'm being 'too sensitive'.",
        "capabilities": ["classification", "recommendations", "alerts"],
        "expected": "recommendation + discrimination + retaliation concerns",
    },
    {
        "id": "combined_03",
        "text": "The training should cover more advanced topics for those of us with experience. Also, the training room has exposed electrical wiring that looks dangerous - someone should fix that before someone gets hurt.",
        "capabilities": ["classification", "recommendations", "alerts"],
        "expected": "recommendation + safety alert",
    },
    # ===== SCENARIO 18-20: Edge Cases and Mixed Scenarios =====
    {
        "id": "edge_01",
        "text": "Keep the current format exactly as it is - don't change anything. The instructor's teaching style is perfect and the materials are excellent.",
        "capabilities": ["classification", "recommendations"],
        "expected": "positive feedback, possibly 'continue' recommendation",
    },
    {
        "id": "edge_02",
        "text": "The online platform keeps crashing and we're losing work. IT hasn't fixed it despite multiple tickets. I'm worried this will cause us to miss important deadlines.",
        "capabilities": ["classification", "recommendations", "alerts"],
        "expected": "technology issue, possible concern about neglected issues",
    },
    {
        "id": "edge_03",
        "text": "Overall solid training. The instructor knows the material well. If I had to suggest one improvement, it would be to provide the slides beforehand so we can take better notes during the session.",
        "capabilities": ["classification", "recommendations"],
        "expected": "positive + minor recommendation",
    },
]


# ============================================================================
# Test Client Class
# ============================================================================


class TestClient:
    """Client for running concurrent classification tests."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []

    async def classify(
        self, scenario_id: str, text: str, capabilities: List[str], expected: str = None
    ) -> Dict[str, Any]:
        """
        Run a single classification request.

        Returns dict with scenario info and results.
        """
        payload = {"texts": [text], "capabilities": capabilities}

        start_time = asyncio.get_event_loop().time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/classify",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        elapsed = asyncio.get_event_loop().time() - start_time

                        return {
                            "scenario_id": scenario_id,
                            "text": text,
                            "capabilities": capabilities,
                            "expected": expected,
                            "status": "success",
                            "elapsed_time": elapsed,
                            "result": result,
                        }
                    else:
                        error = await response.text()
                        return {
                            "scenario_id": scenario_id,
                            "text": text,
                            "capabilities": capabilities,
                            "expected": expected,
                            "status": "error",
                            "error": f"HTTP {response.status}: {error}",
                        }
        except Exception as e:
            return {
                "scenario_id": scenario_id,
                "text": text,
                "capabilities": capabilities,
                "expected": expected,
                "status": "error",
                "error": str(e),
            }

    async def run_concurrent_tests(
        self, scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run all test scenarios concurrently.
        """
        console.print(f"\n[cyan]Running {len(scenarios)} concurrent requests...[/cyan]")
        console.print(f"[cyan]Server: {self.base_url}[/cyan]\n")

        # Create tasks for all scenarios
        tasks = []
        for scenario in scenarios:
            task = self.classify(
                scenario_id=scenario["id"],
                text=scenario["text"],
                capabilities=scenario["capabilities"],
                expected=scenario.get("expected"),
            )
            tasks.append(task)

        # Run concurrently with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing {len(scenarios)} requests...", total=len(scenarios)
            )

            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                progress.advance(task)

        self.results = results
        return results


# ============================================================================
# Result Analysis and Saving
# ============================================================================


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """Save results in multiple formats for easy inspection."""

    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save full results as single JSON
    full_results_path = output_dir / f"full_results_{timestamp}.json"
    with open(full_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"[green]✓ Saved full results to: {full_results_path}[/green]")

    # 2. Save individual scenario results
    scenarios_dir = output_dir / f"scenarios_{timestamp}"
    scenarios_dir.mkdir(exist_ok=True)

    for result in results:
        scenario_file = scenarios_dir / f"{result['scenario_id']}.json"
        with open(scenario_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Saved individual scenarios to: {scenarios_dir}[/green]")

    # 3. Save summary report
    summary = generate_summary(results)
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Saved summary to: {summary_path}[/green]")

    # 4. Save alerts-only results
    alerts_results = [r for r in results if "alerts" in r.get("capabilities", [])]
    if alerts_results:
        alerts_path = output_dir / f"alerts_only_{timestamp}.json"
        with open(alerts_path, "w", encoding="utf-8") as f:
            json.dump(alerts_results, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✓ Saved alert scenarios to: {alerts_path}[/green]")

    return full_results_path, scenarios_dir, summary_path


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from results."""

    total_requests = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")

    # Timing stats
    elapsed_times = [r["elapsed_time"] for r in results if "elapsed_time" in r]
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    min_time = min(elapsed_times) if elapsed_times else 0
    max_time = max(elapsed_times) if elapsed_times else 0

    # Capability distribution
    capability_counts = {}
    for result in results:
        caps_key = "+".join(sorted(result.get("capabilities", [])))
        capability_counts[caps_key] = capability_counts.get(caps_key, 0) + 1

    # Alert detection
    alerts_detected = []
    for result in results:
        if result["status"] == "success":
            result_data = result.get("result", {})
            if result_data.get("results"):
                first_result = result_data["results"][0]
                alerts = first_result.get("alerts", [])
                if alerts:
                    alerts_detected.append(
                        {
                            "scenario_id": result["scenario_id"],
                            "text_preview": result["text"][:80] + "...",
                            "alert_count": len(alerts),
                            "alert_types": [a.get("alert_type") for a in alerts],
                        }
                    )

    # Recommendation detection
    recommendations_detected = []
    for result in results:
        if result["status"] == "success":
            result_data = result.get("result", {})
            if result_data.get("results"):
                first_result = result_data["results"][0]
                recs = first_result.get("recommendations", [])
                if recs:
                    recommendations_detected.append(
                        {
                            "scenario_id": result["scenario_id"],
                            "text_preview": result["text"][:80] + "...",
                            "recommendation_count": len(recs),
                        }
                    )

    return {
        "test_timestamp": datetime.now().isoformat(),
        "total_requests": total_requests,
        "successful": successful,
        "failed": failed,
        "timing": {
            "average_seconds": round(avg_time, 2),
            "min_seconds": round(min_time, 2),
            "max_seconds": round(max_time, 2),
        },
        "capability_distribution": capability_counts,
        "alerts_detected": {"count": len(alerts_detected), "scenarios": alerts_detected},
        "recommendations_detected": {
            "count": len(recommendations_detected),
            "scenarios": recommendations_detected,
        },
    }


def print_summary(summary: Dict[str, Any]):
    """Print summary to console."""

    console.print("\n" + "=" * 80)
    console.print("[bold cyan]TEST SUMMARY[/bold cyan]")
    console.print("=" * 80)

    console.print("\n[bold]Overall Results:[/bold]")
    console.print(f"  Total requests: {summary['total_requests']}")
    console.print(f"  Successful: [green]{summary['successful']}[/green]")
    console.print(f"  Failed: [red]{summary['failed']}[/red]")

    console.print("\n[bold]Timing:[/bold]")
    console.print(f"  Average: {summary['timing']['average_seconds']}s")
    console.print(f"  Min: {summary['timing']['min_seconds']}s")
    console.print(f"  Max: {summary['timing']['max_seconds']}s")

    console.print("\n[bold]Capability Distribution:[/bold]")
    for caps, count in summary["capability_distribution"].items():
        console.print(f"  {caps}: {count} requests")

    console.print("\n[bold]Alerts Detected:[/bold]")
    console.print(
        f"  Total scenarios with alerts: [yellow]{summary['alerts_detected']['count']}[/yellow]"
    )
    for alert_info in summary["alerts_detected"]["scenarios"]:
        console.print(f"    • {alert_info['scenario_id']}: {alert_info['alert_count']} alerts")
        console.print(f"      Types: {', '.join(alert_info['alert_types'])}")

    console.print("\n[bold]Recommendations Detected:[/bold]")
    console.print(
        f"  Total scenarios with recommendations: [blue]{summary['recommendations_detected']['count']}[/blue]"
    )
    for rec_info in summary["recommendations_detected"]["scenarios"][:5]:  # Show first 5
        console.print(
            f"    • {rec_info['scenario_id']}: {rec_info['recommendation_count']} recommendations"
        )

    console.print("\n" + "=" * 80 + "\n")


# ============================================================================
# Main Test Runner
# ============================================================================


async def main():
    """Run comprehensive concurrent stress test."""

    console.print("[bold cyan]Comprehensive Concurrent Stress Test[/bold cyan]")
    console.print(f"Server: {SERVER_URL}")
    console.print(f"Test scenarios: {len(TEST_SCENARIOS)}")
    console.print(f"Output directory: {OUTPUT_DIR}\n")

    # Check server health
    console.print("[cyan]Checking server health...[/cyan]")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SERVER_URL}/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    health = await response.json()
                    console.print("[green]✓ Server is healthy[/green]")
                    console.print(f"  Capabilities: {health.get('capabilities_available', [])}")
                else:
                    console.print(f"[red]✗ Server returned status {response.status}[/red]")
                    return
    except Exception as e:
        console.print(f"[red]✗ Cannot connect to server: {e}[/red]")
        console.print("\n[yellow]Make sure server is running:[/yellow]")
        console.print(
            "  python classifier_server.py --config topics.json --gpu-list 0,1,2,3 --port 9000"
        )
        return

    # Run tests
    client = TestClient(SERVER_URL)
    start_time = asyncio.get_event_loop().time()

    results = await client.run_concurrent_tests(TEST_SCENARIOS)

    total_time = asyncio.get_event_loop().time() - start_time

    # Save results
    console.print("\n[cyan]Saving results...[/cyan]")
    full_path, scenarios_path, summary_path = save_results(results, OUTPUT_DIR)

    # Generate and print summary
    summary = generate_summary(results)
    summary["total_elapsed_time"] = round(total_time, 2)
    print_summary(summary)

    console.print(f"[bold green]✓ Test complete in {total_time:.2f}s[/bold green]")
    console.print("\n[cyan]Results saved to:[/cyan]")
    console.print(f"  • Full results: {full_path}")
    console.print(f"  • Individual scenarios: {scenarios_path}")
    console.print(f"  • Summary: {summary_path}")
    console.print("\n[cyan]Inspect the JSON files to verify outputs![/cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())

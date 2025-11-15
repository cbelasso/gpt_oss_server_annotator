#!/usr/bin/env python3
"""
Real-Time Multi-Server Monitoring Dashboard

Displays live metrics for all VLLM servers and the classification server.
"""

from datetime import datetime
import time
from typing import Dict, List

import requests
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class ServerMonitor:
    """Monitor multiple VLLM servers and classification server."""

    def __init__(
        self, classifier_url: str = "http://localhost:9000", vllm_urls: List[str] = None
    ):
        self.classifier_url = classifier_url
        self.vllm_urls = vllm_urls or [
            "http://localhost:8054",
            "http://localhost:8055",
            "http://localhost:8056",
        ]

    def get_classifier_stats(self) -> Dict:
        """Get classification server stats."""
        try:
            response = requests.get(f"{self.classifier_url}/stats", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}

    def get_processor_metrics(self) -> Dict:
        """Get processor metrics (requires custom endpoint)."""
        try:
            response = requests.get(f"{self.classifier_url}/processor-metrics", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}

    def check_vllm_health(self, url: str) -> bool:
        """Check if VLLM server is healthy."""
        try:
            response = requests.get(f"{url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def create_header(self) -> Panel:
        """Create header panel."""
        title = Text("🚀 Multi-Server Monitoring Dashboard", style="bold cyan")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return Panel(f"{title}\n{timestamp}", border_style="cyan")

    def create_classifier_panel(self, stats: Dict) -> Panel:
        """Create classification server status panel."""
        if not stats:
            return Panel(
                "[red]Classification Server: Offline[/red]",
                title="Classification Server",
                border_style="red",
            )

        uptime = stats.get("uptime_seconds", 0)
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

        content = f"""[green]Status: Online[/green]
Total Requests: {stats.get("total_requests", 0)}
Total Texts Processed: {stats.get("total_texts_processed", 0)}
Avg Processing Time: {stats.get("average_processing_time", 0):.2f}s
Uptime: {uptime_str}
"""

        # Add per-capability stats if available
        cap_stats = stats.get("requests_by_capability", {})
        if cap_stats:
            content += "\nRequests by Capability:"
            for cap, count in sorted(cap_stats.items(), key=lambda x: x[1], reverse=True):
                content += f"\n  • {cap}: {count}"

        return Panel(content, title="Classification Server", border_style="green")

    def create_vllm_servers_table(self, processor_metrics: Dict) -> Table:
        """Create VLLM servers status table."""
        table = Table(title="VLLM Servers", show_header=True, header_style="bold magenta")

        table.add_column("Server", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Requests", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Last Health Check", justify="center")

        if not processor_metrics:
            # Fallback to basic health checks
            for url in self.vllm_urls:
                is_healthy = self.check_vllm_health(url)
                status = "🟢 UP" if is_healthy else "🔴 DOWN"
                host_port = url.split("//")[1]

                table.add_row(host_port, status, "N/A", "N/A", "N/A", "N/A", "N/A")
        else:
            # Use detailed metrics
            for server_url, metrics in processor_metrics.items():
                host_port = server_url.split("//")[1].split("/")[0]

                is_healthy = metrics.get("is_healthy", False)
                status = "🟢 UP" if is_healthy else "🔴 DOWN"

                total_req = metrics.get("total_requests", 0)
                failed_req = metrics.get("failed_requests", 0)
                success_rate = metrics.get("success_rate", "N/A")
                avg_latency = metrics.get("avg_latency", "N/A")
                last_check = metrics.get("last_health_check", "Never")

                if last_check != "Never":
                    # Format timestamp nicely
                    try:
                        dt = datetime.fromisoformat(last_check)
                        last_check = dt.strftime("%H:%M:%S")
                    except:
                        pass

                table.add_row(
                    host_port,
                    status,
                    str(total_req),
                    str(failed_req),
                    success_rate,
                    avg_latency,
                    last_check,
                )

        return table

    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(Layout(name="header", size=3), Layout(name="body"))

        layout["body"].split_row(
            Layout(name="classifier", ratio=1), Layout(name="vllm_servers", ratio=2)
        )

        return layout

    def update_layout(self, layout: Layout):
        """Update layout with current data."""
        # Get data
        classifier_stats = self.get_classifier_stats()
        processor_metrics = self.get_processor_metrics()

        # Update panels
        layout["header"].update(self.create_header())
        layout["classifier"].update(self.create_classifier_panel(classifier_stats))
        layout["vllm_servers"].update(self.create_vllm_servers_table(processor_metrics))

    def run(self, refresh_interval: float = 2.0):
        """Run the monitoring dashboard."""
        layout = self.create_layout()

        try:
            with Live(layout, refresh_per_second=1, screen=True) as live:
                while True:
                    self.update_layout(layout)
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")


def main():
    """Run monitoring dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Server Monitoring Dashboard")
    parser.add_argument(
        "--classifier-url", default="http://localhost:9000", help="Classification server URL"
    )
    parser.add_argument(
        "--vllm-urls",
        default="http://localhost:8054,http://localhost:8055,http://localhost:8056",
        help="Comma-separated VLLM server URLs",
    )
    parser.add_argument(
        "--refresh", type=float, default=2.0, help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    vllm_urls = [url.strip() for url in args.vllm_urls.split(",")]

    monitor = ServerMonitor(classifier_url=args.classifier_url, vllm_urls=vllm_urls)

    console.print("[cyan]Starting monitoring dashboard...[/cyan]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")

    monitor.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()

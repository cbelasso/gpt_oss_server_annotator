#!/usr/bin/env python3
"""
VLLM Multi-Server Manager

Manages multiple VLLM server instances with proper GPU allocation.
Each server gets 2 GPUs (tensor_parallel_size=2).

GPU Allocation Logic:
- Input: [7, 5, 3] means 3 servers
- Server 1: GPUs 7,6 (port 8054)
- Server 2: GPUs 5,4 (port 8055)
- Server 3: GPUs 3,2 (port 8056)
"""

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VLLMServerConfig:
    """Configuration for a single VLLM server instance."""

    def __init__(
        self,
        instance_id: int,
        gpu_pair: Tuple[int, int],
        port: int,
        model_name: str = "openai/gpt-oss-120b",
        max_model_len: int = 65536,
        gpu_memory_utilization: float = 0.9,
    ):
        self.instance_id = instance_id
        self.gpu_pair = gpu_pair
        self.port = port
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        # File paths
        self.pid_file = Path(f"vllm_server_{instance_id}.pid")
        self.stdout_log = Path(f"vllm_server_{instance_id}_stdout.log")
        self.stderr_log = Path(f"vllm_server_{instance_id}_stderr.log")

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://localhost:{self.port}/v1"

    @property
    def cuda_visible_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string for this instance."""
        return f"{self.gpu_pair[0]},{self.gpu_pair[1]}"


class VLLMServerManager:
    """
    Manages multiple VLLM server instances.

    Handles starting, stopping, health checking, and configuration
    of multiple VLLM servers with proper GPU isolation.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize server manager.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self.servers: Dict[int, VLLMServerConfig] = {}
        self.processes: Dict[int, subprocess.Popen] = {}

        if config_path and config_path.exists():
            self.load_config(config_path)

    def load_config(self, config_path: Path):
        """Load server configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_name = config.get("model_name", "openai/gpt-oss-120b")
        base_port = config.get("base_port", 8054)

        gpu_list = config.get("gpu_list", [])
        if not gpu_list:
            raise ValueError("gpu_list must be specified in config")

        self.configure_from_gpu_list(gpu_list, model_name, base_port)

    def configure_from_gpu_list(
        self,
        gpu_list: List[int],
        model_name: str = "openai/gpt-oss-120b",
        base_port: int = 8054,
        **kwargs,
    ):
        """
        Configure servers from a GPU list.

        GPU allocation logic:
        - Each server needs 2 GPUs (tensor_parallel_size=2)
        - GPUs are allocated in descending pairs
        - Example: [7, 5, 3] → servers get (7,6), (5,4), (3,2)

        Args:
            gpu_list: List of GPU IDs (one per server, descending)
            model_name: Model to load
            base_port: Starting port number
        """
        self.servers.clear()

        for i, gpu_high in enumerate(gpu_list):
            gpu_low = gpu_high - 1

            if gpu_low < 0:
                raise ValueError(
                    f"GPU {gpu_high} would pair with {gpu_low} (invalid). "
                    "Ensure all GPUs can form valid pairs."
                )

            config = VLLMServerConfig(
                instance_id=i,
                gpu_pair=(gpu_high, gpu_low),
                port=base_port + i,
                model_name=model_name,
                **kwargs,
            )

            self.servers[i] = config
            logger.info(
                f"Configured Server {i}: GPUs {gpu_high},{gpu_low} on port {config.port}"
            )

    def start_server(self, instance_id: int) -> Optional[subprocess.Popen]:
        """Start a single VLLM server instance."""
        if instance_id not in self.servers:
            logger.error(f"Server {instance_id} not configured")
            return None

        config = self.servers[instance_id]

        # Check if already running
        if config.pid_file.exists():
            logger.warning(
                f"PID file {config.pid_file} exists. "
                f"Server {instance_id} may already be running."
            )
            return None

        # Build command
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            f"--model={config.model_name}",
            "--tensor-parallel-size=2",
            "--host=0.0.0.0",
            f"--port={config.port}",
            f"--gpu-memory-utilization={config.gpu_memory_utilization}",
            f"--max-model-len={config.max_model_len}",
            "--dtype=bfloat16",
        ]

        # Set environment with GPU isolation
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

        logger.info(f"Starting Server {instance_id}:")
        logger.info(f"  Command: {' '.join(command)}")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {config.cuda_visible_devices}")
        logger.info(f"  Port: {config.port}")

        try:
            with open(config.stdout_log, "w") as stdout, open(config.stderr_log, "w") as stderr:
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    env=server_env,
                    close_fds=True,
                )

            # Save PID
            with open(config.pid_file, "w") as f:
                f.write(str(process.pid))

            self.processes[instance_id] = process
            logger.info(f"Server {instance_id} started with PID {process.pid}")
            return process

        except Exception as e:
            logger.error(f"Failed to start server {instance_id}: {e}")
            return None

    def stop_server(self, instance_id: int):
        """Stop a single VLLM server instance."""
        if instance_id not in self.servers:
            logger.error(f"Server {instance_id} not configured")
            return

        config = self.servers[instance_id]

        if not config.pid_file.exists():
            logger.info(f"Server {instance_id} PID file not found. Not running.")
            return

        try:
            with open(config.pid_file, "r") as f:
                pid = int(f.read().strip())
        except Exception as e:
            logger.error(f"Could not read PID for server {instance_id}: {e}")
            try:
                config.pid_file.unlink()
            except:
                pass
            return

        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to server {instance_id} (PID {pid})")

            # Wait for graceful shutdown
            time.sleep(3)

            # Check if still alive
            try:
                os.kill(pid, 0)
                logger.warning(f"Server {instance_id} still alive. Sending SIGKILL.")
                os.kill(pid, signal.SIGKILL)
            except OSError:
                logger.info(f"Server {instance_id} terminated.")

        except ProcessLookupError:
            logger.warning(f"Process {pid} not found. Already stopped?")
        except Exception as e:
            logger.error(f"Error stopping server {instance_id}: {e}")
        finally:
            try:
                config.pid_file.unlink()
            except:
                pass

            if instance_id in self.processes:
                del self.processes[instance_id]

    def start_all(self, wait_for_ready: bool = True, timeout: int = 180):
        """
        Start all configured servers.

        Args:
            wait_for_ready: Wait for all servers to be ready
            timeout: Timeout per server in seconds
        """
        logger.info(f"Starting {len(self.servers)} VLLM servers...")

        for instance_id in self.servers:
            process = self.start_server(instance_id)

            if process and wait_for_ready:
                if self.wait_for_server(instance_id, timeout):
                    logger.info(f"✓ Server {instance_id} is ready")
                else:
                    logger.error(f"✗ Server {instance_id} failed to become ready")
                    self.stop_server(instance_id)
                    return False

        logger.info("All servers started successfully!")
        return True

    def stop_all(self):
        """Stop all running servers."""
        logger.info("Stopping all VLLM servers...")

        for instance_id in list(self.servers.keys()):
            self.stop_server(instance_id)

        logger.info("All servers stopped.")

    def wait_for_server(self, instance_id: int, timeout: int = 180) -> bool:
        """
        Wait for a server to be ready.

        Args:
            instance_id: Server instance ID
            timeout: Timeout in seconds

        Returns:
            True if server became ready, False otherwise
        """
        if instance_id not in self.servers:
            return False

        config = self.servers[instance_id]
        url = f"http://localhost:{config.port}/v1/models"

        start_time = time.time()
        logger.info(f"Waiting for server {instance_id} at {url}...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        logger.error(f"Server {instance_id} not ready after {timeout}s")
        return False

    def health_check(self, instance_id: int) -> bool:
        """Check if a server is healthy."""
        if instance_id not in self.servers:
            return False

        config = self.servers[instance_id]

        try:
            response = requests.get(f"http://localhost:{config.port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_server_urls(self) -> List[str]:
        """Get list of all server URLs."""
        return [config.url for config in self.servers.values()]

    def save_config(self, output_path: Path):
        """Save current configuration to YAML file."""
        config = {
            "model_name": self.servers[0].model_name if self.servers else "",
            "base_port": min(s.port for s in self.servers.values()) if self.servers else 8054,
            "gpu_list": [s.gpu_pair[0] for s in self.servers.values()],
            "servers": [
                {
                    "instance_id": s.instance_id,
                    "gpu_pair": list(s.gpu_pair),
                    "port": s.port,
                    "url": s.url,
                }
                for s in self.servers.values()
            ],
        }

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Configuration saved to {output_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VLLM Multi-Server Manager")
    parser.add_argument(
        "command", choices=["start", "stop", "status", "config"], help="Command to execute"
    )
    parser.add_argument(
        "--gpu-list", type=str, help="Comma-separated GPU IDs (e.g., '7,5,3' for 3 servers)"
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-oss-120b", help="Model name to load"
    )
    parser.add_argument("--base-port", type=int, default=8054, help="Starting port number")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/vllm_servers.yaml"),
        help="Configuration file path",
    )

    args = parser.parse_args()

    manager = VLLMServerManager()

    if args.command == "start":
        if args.gpu_list:
            gpu_list = [int(x.strip()) for x in args.gpu_list.split(",")]
            manager.configure_from_gpu_list(
                gpu_list, model_name=args.model, base_port=args.base_port
            )
        elif args.config.exists():
            manager.load_config(args.config)
        else:
            logger.error("Specify --gpu-list or provide config file")
            sys.exit(1)

        # Save configuration
        args.config.parent.mkdir(exist_ok=True, parents=True)
        manager.save_config(args.config)

        # Start servers
        if manager.start_all():
            logger.info("All servers running!")
        else:
            logger.error("Failed to start all servers")
            manager.stop_all()
            sys.exit(1)

    elif args.command == "stop":
        if args.config.exists():
            manager.load_config(args.config)
            manager.stop_all()
        else:
            logger.error("No config file found")
            sys.exit(1)

    elif args.command == "status":
        if args.config.exists():
            manager.load_config(args.config)
            print("\nVLLM Server Status:")
            print("=" * 60)
            for instance_id, config in manager.servers.items():
                healthy = "✓" if manager.health_check(instance_id) else "✗"
                print(f"{healthy} Server {instance_id}:")
                print(f"    GPUs: {config.cuda_visible_devices}")
                print(f"    Port: {config.port}")
                print(f"    URL: {config.url}")
        else:
            logger.error("No config file found")
            sys.exit(1)

    elif args.command == "config":
        if args.gpu_list:
            gpu_list = [int(x.strip()) for x in args.gpu_list.split(",")]
            manager.configure_from_gpu_list(
                gpu_list, model_name=args.model, base_port=args.base_port
            )
            args.config.parent.mkdir(exist_ok=True, parents=True)
            manager.save_config(args.config)
            logger.info(f"Configuration created: {args.config}")
        else:
            logger.error("--gpu-list required for config command")
            sys.exit(1)


if __name__ == "__main__":
    main()

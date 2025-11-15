"""
Enhanced VLLM Server for GPT-OSS with Multi-Instance Support

Improvements:
- Support for instance IDs (for running multiple servers)
- Configurable ports and GPU allocation
- Better integration with VLLMServerManager
- Backward compatible with original usage
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import requests

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants (can be overridden)
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_VLLM_HOST = "0.0.0.0"
DEFAULT_VLLM_PORT = 8054
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_TENSOR_PARALLEL_SIZE = 2
DEFAULT_MAX_MODEL_LEN = 65536
DEFAULT_MAX_TOKENS = 50000
DEFAULT_DTYPE = "bfloat16"


class ServerConfig:
    """Configuration for a VLLM server instance."""

    def __init__(
        self,
        instance_id: int = 0,
        model_name: str = DEFAULT_MODEL_NAME,
        host: str = DEFAULT_VLLM_HOST,
        port: int = DEFAULT_VLLM_PORT,
        gpu_ids: Optional[List[int]] = None,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        dtype: str = DEFAULT_DTYPE,
    ):
        """
        Initialize server configuration.

        Args:
            instance_id: Unique identifier for this server instance
            model_name: Model to load
            host: Host to bind to
            port: Port to bind to
            gpu_ids: List of GPU IDs to use (e.g., [7, 6])
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_model_len: Maximum model context length
            dtype: Data type for model
        """
        self.instance_id = instance_id
        self.model_name = model_name
        self.host = host
        self.port = port
        self.gpu_ids = gpu_ids or [0, 1]
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype

        # Validate GPU configuration
        if len(self.gpu_ids) != self.tensor_parallel_size:
            raise ValueError(
                f"Number of GPU IDs ({len(self.gpu_ids)}) must match "
                f"tensor_parallel_size ({self.tensor_parallel_size})"
            )

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def cuda_visible_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string."""
        return ",".join(str(gpu) for gpu in self.gpu_ids)

    @property
    def pid_file(self) -> Path:
        """Get PID file path for this instance."""
        if self.instance_id == 0:
            return Path("vllm_server.pid")  # Backward compatibility
        return Path(f"vllm_server_{self.instance_id}.pid")

    @property
    def stdout_log(self) -> Path:
        """Get stdout log path for this instance."""
        if self.instance_id == 0:
            return Path("vllm_server_stdout.log")
        return Path(f"vllm_server_{self.instance_id}_stdout.log")

    @property
    def stderr_log(self) -> Path:
        """Get stderr log path for this instance."""
        if self.instance_id == 0:
            return Path("vllm_server_stderr.log")
        return Path(f"vllm_server_{self.instance_id}_stderr.log")


class VLLMServer:
    """Enhanced VLLM server manager with instance support."""

    def __init__(self, config: ServerConfig):
        """
        Initialize VLLM server.

        Args:
            config: Server configuration
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> Optional[subprocess.Popen]:
        """Start the VLLM server."""
        if self.config.pid_file.exists():
            logger.warning(
                f"PID file {self.config.pid_file} exists. "
                f"Server instance {self.config.instance_id} may already be running."
            )
            return None

        # Build command
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            f"--model={self.config.model_name}",
            f"--tensor-parallel-size={self.config.tensor_parallel_size}",
            f"--host={self.config.host}",
            f"--port={self.config.port}",
            f"--gpu-memory-utilization={self.config.gpu_memory_utilization}",
            f"--max-model-len={self.config.max_model_len}",
            f"--dtype={self.config.dtype}",
        ]

        # Set environment with GPU isolation
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices

        logger.info(f"Starting VLLM Server Instance {self.config.instance_id}:")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Port: {self.config.port}")
        logger.info(f"  GPUs: {self.config.cuda_visible_devices}")
        logger.info(f"  Command: {' '.join(command)}")

        try:
            with (
                open(self.config.stdout_log, "w") as stdout_log,
                open(self.config.stderr_log, "w") as stderr_log,
            ):
                self.process = subprocess.Popen(
                    command,
                    stdout=stdout_log,
                    stderr=stderr_log,
                    env=server_env,
                    close_fds=True,
                )

            # Save PID
            with open(self.config.pid_file, "w") as f:
                f.write(str(self.process.pid))

            logger.info(
                f"Server instance {self.config.instance_id} started with PID {self.process.pid}"
            )
            logger.info(f"  PID file: {self.config.pid_file}")
            logger.info(f"  Logs: {self.config.stdout_log}, {self.config.stderr_log}")

            return self.process

        except FileNotFoundError:
            logger.error("Could not find 'python' or vLLM modules. Ensure vLLM is installed.")
            return None
        except Exception as e:
            logger.error(f"Failed to start VLLM server: {e}")
            return None

    def stop(self):
        """Stop the VLLM server."""
        if not self.config.pid_file.exists():
            logger.info(
                f"PID file not found for instance {self.config.instance_id}. "
                "Server is not running."
            )
            return

        try:
            with open(self.config.pid_file, "r") as f:
                pid = int(f.read().strip())
        except Exception as e:
            logger.error(f"Could not read PID from {self.config.pid_file}: {e}")
            try:
                self.config.pid_file.unlink()
            except:
                pass
            return

        try:
            os.kill(pid, 15)  # SIGTERM
            logger.info(
                f"Sent SIGTERM to server instance {self.config.instance_id} (PID {pid}). "
                "Waiting for shutdown..."
            )
            time.sleep(3)

            # Check if still alive
            try:
                os.kill(pid, 0)
                logger.warning(f"Process PID {pid} still alive. Sending SIGKILL.")
                os.kill(pid, 9)
            except OSError:
                logger.info(f"Server instance {self.config.instance_id} terminated.")

        except ProcessLookupError:
            logger.warning(f"Process with PID {pid} not found. Already stopped?")
        except Exception as e:
            logger.error(f"Error killing process {pid}: {e}")
        finally:
            try:
                self.config.pid_file.unlink()
                logger.info(f"Removed PID file {self.config.pid_file}")
            except OSError:
                pass

    def wait_for_ready(self, timeout: int = 180) -> bool:
        """
        Wait for the server to be ready.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if server became ready, False otherwise
        """
        url = f"http://{self.config.host}:{self.config.port}/v1/chat/completions"
        start_time = time.time()

        logger.info(
            f"Waiting for server instance {self.config.instance_id} "
            f"at {url} to be ready (timeout: {timeout}s)..."
        )

        while time.time() - start_time < timeout:
            try:
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": self.config.model_name,
                    "messages": [{"role": "user", "content": "Ping"}],
                    "max_tokens": 5,
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=data, timeout=10)

                if response.status_code == 200:
                    logger.info(f"Server instance {self.config.instance_id} is ready!")
                    return True
                else:
                    logger.debug(f"Status code {response.status_code}, retrying...")

            except requests.exceptions.RequestException as e:
                logger.debug(f"Connection error: {e}")

            time.sleep(2)

        logger.error(
            f"Server instance {self.config.instance_id} not ready "
            f"after {timeout} seconds. Check logs:\n"
            f"  stdout: {self.config.stdout_log}\n"
            f"  stderr: {self.config.stderr_log}"
        )
        return False

    def status(self) -> dict:
        """Get server status."""
        status = {
            "instance_id": self.config.instance_id,
            "port": self.config.port,
            "url": self.config.url,
            "gpu_ids": self.config.gpu_ids,
            "pid_file": str(self.config.pid_file),
            "is_running": False,
            "pid": None,
        }

        if self.config.pid_file.exists():
            try:
                with open(self.config.pid_file, "r") as f:
                    pid = int(f.read().strip())
                    status["pid"] = pid

                    # Check if process is still running
                    try:
                        os.kill(pid, 0)
                        status["is_running"] = True
                    except OSError:
                        status["is_running"] = False
            except Exception as e:
                logger.error(f"Error reading PID file: {e}")

        return status


# Pydantic schemas for structured output
class Response(BaseModel):
    texts: List[str] = Field(..., description="The returned text from the model")


class SingleClassificationResult(BaseModel):
    """Result of classifying a single text against a single node."""

    is_relevant: bool
    confidence: int
    reasoning: str
    excerpt: str


class ListOfSingleClassificationResult(BaseModel):
    results: List[SingleClassificationResult]


# Inference functions (maintained for backward compatibility)
async def run_single_inference(
    prompt: str,
    schema_class: type[BaseModel],
    model: str,
    async_client: AsyncOpenAI,
    prompt_index: int,
) -> dict:
    """Run inference for a single prompt."""
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await async_client.responses.parse(
            model=model,
            input=messages,
            text_format=schema_class,
        )

        if hasattr(response, "output_parsed") and isinstance(
            response.output_parsed, schema_class
        ):
            result_text = response.output_parsed
            logger.info(f"Prompt {prompt_index} result: \n └── {result_text}")

            return {
                "prompt_index": prompt_index,
                "success": True,
                "result": result_text.model_dump(),
            }
        else:
            logger.warning(f"Prompt {prompt_index}: Could not reliably access 'output_parsed'")
            return {
                "prompt_index": prompt_index,
                "success": False,
                "error": "Could not parse output",
                "raw_response": (
                    response.model_dump() if hasattr(response, "model_dump") else str(response)
                ),
            }

    except Exception as e:
        logger.error(f"Error during inference for prompt {prompt_index}: {e}")
        return {
            "prompt_index": prompt_index,
            "success": False,
            "error": str(e),
        }


async def run_inference(
    schema_class: type[BaseModel],
    prompt_file: str,
    model_name: str = None,
    output_file: str | None = None,
    max_concurrent: int = 5,
    server_url: str = None,
):
    """Run inference requests against the VLLM server for multiple prompts."""
    # Load prompts from JSON file
    try:
        with open(prompt_file, "r") as f:
            prompts = json.load(f)
        logger.info(f"Loaded {len(prompts)} prompts from '{prompt_file}'")
    except FileNotFoundError:
        logger.error(f"Error: Prompt file '{prompt_file}' not found.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error: Invalid JSON in '{prompt_file}': {e}")
        return

    if not isinstance(prompts, list):
        logger.error("Error: Prompt file must contain a JSON array of prompts")
        return

    prompts = prompts[:100]
    model = model_name if model_name else DEFAULT_MODEL_NAME
    vllm_url = server_url if server_url else f"http://localhost:{DEFAULT_VLLM_PORT}/v1"

    async_client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")

    # Create tasks for all prompts with a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_inference(prompt, idx):
        async with semaphore:
            return await run_single_inference(prompt, schema_class, model, async_client, idx)

    # Run all inferences concurrently (with limit)
    logger.info(f"Starting inference for {len(prompts)} prompts...")
    tasks = [bounded_inference(prompt, idx) for idx, prompt in enumerate(prompts)]

    all_results = await asyncio.gather(*tasks)

    # Separate successful and failed results
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]

    logger.info(
        f"Inference complete: {len(successful_results)} successful, "
        f"{len(failed_results)} failed"
    )

    # Prepare output data
    output_data = {
        "metadata": {
            "prompt_file": prompt_file,
            "model_name": model,
            "total_prompts": len(prompts),
            "successful": len(successful_results),
            "failed": len(failed_results),
        },
        "results": successful_results,
        "errors": failed_results if failed_results else None,
    }

    # Save results to file
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Results successfully written to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write results to '{output_file}': {e}")

    return output_data


def main():
    """Enhanced CLI entry point with instance support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced VLLM Server with Multi-Instance Support"
    )
    parser.add_argument(
        "command", choices=["start", "stop", "status", "run"], help="Command to execute"
    )

    # Server configuration
    parser.add_argument(
        "--instance-id", type=int, default=0, help="Server instance ID (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port number (default: {DEFAULT_VLLM_PORT} + instance_id)",
    )
    parser.add_argument(
        "--gpu-ids", type=str, default=None, help="Comma-separated GPU IDs (e.g., '7,6')"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help=f"Tensor parallel size (default: {DEFAULT_TENSOR_PARALLEL_SIZE})",
    )

    # Inference options
    parser.add_argument("--prompt-file", type=str, help="Path to prompt file for inference")
    parser.add_argument("--output-file", type=str, help="Output file for inference results")
    parser.add_argument("--server-url", type=str, help="Server URL for inference")

    args = parser.parse_args()

    # Determine port
    if args.port is None:
        port = DEFAULT_VLLM_PORT + args.instance_id
    else:
        port = args.port

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        # Default: descending pairs based on instance_id
        # Instance 0: [7,6], Instance 1: [5,4], Instance 2: [3,2]
        high_gpu = 7 - (args.instance_id * 2)
        gpu_ids = [high_gpu, high_gpu - 1]

    # Create configuration
    config = ServerConfig(
        instance_id=args.instance_id,
        model_name=args.model,
        port=port,
        gpu_ids=gpu_ids,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Create server
    server = VLLMServer(config)

    # Execute command
    if args.command == "start":
        process = server.start()
        if process and server.wait_for_ready():
            logger.info("VLLM server is running and ready!")
            logger.info(f"  URL: {config.url}")
            logger.info(f"  GPUs: {config.cuda_visible_devices}")
        elif process:
            logger.error("Server process started but did not become ready in time.")
            server.stop()
            sys.exit(1)
        else:
            logger.error("Failed to start server.")
            sys.exit(1)

    elif args.command == "stop":
        server.stop()
        sys.exit(0)

    elif args.command == "status":
        status = server.status()
        print(f"\nVLLM Server Instance {status['instance_id']} Status:")
        print("=" * 60)
        print(f"  Port: {status['port']}")
        print(f"  URL: {status['url']}")
        print(f"  GPUs: {status['gpu_ids']}")
        print(f"  PID File: {status['pid_file']}")
        print(f"  Running: {'✓ Yes' if status['is_running'] else '✗ No'}")
        if status["pid"]:
            print(f"  PID: {status['pid']}")
        sys.exit(0)

    elif args.command == "run":
        if not args.prompt_file:
            logger.error("--prompt-file required for run command")
            sys.exit(1)

        SCHEMA_TO_USE = ListOfSingleClassificationResult

        # Check if server is running
        if not server.wait_for_ready(timeout=10):
            logger.error(
                "Server is not running or not responding. Please start the server first."
            )
            sys.exit(1)

        server_url = args.server_url or config.url
        asyncio.run(
            run_inference(
                SCHEMA_TO_USE,
                args.prompt_file,
                args.model,
                args.output_file,
                server_url=server_url,
            )
        )
        sys.exit(0)


if __name__ == "__main__":
    main()

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import List

import aiohttp
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import requests

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "openai/gpt-oss-120b"
# "casperhansen/llama-3.3-70b-instruct-awq"
# "openai/gpt-oss-20b"
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8054
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 10240
MAX_TOKENS = 50000
DTYPE = "bfloat16"
PID_FILE = "vllm_server.pid"
STDOUT_LOG = "vllm_server_stdout.log"
STDERR_LOG = "vllm_server_stderr.log"


class Response(BaseModel):
    texts: List[str] = Field(..., description="The returned text from the model")


class SingleClassificationResult(BaseModel):
    """
    Result of classifying a single text against a single node.

    Attributes:
        is_relevant: Whether the text is relevant to the node's topic
        confidence: Confidence score from 1-5 (1=very uncertain, 5=very certain)
        reasoning: Explanation of the classification decision (1-2 sentences)
        excerpt: Exact text span supporting the classification (empty if not relevant)
    """

    is_relevant: bool
    confidence: int
    reasoning: str
    excerpt: str


# class list of SingleClassificationResult
class ListOfSingleClassificationResult(BaseModel):
    results: List[SingleClassificationResult]


def start_server(model_name: str = None):
    """Start the vLLM OpenAI-compatible API server for gpt-oss-120b."""
    if os.path.exists(PID_FILE):
        logger.warning(f"PID file {PID_FILE} exists. Server may already be running.")
        return None

    model = model_name if model_name else MODEL_NAME

    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        f"--model={model}",
        f"--tensor-parallel-size={TENSOR_PARALLEL_SIZE}",
        f"--host={VLLM_HOST}",
        f"--port={VLLM_PORT}",
        f"--gpu-memory-utilization={GPU_MEMORY_UTILIZATION}",
        f"--max-model-len={MAX_MODEL_LEN}",
        f"--dtype={DTYPE}",
    ]

    server_env = os.environ.copy()

    # 1. Force use of InfiniBand (IB)
    # server_env["NCCL_NET"] = "IB"

    # add for link diagnostics :
    # server_env["NCCL_DEBUG"] = "INFO"

    # logger.info(
    #     f"Using NCCL environment: NCCL_NET={server_env['NCCL_NET']}, NCCL_DEBUG={server_env['NCCL_DEBUG']}"
    # )

    logger.info(f"Starting vLLM server with command: {' '.join(command)}")

    try:
        with open(STDOUT_LOG, "w") as stdout_log, open(STDERR_LOG, "w") as stderr_log:
            process = subprocess.Popen(
                command,
                stdout=stdout_log,
                stderr=stderr_log,
                close_fds=True,
                # env=server_env,
            )
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))
        logger.info(f"Server started with PID {process.pid}. PID saved to {PID_FILE}")
        return process
    except FileNotFoundError:
        logger.error("Could not find 'python' or vLLM modules. Ensure vLLM is installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        return None


def stop_server():
    """Stop the vLLM server using the PID file."""
    if not os.path.exists(PID_FILE):
        logger.info("PID file not found. Server is not running.")
        return
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
    except Exception as e:
        logger.error(f"Could not read PID from {PID_FILE}: {e}")
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        return
    try:
        os.kill(pid, 15)  # SIGTERM
        # pkill -9 -f VLLM
        logger.info(f"Sent SIGTERM to server with PID {pid}. Waiting for shutdown...")
        time.sleep(3)
        try:
            os.kill(pid, 0)
            logger.warning(f"Process PID {pid} still alive. Sending SIGKILL.")
            os.kill(pid, 9)
        except OSError:
            logger.info(f"Server PID {pid} terminated.")
    except ProcessLookupError:
        logger.warning(f"Process with PID {pid} not found. Already stopped?")
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
    finally:
        try:
            os.remove(PID_FILE)
            logger.info(f"Removed PID file {PID_FILE}")
        except OSError:
            pass


def wait_for_server(timeout=180):
    """Wait for the vLLM server to be ready."""
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"
    start_time = time.time()
    logger.info(f"Waiting for server at {url} to be ready (timeout: {timeout}s)...")
    while time.time() - start_time < timeout:
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Ping"}],
                "max_tokens": 5,
                "temperature": 0,
            }
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready.")
                return True
            else:
                logger.debug(f"Status code {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            logger.debug(f"Connection error: {e}")
        time.sleep(2)
    logger.error(f"Server not ready after {timeout} seconds. Check logs.")
    return False


def run_inference_http(prompt_file: str, output_file: str = None):
    """Run inference against the running vLLM server using gpt-oss-120b."""
    try:
        with open(prompt_file, "r") as file:
            prompt = file.read()
        logger.info(f"Read prompt from file: '{prompt_file}'")
    except FileNotFoundError:
        logger.error(f"Error: Prompt file '{prompt_file}' not found.")
        return

    url = f"{VLLM_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        # "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Inference result: {result}")
            if output_file:
                try:
                    with open(output_file, "w") as f:
                        json.dump(result, f, indent=4)
                    logger.info(f"Inference result written to: {output_file}")
                except Exception as e:
                    logger.error(f"Failed to write result to '{output_file}': {e}")
            else:
                print(json.dumps(result, indent=4))
        else:
            logger.error(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")


async def run_single_inference(
    prompt: str,
    schema_class: type[BaseModel],
    model: str,
    async_client: AsyncOpenAI,
    prompt_index: int,
) -> dict:
    """Run inference for a single prompt"""
    messages = [
        {"role": "user", "content": prompt},
    ]

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
    max_concurrent: int = 5,  # Nombre maximum de requêtes simultanées
):
    """
    Runs inference requests against the VLLM server for multiple prompts.

    Args:
        schema_class: Pydantic model class for parsing responses
        prompt_file: Path to JSON file containing list of prompts
        model_name: Model name to use
        output_file: Path to save results as JSON
        max_concurrent: Maximum number of concurrent requests
    """
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
    model = model_name if model_name else MODEL_NAME
    async_client = AsyncOpenAI(base_url=VLLM_URL, api_key="EMPTY")

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


async def call_vllm_async(session, payload):
    url = f"{VLLM_URL}/completions"

    async with session.post(url, json=payload) as response:
        if response.status == 200:
            data = await response.json()
            cleaned_text = data["choices"][0]["text"].strip()
            logger.info(f"Response received (raw text, supposed JSON): {cleaned_text}")
            return cleaned_text
        else:
            error_text = await response.text()
            logger.error(f"API Error: {response.status} {error_text}")
            return None


async def run_async_inference(
    prompt_file: str,
    schema_class: type[BaseModel],
    output_file: str | None = None,
):
    """
    Main asynchronous function to run guided inference and parse the result
    with a generic Pydantic model passed as schema_class.
    """
    try:
        with open(prompt_file, "r") as file:
            prompt = file.read()
        logger.info(f"Read prompt from file: '{prompt_file}' for async guided inference.")
    except FileNotFoundError:
        logger.error(f"Error: Prompt file '{prompt_file}' not found.")
        return

    schema_json = schema_class.model_json_schema()

    payload = {
        "prompt": prompt,
        "model": MODEL_NAME,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "n": 1,
        "stream": False,
        "guided_json": schema_json,
    }

    # 2. Exécution de la requête
    try:
        async with aiohttp.ClientSession() as session:
            response_text = await call_vllm_async(session, payload)

            if not response_text:
                logger.error("No response text received. Aborting parsing.")
                return

            try:
                parsed = schema_class.model_validate_json(response_text)

                result_data = parsed.model_dump()
                logger.info(f"Pydantic validation successful using {schema_class.__name__}.")
                logger.debug(f"Parsed data: {result_data}")

                if output_file:
                    try:
                        with open(output_file, "w") as f:
                            json.dump(result_data, f, indent=4)
                            print(f"Inference result successfully written to: {output_file}")
                        logger.info(
                            f"Async inference result successfully written to: {output_file}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to write result to JSON file '{output_file}': {e}"
                        )
                else:
                    print(json.dumps(result_data, indent=4))

            except Exception as e:
                logger.error(f"JSON Parsing/Validation Error for {schema_class.__name__}: {e}")
                logger.error(f"Raw text received: {response_text}")

    except Exception as e:
        logger.error(f"An error occurred during async inference: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_gpt_oss_120b_server.py <start|stop|status|run|run_async>")
        print(
            "       python run_gpt_oss_120b_server.py run <prompt_file.txt> [output_file.json] [model_name]"
        )
        print(
            "       python run_gpt_oss_120b_server.py run_async <prompt_file.txt> [output_file.json]"
        )
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "start":
        # Optionally allow: python vllm_server_gpt_oss.py start [model_name]
        model_name = sys.argv[2] if len(sys.argv) >= 3 else None
        process = start_server(model_name)
        if process and wait_for_server():
            logger.info("vLLM server is running and ready.")
        elif process:
            logger.error("Server process started but did not become ready in time.")
            try:
                process.terminate()
            except Exception:
                pass
            stop_server()
            sys.exit(1)
        else:
            logger.error("Failed to start server.")
            sys.exit(1)

    elif command == "stop":
        stop_server()
        sys.exit(0)

    elif command == "status":
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                pid = f.read().strip()
            print(f"Server PID file exists. PID: {pid}")
        else:
            print("Server is not running (no PID file found).")
        sys.exit(0)

    elif command == "run":
        # Allow: python vllm_server_gpt_oss.py run <prompt_file.txt> [output_file.json] [model_name]
        if len(sys.argv) < 3:
            print(
                "Usage: python vllm_server_gpt_oss.py run <prompt_file.txt> [output_file.json] [model_name]"
            )
            sys.exit(1)
        prompt_file = sys.argv[2]
        output_file = None
        model_name = None

        SCHEMA_TO_USE = ListOfSingleClassificationResult

        if len(sys.argv) == 4:
            # Could be output_file or model_name
            if sys.argv[3].endswith(".json"):
                output_file = sys.argv[3]
            else:
                model_name = sys.argv[3]
        elif len(sys.argv) == 5:
            output_file = sys.argv[3]
            model_name = sys.argv[4]
        # Check if the server is running before inference
        if not wait_for_server(timeout=10):
            logger.error(
                "Server is not running or not responding. Please start the server first."
            )
            sys.exit(1)
        asyncio.run(run_inference(SCHEMA_TO_USE, prompt_file, model_name, output_file))
        sys.exit(0)

    elif command == "run_async":
        # Allow: python vllm_server_gpt_oss.py run_async <prompt_file.txt> [output_file.json]
        if len(sys.argv) < 3:
            print(
                "Usage: python vllm_server_gpt_oss.py run_async <prompt_file.txt> [output_file.json]"
            )
            sys.exit(1)
        prompt_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None

        SCHEMA_TO_USE = SingleClassificationResult
        # -----------------------------------------------------------

        # Check if the server is running before inference
        if not wait_for_server(timeout=10):
            logger.error(
                "Server is not running or not responding. Please start the server first."
            )
            sys.exit(1)

        # Lance la fonction asynchrone avec le schéma en paramètre
        asyncio.run(run_async_inference(prompt_file, SCHEMA_TO_USE, output_file))
        sys.exit(0)

    else:
        print(f"Unknown command: {command}")
        print("Usage: python run_gpt_oss_120b_server.py <start|stop|status|run>")
        print("python run_gpt_oss_120b_server.py run <prompt_file.txt> [output_file.json]")
        sys.exit(1)

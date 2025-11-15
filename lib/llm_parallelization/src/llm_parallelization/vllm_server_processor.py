"""
Enhanced VLLM Server Processor with all improvements:
- Connection pooling
- Load balancing
- Health checks
- Automatic retry
- Metrics collection
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from typing import Dict, List, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ServerMetrics:
    """Metrics for a single server."""

    url: str
    total_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_latency(self) -> float:
        """Average latency in seconds."""
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.failed_requests) / self.total_requests) * 100


class VLLMServerProcessor:
    """
    Enhanced VLLM Server Processor with production features.

    Features:
    - Multiple server support with load balancing
    - Connection pooling per server
    - Automatic health checks
    - Retry logic with exponential backoff
    - Metrics collection
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8054/v1",
        server_urls: Optional[List[str]] = None,
        model_name: str = "openai/gpt-oss-120b",
        max_concurrent: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        health_check_interval: int = 30,
        pool_size: int = 3,
        **kwargs,
    ):
        """
        Initialize enhanced VLLM server processor.

        Args:
            server_url: Primary server URL (used if server_urls not provided)
            server_urls: List of server URLs for load balancing
            model_name: Model name to use
            max_concurrent: Maximum concurrent requests per server
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Initial retry delay in seconds
            health_check_interval: Seconds between health checks
            pool_size: Connection pool size per server
        """
        # Server configuration
        self.server_urls = server_urls if server_urls else [server_url]
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        self.pool_size = pool_size

        # Connection pools (one per server)
        self._client_pools: Dict[str, List[AsyncOpenAI]] = {}
        self._pool_locks: Dict[str, asyncio.Lock] = {}

        # Load balancing
        self._round_robin_index = 0
        self._server_locks: Dict[str, asyncio.Semaphore] = {}

        # Metrics
        self.metrics: Dict[str, ServerMetrics] = {}

        # Initialize for each server
        for url in self.server_urls:
            self._client_pools[url] = []
            self._pool_locks[url] = asyncio.Lock()
            self._server_locks[url] = asyncio.Semaphore(max_concurrent)
            self.metrics[url] = ServerMetrics(url=url)

        # Storage for results (FlexibleSchemaProcessor compatibility)
        self._results = []
        self._schema = None

        logger.info(f"Initialized VLLMServerProcessor with {len(self.server_urls)} servers")
        for url in self.server_urls:
            logger.info(f"  • {url}")

    async def _get_client(self, server_url: str) -> AsyncOpenAI:
        """Get a client from the pool or create a new one."""
        async with self._pool_locks[server_url]:
            pool = self._client_pools[server_url]

            if pool:
                return pool.pop()
            else:
                # Create new client
                return AsyncOpenAI(base_url=server_url, api_key="EMPTY")

    async def _return_client(self, server_url: str, client: AsyncOpenAI):
        """Return a client to the pool."""
        async with self._pool_locks[server_url]:
            pool = self._client_pools[server_url]

            if len(pool) < self.pool_size:
                pool.append(client)
            else:
                # Pool full, close this client
                await client.close()

    def _get_next_server(self) -> str:
        """Get next server URL using round-robin load balancing."""
        # Filter to healthy servers
        healthy_servers = [url for url in self.server_urls if self.metrics[url].is_healthy]

        if not healthy_servers:
            logger.warning("No healthy servers available, using all servers")
            healthy_servers = self.server_urls

        # Round-robin selection
        server = healthy_servers[self._round_robin_index % len(healthy_servers)]
        self._round_robin_index += 1

        return server

    async def _health_check(self, server_url: str) -> bool:
        """Check if a server is healthy."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server_url.replace('/v1', '')}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    is_healthy = response.status == 200
                    self.metrics[server_url].is_healthy = is_healthy
                    self.metrics[server_url].last_health_check = datetime.now()
                    return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for {server_url}: {e}")
            self.metrics[server_url].is_healthy = False
            return False

    async def _single_inference_with_retry(
        self,
        prompt: str,
        schema: Type[BaseModel],
        prompt_index: int,
        server_url: Optional[str] = None,
    ) -> dict:
        """
        Run inference with automatic retry and failover.

        Args:
            prompt: The prompt text
            schema: Pydantic schema
            prompt_index: Index for ordering
            server_url: Specific server to use (None for load balancing)
        """
        last_error = None

        for attempt in range(self.max_retries):
            # Select server
            if server_url is None:
                selected_server = self._get_next_server()
            else:
                selected_server = server_url

            try:
                # Acquire semaphore for this server
                async with self._server_locks[selected_server]:
                    # Get client from pool
                    client = await self._get_client(selected_server)

                    try:
                        start_time = time.time()

                        # Make request
                        result = await self._single_inference(
                            prompt, schema, prompt_index, client
                        )

                        # Record metrics
                        latency = time.time() - start_time
                        metrics = self.metrics[selected_server]
                        metrics.total_requests += 1
                        metrics.total_latency += latency
                        metrics.recent_latencies.append(latency)

                        if result["success"]:
                            return result
                        else:
                            metrics.failed_requests += 1
                            last_error = result.get("error", "Unknown error")

                    finally:
                        # Return client to pool
                        await self._return_client(selected_server, client)

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for "
                    f"server {selected_server}: {e}"
                )
                last_error = str(e)
                self.metrics[selected_server].failed_requests += 1

                # Mark server as potentially unhealthy
                if attempt == self.max_retries - 1:
                    self.metrics[selected_server].is_healthy = False

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        # All retries failed
        return {
            "prompt_index": prompt_index,
            "success": False,
            "error": f"All {self.max_retries} retries failed. Last error: {last_error}",
        }

    async def _single_inference(
        self, prompt: str, schema: Type[BaseModel], prompt_index: int, client: AsyncOpenAI
    ) -> dict:
        """Execute a single inference request."""
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await client.responses.parse(
                model=self.model_name,
                input=messages,
                text_format=schema,
            )

            if hasattr(response, "output_parsed") and isinstance(
                response.output_parsed, schema
            ):
                return {
                    "prompt_index": prompt_index,
                    "success": True,
                    "result": response.output_parsed,
                }
            else:
                return {
                    "prompt_index": prompt_index,
                    "success": False,
                    "error": "Could not parse output",
                }

        except Exception as e:
            import traceback

            return {
                "prompt_index": prompt_index,
                "success": False,
                "error": f"{str(e)}\n{traceback.format_exc()}",
            }

    def process_with_schema(
        self,
        prompts: List[str],
        schema: Type[BaseModel],
        batch_size: int = 25,
        formatted: bool = False,
    ) -> None:
        """Process prompts with retry and load balancing."""
        self._schema = schema
        self._results = asyncio.run(self._async_process_batch(prompts, schema))

    async def _async_process_batch(
        self, prompts: List[str], schema: Type[BaseModel]
    ) -> List[BaseModel]:
        """Process batch with enhanced features."""
        # Create tasks with retry
        tasks = [
            self._single_inference_with_retry(prompt, schema, idx)
            for idx, prompt in enumerate(prompts)
        ]

        # Execute all tasks
        results = await asyncio.gather(*tasks)

        # Extract successful results
        successful_results = []
        for result in sorted(results, key=lambda x: x["prompt_index"]):
            if result["success"]:
                successful_results.append(result["result"])
            else:
                raise RuntimeError(
                    f"Inference failed for prompt {result['prompt_index']}: "
                    f"{result.get('error', 'Unknown error')}"
                )

        return successful_results

    def parse_results_with_schema(
        self, schema: Type[BaseModel], validate: bool = True
    ) -> List[BaseModel]:
        """Return stored results."""
        return self._results

    def get_metrics_summary(self) -> Dict[str, Dict]:
        """Get metrics summary for all servers."""
        return {
            url: {
                "total_requests": m.total_requests,
                "failed_requests": m.failed_requests,
                "success_rate": f"{m.success_rate:.2f}%",
                "avg_latency": f"{m.avg_latency:.3f}s",
                "is_healthy": m.is_healthy,
                "last_health_check": m.last_health_check.isoformat()
                if m.last_health_check
                else None,
            }
            for url, m in self.metrics.items()
        }

    async def run_health_checks(self):
        """Run health checks on all servers."""
        tasks = [self._health_check(url) for url in self.server_urls]
        results = await asyncio.gather(*tasks)
        return dict(zip(self.server_urls, results))

    def terminate(self):
        """Cleanup all connections."""
        asyncio.run(self._cleanup_pools())

    async def _cleanup_pools(self):
        """Close all pooled clients."""
        for url, pool in self._client_pools.items():
            for client in pool:
                await client.close()
            pool.clear()

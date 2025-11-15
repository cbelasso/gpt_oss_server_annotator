"""
VLLM Server Processor - Adapter for external VLLM server via OpenAI API

This provides the same interface as FlexibleSchemaProcessor but communicates
with an external VLLM server via HTTP instead of loading models directly.
"""

import asyncio
from typing import List, Type

from openai import AsyncOpenAI
from pydantic import BaseModel


class VLLMServerProcessor:
    """
    Processor that communicates with external VLLM server.

    Provides same interface as FlexibleSchemaProcessor for compatibility.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8054/v1",
        model_name: str = "openai/gpt-oss-120b",
        max_concurrent: int = 5,
        **kwargs,  # Accept but ignore FlexibleSchemaProcessor args for compatibility
    ):
        """
        Initialize VLLM server processor.

        Args:
            server_url: URL of the VLLM server
            model_name: Model name to use
            max_concurrent: Maximum concurrent requests
        """
        self.server_url = server_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent

        # Storage for results (to match FlexibleSchemaProcessor interface)
        self._results = []
        self._schema = None

    def process_with_schema(
        self,
        prompts: List[str],
        schema: Type[BaseModel],
        batch_size: int = 25,
        formatted: bool = False,
    ) -> None:
        """
        Process prompts and store results (sync wrapper for async processing).

        Args:
            prompts: List of prompts to process
            schema: Pydantic schema for structured output
            batch_size: Batch size (used for grouping, not enforced by server)
            formatted: Whether prompts are pre-formatted
        """
        self._schema = schema
        # Run async processing in sync context
        self._results = asyncio.run(self._async_process_batch(prompts, schema))

    async def _async_process_batch(
        self, prompts: List[str], schema: Type[BaseModel]
    ) -> List[BaseModel]:
        """
        Process batch of prompts asynchronously with concurrency control.

        Args:
            prompts: List of prompts
            schema: Pydantic schema for output

        Returns:
            List of parsed results
        """
        # Create a fresh client for this batch
        client = AsyncOpenAI(base_url=self.server_url, api_key="EMPTY")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_inference(prompt: str, idx: int):
            async with semaphore:
                return await self._single_inference(prompt, schema, idx, client)

        # Create tasks for all prompts
        tasks = [bounded_inference(prompt, idx) for idx, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        # Close the client after batch
        await client.close()

        # Extract successful results and maintain order
        successful_results = []
        for result in sorted(results, key=lambda x: x["prompt_index"]):
            if result["success"]:
                successful_results.append(result["result"])
            else:
                # Raise with detailed error
                raise RuntimeError(
                    f"Inference failed for prompt {result['prompt_index']}: "
                    f"{result.get('error', 'Unknown error')}"
                )

        return successful_results

    async def _single_inference(
        self, prompt: str, schema: Type[BaseModel], prompt_index: int, client: AsyncOpenAI
    ) -> dict:
        """
        Run inference for a single prompt.

        Args:
            prompt: The prompt text
            schema: Pydantic schema
            prompt_index: Index for ordering
            client: AsyncOpenAI client instance

        Returns:
            Dict with success status and result
        """
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
                    "result": response.output_parsed,  # Already a Pydantic model
                }
            else:
                return {
                    "prompt_index": prompt_index,
                    "success": False,
                    "error": "Could not parse output",
                }

        except Exception as e:
            # Log the actual error for debugging
            import traceback

            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            return {
                "prompt_index": prompt_index,
                "success": False,
                "error": error_detail,
            }

    def parse_results_with_schema(
        self, schema: Type[BaseModel], validate: bool = True
    ) -> List[BaseModel]:
        """
        Return stored results (already parsed).

        Args:
            schema: Schema (for compatibility, already used in processing)
            validate: Validation flag (for compatibility)

        Returns:
            List of Pydantic model instances
        """
        return self._results

    def terminate(self):
        """Cleanup (no-op for server-based processor)."""
        pass

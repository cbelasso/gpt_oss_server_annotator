"""
Request Manager for Intelligent Batching

Handles request batching, scheduling, and optimal capability execution
to maximize GPU utilization across heterogeneous requests.

Key Features:
- Batches requests over a time window
- Groups requests by capability sets
- Executes capabilities in dependency order
- Minimizes redundant processing
"""

import asyncio
from typing import Any, Dict, List, Set

from rich.console import Console

from classifier.capabilities import CapabilityRegistry

console = Console()


class ProcessorPool:
    def __init__(
        self,
        config_path,
        gpu_list: List[int],
        policy,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 10240,
        batch_size: int = 25,
        backend: str = "local",
        server_urls: List[str] = None,
        max_concurrent: int = 5,
    ):
        from classifier import ClassificationProcessor

        self.config_path = config_path
        self.gpu_list = gpu_list
        self.batch_size = batch_size
        self.backend = backend

        # Initialize processor
        console.print(f"[cyan]Initializing processor ({backend} backend)...[/cyan]")
        self.processor = ClassificationProcessor(
            config_path=config_path,
            gpu_list=gpu_list,
            policy=policy,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            batch_size=batch_size,
            backend=backend,
            server_urls=server_urls,
            max_concurrent=max_concurrent,
        )
        console.print("[green]✓ Processor ready[/green]")

        self._lock = asyncio.Lock()
        self._is_ready = True

    def is_ready(self) -> bool:
        """Check if processor is ready."""
        return self._is_ready

    def get_processor(self):
        """Get the processor instance."""
        return self.processor

    def shutdown(self):
        """Shutdown the processor."""
        if self.processor:
            console.print("[cyan]Shutting down processor...[/cyan]")
            self.processor.cleanup()
            self._is_ready = False
            console.print("[green]✓ Processor shutdown complete[/green]")


class RequestBatch:
    """
    A batch of texts with their requested capabilities.

    Tracks which texts need which capabilities for intelligent execution.
    """

    def __init__(self):
        self.texts: List[str] = []
        self.text_to_capabilities: Dict[str, List[str]] = {}
        self.text_to_project: Dict[str, str] = {}
        self.request_ids: List[str] = []
        self.futures: List[asyncio.Future] = []

    def add_request(
        self,
        request_id: str,
        texts: List[str],
        capabilities: List[str],
        project_name: str = None,
    ) -> asyncio.Future:
        """Add a request to the batch."""
        future = asyncio.Future()

        for text in texts:
            if text not in self.texts:
                self.texts.append(text)

            self.text_to_capabilities[text] = capabilities
            self.text_to_project[text] = project_name

        self.request_ids.append(request_id)
        self.futures.append((request_id, texts, future))

        return future

    def get_all_capabilities(self) -> Set[str]:
        """Get set of all capabilities needed for this batch."""
        all_caps = set()
        for caps in self.text_to_capabilities.values():
            all_caps.update(caps)
        return all_caps

    def get_texts_for_capability(self, capability: str) -> List[str]:
        """Get texts that need a specific capability."""
        return [text for text, caps in self.text_to_capabilities.items() if capability in caps]

    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return len(self.texts) == 0


class RequestBatcher:
    """
    Intelligent request batcher with capability-aware processing.

    Batches requests over a time window and executes capabilities optimally
    to maximize GPU utilization and minimize redundant processing.
    """

    def __init__(
        self,
        processor_pool: ProcessorPool,
        registry: CapabilityRegistry,
        batch_timeout: float = 0.1,
    ):
        """
        Initialize request batcher.

        Args:
            processor_pool: ProcessorPool instance
            registry: CapabilityRegistry with registered capabilities
            batch_timeout: Maximum seconds to wait before processing batch
        """
        self.processor_pool = processor_pool
        self.registry = registry
        self.batch_timeout = batch_timeout

        self._current_batch = RequestBatch()
        self._batch_lock = asyncio.Lock()
        self._batch_timer_task = None
        self._request_counter = 0

    async def process_request(
        self,
        texts: List[str],
        capabilities: List[str],
        project_name: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process a classification request.

        Adds request to current batch and waits for batch processing.

        Args:
            texts: List of texts to classify
            capabilities: List of capabilities to execute
            project_name: Optional project name

        Returns:
            Dict mapping texts to their results
        """
        async with self._batch_lock:
            request_id = f"req_{self._request_counter}"
            self._request_counter += 1

            # Add to current batch
            future = self._current_batch.add_request(
                request_id=request_id,
                texts=texts,
                capabilities=capabilities,
                project_name=project_name,
            )

            # Start batch timer if not already running
            if self._batch_timer_task is None:
                self._batch_timer_task = asyncio.create_task(self._batch_timer())

        # Wait for results
        results = await future
        return results

    async def _batch_timer(self):
        """Timer that triggers batch processing after timeout."""
        await asyncio.sleep(self.batch_timeout)

        async with self._batch_lock:
            if not self._current_batch.is_empty():
                # Process the batch
                await self._process_batch()

            # Reset timer
            self._batch_timer_task = None

    async def _process_batch(self):
        """
        Process the current batch intelligently.

        Groups texts by capability requirements and executes capabilities
        in optimal order to maximize GPU utilization.
        """
        if self._current_batch.is_empty():
            return

        batch = self._current_batch
        self._current_batch = RequestBatch()  # Start new batch

        console.print(
            f"[cyan]Processing batch: {len(batch.texts)} texts, "
            f"{len(batch.get_all_capabilities())} capabilities[/cyan]"
        )

        try:
            # Execute capabilities efficiently
            all_results = await self._execute_batch_capabilities(batch)

            # Distribute results to futures
            for request_id, request_texts, future in batch.futures:
                request_results = {text: all_results[text] for text in request_texts}
                future.set_result(request_results)

        except Exception as e:
            console.print(f"[red]Batch processing error: {e}[/red]")
            # Set exception on all futures
            for _, _, future in batch.futures:
                if not future.done():
                    future.set_exception(e)

    async def _execute_batch_capabilities(
        self, batch: RequestBatch
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute capabilities for a batch using intelligent grouping.

        Strategy:
        1. Get dependency-ordered list of all needed capabilities
        2. For each capability:
           a. Filter texts that need this capability
           b. Check if they already have results from dependencies
           c. Execute capability for filtered texts
           d. Merge results

        This minimizes redundant processing and maximizes batching efficiency.
        """
        # Get all unique capabilities needed
        all_caps = batch.get_all_capabilities()

        # Resolve execution order
        execution_order = self.registry.resolve_dependencies(list(all_caps))

        console.print(f"[cyan]Execution order: {' → '.join(execution_order)}[/cyan]")

        # Initialize results
        all_results: Dict[str, Dict[str, Any]] = {text: {"text": text} for text in batch.texts}

        # Context for dependent capabilities
        context: Dict[str, Dict[str, Any]] = {}

        # Execute capabilities in order
        processor = self.processor_pool.get_processor()

        for cap_name in execution_order:
            # Get texts that need this capability
            texts_needing_cap = batch.get_texts_for_capability(cap_name)

            if not texts_needing_cap:
                continue

            console.print(
                f"[cyan]Executing {cap_name} for {len(texts_needing_cap)} texts[/cyan]"
            )

            capability = self.registry.get(cap_name)

            # Execute capability
            if cap_name == "classification":
                cap_results = await asyncio.to_thread(
                    self._execute_classification, processor, texts_needing_cap
                )

                # Build context for dependent capabilities
                if any(
                    cap in ["stem_recommendations", "stem_polarity", "stem_trend"]
                    for cap in all_caps
                ):
                    context = self._build_classification_context(
                        texts_needing_cap, cap_results, batch
                    )
            else:
                cap_results = await asyncio.to_thread(
                    self._execute_capability, processor, capability, texts_needing_cap, context
                )

            # Merge results
            self._merge_capability_results(all_results, cap_results, capability)

            console.print(f"[green]✓ {cap_name} complete[/green]")

        return all_results

    def _execute_classification(self, processor, texts: List[str]) -> Dict[str, Any]:
        """Execute classification capability."""
        return processor.classify_hierarchical(texts=texts)

    def _execute_capability(
        self, processor, capability, texts: List[str], context: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a single capability."""
        # Prepare prompts
        prompts = capability.prepare_batch(texts, context)

        # Execute with processor
        prompt_results = processor.classify_with_custom_schema(
            texts=prompts,
            prompt_fn=lambda x: x,
            schema=capability.schema,
        )

        # Post-process
        processed_results = capability.post_process(prompt_results, context)

        # Remap if needed
        if len(prompts) == len(texts):
            text_to_prompt = {text: prompt for text, prompt in zip(texts, prompts)}
            return {
                text: processed_results.get(prompt) for text, prompt in text_to_prompt.items()
            }
        else:
            return processed_results

    def _build_classification_context(
        self, texts: List[str], classification_results: Dict[str, Any], batch: RequestBatch
    ) -> Dict[str, Dict[str, Any]]:
        """Build context from classification results."""

        from classifier.capabilities.orchestrator import (
            get_leaf_paths_set,
            get_root_prefix,
            identify_complete_stems,
        )

        processor = self.processor_pool.get_processor()
        hierarchy = processor.topic_hierarchy
        leaf_paths_set = get_leaf_paths_set(hierarchy)
        root_prefix = get_root_prefix(hierarchy, project_name=None)

        context = {"_hierarchy": hierarchy}

        for text in texts:
            if text in classification_results:
                result = classification_results[text]

                # Convert to dict
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = result

                classification_paths = result_dict.get("classification_paths", [])
                complete_stems = identify_complete_stems(
                    classification_paths, leaf_paths_set, root_prefix
                )

                context[text] = {"complete_stems": complete_stems}

        return context

    def _merge_capability_results(
        self, all_results: Dict[str, Dict[str, Any]], cap_results: Dict[str, Any], capability
    ):
        """Merge capability results into main results."""
        result_key = capability.get_result_key()

        for text in all_results.keys():
            if text in cap_results and cap_results[text] is not None:
                formatted_result = capability.format_for_export(cap_results[text])
                all_results[text][result_key] = formatted_result
            else:
                all_results[text][result_key] = capability.format_for_export(None)

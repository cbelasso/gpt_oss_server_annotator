"""
Request Manager with Dynamic Config Support

Handles request batching, scheduling, and optimal capability execution
with support for different configs per request.

Key Features:
- Dynamic config loading and caching
- Batches requests over a time window
- Groups requests by capability sets
- Executes capabilities in dependency order
- Minimizes redundant processing
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich.console import Console

from classifier.capabilities import CapabilityRegistry
from classifier.hierarchy import load_topic_hierarchy

console = Console()


class ProcessorPool:
    """Processor pool with dynamic config support."""

    def __init__(
        self,
        default_config_path: Optional[Path],
        gpu_list: List[int],
        policy,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 10240,
        batch_size: int = 25,
        backend: str = "local",
        server_urls: Optional[List[str]] = None,
        max_concurrent: int = 5,
    ):
        from classifier import ClassificationProcessor

        self.default_config_path = default_config_path
        self.gpu_list = gpu_list
        self.batch_size = batch_size
        self.backend = backend

        # Config cache: path -> hierarchy
        self._config_cache: Dict[str, Any] = {}

        # Load default config if provided
        if default_config_path:
            console.print(f"[cyan]Loading default config: {default_config_path}[/cyan]")
            hierarchy = load_topic_hierarchy(default_config_path)
            if hierarchy:
                self._config_cache[str(default_config_path)] = hierarchy
                console.print("[green]✓ Default config loaded[/green]")
            else:
                console.print("[yellow]⚠ Failed to load default config[/yellow]")

        # Initialize processor WITHOUT a fixed config
        # We'll pass the hierarchy dynamically per request
        console.print(f"[cyan]Initializing processor ({backend} backend)...[/cyan]")

        # Create a dummy minimal config for processor initialization
        # The processor needs SOME config to initialize, but we'll override per request
        dummy_config = default_config_path or Path("/tmp/dummy_config.json")
        if not default_config_path:
            # Create a minimal dummy config file
            import json

            dummy_data = {"name": "ROOT", "description": "Dummy root", "children": []}
            dummy_config.parent.mkdir(parents=True, exist_ok=True)
            with open(dummy_config, "w") as f:
                json.dump(dummy_data, f)

        self.processor = ClassificationProcessor(
            config_path=dummy_config,
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

    def get_config(self, config_path: Optional[Path]) -> Optional[Any]:
        """
        Get hierarchy for a config path, loading and caching if needed.

        Args:
            config_path: Path to config file, or None

        Returns:
            Hierarchy dict or None
        """
        if config_path is None:
            return None

        config_key = str(config_path)

        # Check cache
        if config_key in self._config_cache:
            return self._config_cache[config_key]

        # Load and cache
        console.print(f"[cyan]Loading config: {config_path}[/cyan]")
        hierarchy = load_topic_hierarchy(config_path)

        if hierarchy:
            self._config_cache[config_key] = hierarchy
            console.print(
                f"[green]✓ Config loaded and cached ({len(self._config_cache)} total)[/green]"
            )
            return hierarchy
        else:
            console.print(f"[red]✗ Failed to load config: {config_path}[/red]")
            return None

    def get_config_cache_size(self) -> int:
        """Get number of cached configs."""
        return len(self._config_cache)

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
    A batch of texts with their requested capabilities and configs.

    Tracks which texts need which capabilities and which config to use.
    """

    def __init__(self):
        self.texts: List[str] = []
        self.text_to_capabilities: Dict[str, List[str]] = {}
        self.text_to_project: Dict[str, Optional[str]] = {}
        self.text_to_config: Dict[str, Optional[Path]] = {}
        self.request_ids: List[str] = []
        self.futures: List[tuple] = []

    def add_request(
        self,
        request_id: str,
        texts: List[str],
        capabilities: List[str],
        config_path: Optional[Path] = None,
        project_name: Optional[str] = None,
    ) -> asyncio.Future:
        """Add a request to the batch."""
        future = asyncio.Future()

        for text in texts:
            if text not in self.texts:
                self.texts.append(text)

            self.text_to_capabilities[text] = capabilities
            self.text_to_project[text] = project_name
            self.text_to_config[text] = config_path

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

    def get_unique_configs(self) -> Set[Optional[Path]]:
        """Get set of unique config paths in this batch."""
        return set(self.text_to_config.values())

    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return len(self.texts) == 0


class RequestBatcher:
    """
    Intelligent request batcher with capability-aware processing and dynamic configs.

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
            processor_pool: ProcessorPool instance with dynamic config support
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
        config_path: Optional[Path] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process a classification request with optional config override.

        Args:
            texts: List of texts to classify
            capabilities: List of capabilities to execute
            config_path: Optional config file path (overrides default)
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
                config_path=config_path,
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

        Groups texts by capability requirements and config, executes capabilities
        in optimal order to maximize GPU utilization.
        """
        if self._current_batch.is_empty():
            return

        batch = self._current_batch
        self._current_batch = RequestBatch()  # Start new batch

        unique_configs = batch.get_unique_configs()
        console.print(
            f"[cyan]Processing batch: {len(batch.texts)} texts, "
            f"{len(batch.get_all_capabilities())} capabilities, "
            f"{len(unique_configs)} config(s)[/cyan]"
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
        Execute capabilities for a batch with dynamic config support.

        Strategy:
        1. Group texts by config path
        2. For each config group:
           a. Load/get the config
           b. Execute capabilities for that group
        3. Merge all results
        """
        # Get all unique capabilities needed
        all_caps = batch.get_all_capabilities()

        # Resolve execution order
        execution_order = self.registry.resolve_dependencies(list(all_caps))

        console.print(f"[cyan]Execution order: {' → '.join(execution_order)}[/cyan]")

        # Initialize results
        all_results: Dict[str, Dict[str, Any]] = {text: {"text": text} for text in batch.texts}

        # Group texts by config
        config_groups: Dict[Optional[str], List[str]] = {}
        for text in batch.texts:
            config_path = batch.text_to_config[text]
            config_key = str(config_path) if config_path else None
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(text)

        # Process each config group
        for config_key, group_texts in config_groups.items():
            config_path = Path(config_key) if config_key else None

            console.print(
                f"[cyan]Processing {len(group_texts)} texts with config: "
                f"{config_path or 'None'}[/cyan]"
            )

            # Load config
            hierarchy = self.processor_pool.get_config(config_path)

            # Temporarily override processor's hierarchy for this group
            processor = self.processor_pool.get_processor()
            original_hierarchy = processor.topic_hierarchy
            if hierarchy:
                processor.topic_hierarchy = hierarchy

            # Context for dependent capabilities
            context: Dict[str, Dict[str, Any]] = {}

            # Execute capabilities in order for this group
            for cap_name in execution_order:
                # Get texts from this group that need this capability
                texts_needing_cap = [
                    text for text in group_texts if cap_name in batch.text_to_capabilities[text]
                ]

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
                        self._execute_capability,
                        processor,
                        capability,
                        texts_needing_cap,
                        context,
                    )

                # Merge results
                self._merge_capability_results(all_results, cap_results, capability)

                console.print(f"[green]✓ {cap_name} complete[/green]")

            # Restore original hierarchy
            processor.topic_hierarchy = original_hierarchy

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

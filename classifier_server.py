"""
Classification Server with Intelligent Request Batching

FastAPI server that keeps LLMs loaded and optimally batches requests
across different capability combinations for maximum GPU utilization.

Usage:
    python classifier_server.py --config topics.json --gpu-list 0,1,2,3
    
    # Then make requests:
    curl -X POST http://localhost:8000/classify \
         -H "Content-Type: application/json" \
         -d '{"texts": ["example text"], "capabilities": ["classification", "recommendations"]}'
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import signal
import time
from typing import Any, Dict, List, Optional

import click
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rich.console import Console
import uvicorn

from classifier.capabilities import (
    CapabilityRegistry,
    StemPolarityCapability,
    StemRecommendationsCapability,
    StemTrendCapability,
    create_default_registry,
)
from classifier.policies import (
    CompositePolicy,
    ConfidenceThresholdPolicy,
    DefaultPolicy,
    ExcerptRequiredPolicy,
)
from classifier_server_manager import ProcessorPool, RequestBatcher

console = Console()

SHUTDOWN_KEY = os.getenv("CLASSIFIER_SHUTDOWN_KEY", "toooo myyyy!!!!")


def initialize_state_from_env():
    """Initialize server state when running via `uvicorn classifier_server:app`."""
    config_path = os.getenv("CLASSIFIER_CONFIG")
    gpu_list = os.getenv("CLASSIFIER_GPU_LIST", "0,1,2,3")
    if not config_path:
        console.print("[yellow]⚠️ CLASSIFIER_CONFIG not set - skipping model init[/yellow]")
        return

    gpu_ids = [int(x.strip()) for x in gpu_list.split(",")]

    console.print("[cyan]Initializing server from environment...[/cyan]")
    console.print(f"  • Config: {config_path}")
    console.print(f"  • GPUs: {gpu_ids}")

    policy = DefaultPolicy()
    state.processor_pool = ProcessorPool(
        config_path=Path(config_path),
        gpu_list=gpu_ids,
        policy=policy,
        gpu_memory_utilization=0.95,
        max_model_len=10240,
        batch_size=25,
    )
    state.registry = create_default_registry()
    state.registry.register(StemRecommendationsCapability())
    state.registry.register(StemPolarityCapability())
    state.registry.register(StemTrendCapability())
    state.request_batcher = RequestBatcher(
        processor_pool=state.processor_pool,
        registry=state.registry,
        batch_timeout=0.1,
    )

    console.print("[green]✓ Processor ready (CLI mode)[/green]")


# ============================================================================
# Request/Response Models
# ============================================================================


class ClassificationRequest(BaseModel):
    """Request for text classification."""

    texts: List[str] = Field(..., description="List of texts to classify")
    capabilities: List[str] = Field(
        default=["classification"],
        description="Capabilities to execute (classification, recommendations, alerts, etc.)",
    )
    project_name: Optional[str] = Field(
        default=None, description="Optional project name for root prefix"
    )


class ClassificationResponse(BaseModel):
    """Response containing classification results."""

    results: List[Dict[str, Any]] = Field(..., description="Classification results per text")
    processing_time: float = Field(..., description="Processing time in seconds")
    batch_info: Dict[str, Any] = Field(..., description="Information about batch processing")


class HealthResponse(BaseModel):
    """Server health status."""

    status: str
    processor_loaded: bool
    gpu_count: int
    capabilities_available: List[str]
    requests_processed: int
    uptime_seconds: float


class ServerStats(BaseModel):
    """Server statistics."""

    total_requests: int
    total_texts_processed: int
    average_processing_time: float
    requests_by_capability: Dict[str, int]
    uptime_seconds: float


# ============================================================================
# Global Server State
# ============================================================================


class ServerState:
    """Global server state container."""

    def __init__(self):
        self.processor_pool: Optional[ProcessorPool] = None
        self.request_batcher: Optional[RequestBatcher] = None
        self.registry: Optional[CapabilityRegistry] = None
        self.start_time: float = time.time()
        self.stats = {
            "total_requests": 0,
            "total_texts": 0,
            "processing_times": [],
            "requests_by_capability": {},
        }


state = ServerState()


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - startup and shutdown."""
    console.print("[cyan]Starting Classification Server...[/cyan]")

    # Server will be initialized via CLI parameters
    # stored in app.state by the CLI command

    yield

    # Shutdown
    console.print("\n[cyan]Shutting down server...[/cyan]")
    if state.processor_pool:
        state.processor_pool.shutdown()
    console.print("[green]✓ Server shutdown complete[/green]")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Hierarchical Text Classification Server",
    description="High-performance text classification with intelligent request batching",
    version="1.0.0",
    lifespan=lifespan,
)

# Auto-initialize if running under Uvicorn CLI
if not state.processor_pool:
    initialize_state_from_env()


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Hierarchical Text Classification Server",
        "version": "1.0.0",
        "endpoints": {
            "classify": "POST /classify - Classify texts",
            "health": "GET /health - Health check",
            "stats": "GET /stats - Server statistics",
            "capabilities": "GET /capabilities - List available capabilities",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not state.processor_pool or not state.registry:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return HealthResponse(
        status="healthy",
        processor_loaded=state.processor_pool.is_ready(),
        gpu_count=len(state.processor_pool.gpu_list),
        capabilities_available=state.registry.list_capabilities(),
        requests_processed=state.stats["total_requests"],
        uptime_seconds=time.time() - state.start_time,
    )


@app.get("/stats", response_model=ServerStats)
async def get_stats():
    """Get server statistics."""
    if not state.processor_pool:
        raise HTTPException(status_code=503, detail="Server not initialized")

    avg_time = (
        sum(state.stats["processing_times"]) / len(state.stats["processing_times"])
        if state.stats["processing_times"]
        else 0.0
    )

    return ServerStats(
        total_requests=state.stats["total_requests"],
        total_texts_processed=state.stats["total_texts"],
        average_processing_time=avg_time,
        requests_by_capability=state.stats["requests_by_capability"],
        uptime_seconds=time.time() - state.start_time,
    )


@app.get("/capabilities", response_model=Dict[str, List[str]])
async def list_capabilities():
    """List all available capabilities."""
    if not state.registry:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return {"capabilities": state.registry.list_capabilities()}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_texts(request: ClassificationRequest):
    """
    Classify texts using specified capabilities.

    This endpoint intelligently batches requests to maximize GPU utilization
    and minimize redundant processing.
    """
    if not state.request_batcher or not state.registry:
        raise HTTPException(status_code=503, detail="Server not initialized")

    start_time = time.time()

    # Validate capabilities
    errors = state.registry.validate_capabilities(request.capabilities)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Update stats
    state.stats["total_requests"] += 1
    state.stats["total_texts"] += len(request.texts)
    for cap in request.capabilities:
        state.stats["requests_by_capability"][cap] = (
            state.stats["requests_by_capability"].get(cap, 0) + 1
        )

    try:
        # Process through batcher
        results = await state.request_batcher.process_request(
            texts=request.texts,
            capabilities=request.capabilities,
            project_name=request.project_name,
        )

        processing_time = time.time() - start_time
        state.stats["processing_times"].append(processing_time)

        # Keep only last 1000 processing times
        if len(state.stats["processing_times"]) > 1000:
            state.stats["processing_times"] = state.stats["processing_times"][-1000:]

        return ClassificationResponse(
            results=[results[text] for text in request.texts],
            processing_time=processing_time,
            batch_info={
                "text_count": len(request.texts),
                "capabilities_executed": request.capabilities,
            },
        )

    except Exception as e:
        console.print(f"[red]Error processing request: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
async def shutdown(request: Request):
    """Gracefully shut down regardless of launch mode."""
    provided_key = request.headers.get("X-Shutdown-Key")
    if provided_key != SHUTDOWN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: invalid shutdown key")

    console.print("[yellow]Shutdown requested via API[/yellow]")
    if state.processor_pool:
        state.processor_pool.shutdown()

    # Try sending SIGINT (CLI-friendly)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        console.print("[cyan]Sent SIGINT to self (CLI mode)[/cyan]")
    except Exception as e:
        console.print(f"[red]SIGINT failed: {e}[/red]")
        # Fallback for embedded or programmatic mode
        loop = asyncio.get_event_loop()
        loop.call_later(0.1, loop.stop)
        console.print("[cyan]Stopped event loop directly (programmatic mode)[/cyan]")

    return {"message": "Server shutting down..."}


# ============================================================================
# CLI Entry Point
# ============================================================================


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to topic hierarchy JSON file",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind to",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to bind to",
)
@click.option(
    "--gpu-list",
    type=str,
    default="0,1,2,3,4,5,6,7",
    help="Comma-separated GPU IDs",
)
@click.option(
    "--gpu-memory",
    type=float,
    default=0.95,
    help="GPU memory utilization (0-1)",
)
@click.option(
    "--max-length",
    type=int,
    default=10240,
    help="Maximum model context length",
)
@click.option(
    "--batch-size",
    type=int,
    default=25,
    help="Batch size for processing",
)
@click.option(
    "--batch-timeout",
    type=float,
    default=0.1,
    help="Max seconds to wait before processing batch",
)
@click.option(
    "--min-confidence",
    type=click.IntRange(1, 5),
    default=None,
    help="Minimum confidence threshold",
)
@click.option(
    "--require-excerpt",
    is_flag=True,
    help="Require non-empty excerpts",
)
@click.option(
    "--max-stem-definitions",
    type=int,
    default=None,
    help="Max definitions for stem analysis",
)
def serve(
    config: Path,
    host: str,
    port: int,
    gpu_list: str,
    gpu_memory: float,
    max_length: int,
    batch_size: int,
    batch_timeout: float,
    min_confidence: int,
    require_excerpt: bool,
    max_stem_definitions: int,
):
    """
    Start the classification server.

    The server keeps LLMs loaded in memory and intelligently batches
    requests for optimal GPU utilization.
    """
    console.print("[cyan]Initializing Classification Server...[/cyan]")

    # Parse GPU list
    gpu_ids = [int(x.strip()) for x in gpu_list.split(",")]
    console.print(f"  • GPUs: {gpu_ids}")
    console.print(f"  • Config: {config}")
    console.print(f"  • Batch size: {batch_size}")
    console.print(f"  • Batch timeout: {batch_timeout}s")

    # Create policy
    policies = []
    if min_confidence:
        policies.append(ConfidenceThresholdPolicy(min_confidence=min_confidence))
    if require_excerpt:
        policies.append(ExcerptRequiredPolicy())
    policy = CompositePolicy(*policies) if policies else DefaultPolicy()

    # Initialize processor pool
    console.print("\n[cyan]Loading models...[/cyan]")
    state.processor_pool = ProcessorPool(
        config_path=config,
        gpu_list=gpu_ids,
        policy=policy,
        gpu_memory_utilization=gpu_memory,
        max_model_len=max_length,
        batch_size=batch_size,
    )

    # Create capability registry
    state.registry = create_default_registry()
    state.registry.register(
        StemRecommendationsCapability(max_stem_definitions=max_stem_definitions)
    )
    state.registry.register(StemPolarityCapability(max_stem_definitions=max_stem_definitions))
    state.registry.register(StemTrendCapability(max_stem_definitions=max_stem_definitions))

    console.print(
        f"[green]✓ Available capabilities: {state.registry.list_capabilities()}[/green]"
    )

    # Initialize request batcher
    state.request_batcher = RequestBatcher(
        processor_pool=state.processor_pool,
        registry=state.registry,
        batch_timeout=batch_timeout,
    )

    console.print("\n[green]✓ Server ready![/green]")
    console.print(f"[cyan]Listening on {host}:{port}[/cyan]\n")

    # Start server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()

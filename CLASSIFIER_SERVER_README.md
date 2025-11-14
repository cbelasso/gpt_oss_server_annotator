# Classification Server

High-performance text classification server with intelligent request batching and persistent LLM instances.

## Features

- 🚀 **Persistent LLMs**: Models stay loaded in memory, no reinitialization overhead
- 🎯 **Intelligent Batching**: Automatically batches requests over a time window for optimal GPU utilization
- 🔄 **Multi-Capability Support**: Execute different capability combinations per request
- 📊 **Smart Execution**: Groups requests by capabilities and executes them efficiently
- 🔌 **RESTful API**: Simple HTTP interface with FastAPI
- 📈 **Built-in Monitoring**: Health checks, statistics, and performance metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Requests                          │
│  (Different texts, different capability combinations)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Request Batcher                            │
│  • Collects requests over batch_timeout window              │
│  • Groups by capability requirements                         │
│  • Resolves capability dependencies                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                Capability Orchestrator                       │
│  • Executes capabilities in dependency order                │
│  • Batches texts that need same capability together          │
│  • Minimizes redundant processing                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Processor Pool                              │
│  • Persistent FlexibleSchemaProcessor instance              │
│  • LLMs stay loaded across requests                          │
│  • Thread-safe access                                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install dependencies
pip install fastapi uvicorn aiohttp click rich pandas

# Your existing classifier package should already be installed
```

## Quick Start

### 1. Start the Server

```bash
python classifier_server.py \
    --config topics.json \
    --gpu-list 0,1,2,3 \
    --port 8000 \
    --batch-timeout 0.1
```

**Options:**
- `--config`: Path to topic hierarchy JSON (required)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--gpu-list`: Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
- `--gpu-memory`: GPU memory utilization 0-1 (default: 0.95)
- `--max-length`: Max model context length (default: 10240)
- `--batch-size`: Batch size for processing (default: 25)
- `--batch-timeout`: Max seconds to wait before processing batch (default: 0.1)
- `--min-confidence`: Minimum confidence threshold 1-5
- `--require-excerpt`: Require non-empty excerpts
- `--max-stem-definitions`: Max definitions for stem analysis

### 2. Check Server Health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "processor_loaded": true,
  "gpu_count": 4,
  "capabilities_available": [
    "classification",
    "recommendations",
    "alerts",
    "stem_recommendations",
    "stem_polarity"
  ],
  "requests_processed": 0,
  "uptime_seconds": 5.2
}
```

### 3. Make Classification Requests

#### Basic Classification

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The instructor was excellent and very knowledgeable."],
    "capabilities": ["classification"]
  }'
```

#### Multiple Capabilities

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "We should add more hands-on exercises to the training.",
      "The course was great but the room was too cold."
    ],
    "capabilities": ["classification", "recommendations", "alerts"]
  }'
```

#### Batch Request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Text 1...",
      "Text 2...",
      "Text 3..."
    ],
    "capabilities": ["classification", "recommendations"]
  }'
```

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "processor_loaded": true,
  "gpu_count": 4,
  "capabilities_available": ["classification", "recommendations", "alerts"],
  "requests_processed": 42,
  "uptime_seconds": 120.5
}
```

### `GET /capabilities`
List available capabilities.

**Response:**
```json
{
  "capabilities": [
    "classification",
    "recommendations",
    "alerts",
    "stem_recommendations",
    "stem_polarity"
  ]
}
```

### `GET /stats`
Get server statistics.

**Response:**
```json
{
  "total_requests": 42,
  "total_texts_processed": 156,
  "average_processing_time": 0.85,
  "requests_by_capability": {
    "classification": 42,
    "recommendations": 28,
    "alerts": 15
  },
  "uptime_seconds": 300.2
}
```

### `POST /classify`
Classify texts with specified capabilities.

**Request Body:**
```json
{
  "texts": ["Text to classify", "Another text"],
  "capabilities": ["classification", "recommendations"],
  "project_name": "optional_project_name"
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Text to classify",
      "classification_result": {
        "classification_paths": ["Root>Topic>Subtopic"],
        "node_results": {...}
      },
      "recommendations": [...]
    }
  ],
  "processing_time": 0.45,
  "batch_info": {
    "text_count": 2,
    "capabilities_executed": ["classification", "recommendations"]
  }
}
```

### `POST /shutdown`
Gracefully shutdown the server.

## Python Client Example

```python
import asyncio
import aiohttp

async def classify_text():
    async with aiohttp.ClientSession() as session:
        payload = {
            "texts": ["The training was excellent!"],
            "capabilities": ["classification", "recommendations"]
        }
        
        async with session.post(
            "http://localhost:8000/classify",
            json=payload
        ) as response:
            result = await response.json()
            print(result)

asyncio.run(classify_text())
```

Or use the provided client:

```python
from classifier_client_example import ClassificationClient

async def main():
    client = ClassificationClient("http://localhost:8000")
    
    result = await client.classify(
        texts=["Example text"],
        capabilities=["classification", "recommendations"]
    )
    
    print(result)

asyncio.run(main())
```

## Intelligent Batching Explained

The server uses intelligent batching to maximize GPU utilization:

### Example Scenario

Imagine 4 concurrent requests arrive:
1. Text A: wants `["classification"]`
2. Text B: wants `["classification", "recommendations"]`
3. Text C: wants `["classification", "alerts"]`
4. Text D: wants `["classification", "recommendations", "alerts"]`

### Traditional Approach (Inefficient)
Execute each request separately:
- Request 1: Run classification for Text A
- Request 2: Run classification + recommendations for Text B
- Request 3: Run classification + alerts for Text C
- Request 4: Run classification + recommendations + alerts for Text D

**Result**: 10 total capability executions (lots of redundant work!)

### Our Intelligent Batching (Efficient)
1. Batch all requests together
2. Run classification ONCE for all texts [A, B, C, D]
3. Run recommendations ONCE for texts that need it [B, D]
4. Run alerts ONCE for texts that need it [C, D]

**Result**: 3 total capability executions (optimal GPU utilization!)

## Performance Tips

### Batch Timeout
- **Lower values** (0.05-0.1s): More responsive, less batching
- **Higher values** (0.2-0.5s): Better batching, slightly higher latency
- **Recommendation**: Start with 0.1s and adjust based on your traffic pattern

### GPU Configuration
- Use as many GPUs as you have available
- Set `--gpu-memory 0.95` for maximum utilization
- Adjust `--batch-size` based on your GPU memory and text length

### Request Patterns
- **High traffic**: Server automatically batches for you
- **Low traffic**: Set lower `--batch-timeout` for quick responses
- **Mixed capabilities**: Server handles this optimally

## Monitoring

### View Real-time Stats

```bash
# Watch stats every 2 seconds
watch -n 2 'curl -s http://localhost:8000/stats | jq'
```

### Load Testing

```python
import asyncio
import aiohttp
from classifier_client_example import ClassificationClient

async def load_test():
    client = ClassificationClient()
    
    # 100 concurrent requests
    tasks = [
        client.classify(
            texts=[f"Test text {i}"],
            capabilities=["classification"]
        )
        for i in range(100)
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} requests")

asyncio.run(load_test())
```

## Troubleshooting

### Server won't start
- Check if port is already in use: `lsof -i :8000`
- Verify config file exists and is valid JSON
- Check GPU availability: `nvidia-smi`

### Slow responses
- Increase `--batch-timeout` to allow more batching
- Check GPU utilization: `nvidia-smi`
- Reduce `--max-length` if texts are very long
- Increase `--batch-size` if GPU memory allows

### Out of memory errors
- Reduce `--gpu-memory` (try 0.85 or 0.8)
- Reduce `--batch-size`
- Reduce `--max-length`
- Use fewer GPUs with higher memory per GPU

## Shutdown

```bash
# Graceful shutdown via API
curl -X POST http://localhost:8000/shutdown

# Or use Ctrl+C in the terminal
```

## Example Use Cases

### 1. Real-time Feedback Analysis
```python
# Process user feedback as it arrives
await client.classify(
    texts=[user_feedback],
    capabilities=["classification", "recommendations", "alerts"]
)
```

### 2. Batch Processing
```python
# Process large datasets efficiently
for chunk in chunks(large_dataset, 100):
    results = await client.classify(
        texts=chunk,
        capabilities=["classification"]
    )
```

### 3. Different Analysis Types
```python
# Some texts need full analysis
full_analysis = await client.classify(
    texts=priority_texts,
    capabilities=["classification", "recommendations", "alerts", 
                  "stem_recommendations", "stem_polarity"]
)

# Others just need classification
quick_classification = await client.classify(
    texts=regular_texts,
    capabilities=["classification"]
)
```

## Advanced Configuration

### Custom Policies

Edit `classifier_server.py` to add custom policies:

```python
from classifier.policies import CustomPolicy

policy = CompositePolicy(
    ConfidenceThresholdPolicy(min_confidence=4),
    ExcerptRequiredPolicy(),
    CustomPolicy()  # Your custom policy
)
```

### Custom Capabilities

Register custom capabilities in the server:

```python
from classifier.capabilities import Capability

class MyCustomCapability(Capability):
    # ... implementation ...

# In server initialization
state.registry.register(MyCustomCapability())
```

## License

Same as your main classifier package.

# Classification Server - Quick Start Guide

Get your classification server up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPUs
- Your existing classifier package installed

## Installation

```bash
# Install server dependencies
pip install -r server_requirements.txt
```

## Start the Server

### Basic Start

```bash
python classifier_server.py --config topics.json --gpu-list 0,1,2,3
```

### With Custom Settings

```bash
python classifier_server.py \
    --config topics.json \
    --gpu-list 0,1,2,3 \
    --port 8000 \
    --batch-size 25 \
    --batch-timeout 0.1 \
    --gpu-memory 0.95 \
    --min-confidence 3
```

The server will:
1. Load your LLMs onto the specified GPUs (this takes a minute)
2. Start listening for requests
3. Keep models loaded for fast processing

You should see:
```
Initializing Classification Server...
  • GPUs: [0, 1, 2, 3]
  • Config: topics.json
  ...
✓ Server ready!
Listening on 0.0.0.0:8000
```

## Test the Server

### Method 1: Shell Script (Easiest)

```bash
chmod +x test_server.sh
./test_server.sh
```

### Method 2: Python Client

```bash
python classifier_client_example.py
```

### Method 3: Direct curl

```bash
# Health check
curl http://localhost:8000/health

# Classification
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The training was excellent!"],
    "capabilities": ["classification"]
  }'
```

## Common Use Cases

### 1. Real-time Classification

```python
from classifier_client_example import ClassificationClient
import asyncio

async def classify_feedback(feedback_text):
    client = ClassificationClient("http://localhost:8000")
    
    result = await client.classify(
        texts=[feedback_text],
        capabilities=["classification", "recommendations", "alerts"]
    )
    
    return result

# Use it
feedback = "The instructor needs to speak more slowly."
result = asyncio.run(classify_feedback(feedback))
print(result)
```

### 2. Batch Processing

```python
async def process_batch(texts):
    client = ClassificationClient("http://localhost:8000")
    
    # Process 100 texts at once
    result = await client.classify(
        texts=texts,
        capabilities=["classification"]
    )
    
    return result

# Process your data
texts = ["text 1", "text 2", ..., "text 100"]
results = asyncio.run(process_batch(texts))
```

### 3. Concurrent Requests

```python
async def process_concurrent():
    client = ClassificationClient("http://localhost:8000")
    
    # Multiple requests at once - server batches them!
    tasks = [
        client.classify(texts=[text], capabilities=["classification"])
        for text in my_texts
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

## Performance Testing

Run the stress test to see batching in action:

```bash
python stress_test.py
```

This will show you:
- Batching efficiency comparison
- Throughput metrics
- Latency percentiles
- Capability performance comparison

## Monitoring

### View Real-time Stats

```bash
# In a separate terminal
watch -n 2 'curl -s http://localhost:8000/stats | python -m json.tool'
```

### Check Health

```bash
curl http://localhost:8000/health | python -m json.tool
```

## Key Configuration Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--batch-timeout` | Time to wait before processing batch | 0.1s for balanced performance |
| `--batch-size` | Texts per GPU batch | 25 for most use cases |
| `--gpu-memory` | GPU memory utilization | 0.95 for maximum throughput |
| `--max-length` | Max context length | 10240 (adjust based on text length) |

## Troubleshooting

### "Server not responding"
- Check if server is running: `curl http://localhost:8000/health`
- Check port is not in use: `lsof -i :8000`

### Out of Memory
- Reduce `--gpu-memory` to 0.85 or 0.8
- Reduce `--batch-size` to 15 or 20
- Reduce `--max-length` to 8192 or 4096

### Slow Performance
- Increase `--batch-timeout` to allow more batching
- Check GPU utilization: `nvidia-smi`
- Increase `--batch-size` if GPU memory allows

## Shutdown

```bash
# Graceful shutdown
curl -X POST http://localhost:8000/shutdown

# Or press Ctrl+C in the server terminal
```

## Integration Examples

### With Web App

```python
# app.py
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.post("/analyze-feedback")
async def analyze(feedback: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/classify",
            json={
                "texts": [feedback],
                "capabilities": ["classification", "recommendations"]
            }
        )
        return response.json()
```

### With Background Worker

```python
# worker.py
import asyncio
from classifier_client_example import ClassificationClient

async def process_queue():
    client = ClassificationClient("http://localhost:8000")
    
    while True:
        # Get items from your queue
        items = await get_queue_items()
        
        if items:
            results = await client.classify(
                texts=[item.text for item in items],
                capabilities=["classification"]
            )
            
            # Save results
            await save_results(results)
        
        await asyncio.sleep(1)

asyncio.run(process_queue())
```

## Next Steps

1. **Production Deployment**: See CLASSIFIER_SERVER_README.md for detailed deployment guide
2. **Custom Capabilities**: Learn how to add your own capabilities
3. **Advanced Configuration**: Explore custom policies and orchestration

## Need Help?

- Read the full documentation: `CLASSIFIER_SERVER_README.md`
- Check the examples: `classifier_client_example.py`
- Run tests: `stress_test.py`

Happy classifying! 🚀

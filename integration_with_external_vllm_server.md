Integration with External VLLM Server
Date: November 15, 2025
Purpose: Enable the hierarchical classifier to work with an external VLLM server instead of loading models directly

📋 Table of Contents

Overview
Architecture
Changes Made
Why This Design Works
Usage
Testing
Troubleshooting
Future Improvements


🎯 Overview
Problem Statement
Originally, our classifier used FlexibleSchemaProcessor which:

Loaded LLM models directly into GPU memory
Managed model lifecycle within the Python process
Used vLLM's Python API for inference

Challenge: A colleague built a separate VLLM server that:

Already has models loaded and ready
Exposes an OpenAI-compatible HTTP API
Handles GPU management independently

Goal: Make our classifier work with the external server without rewriting core logic.
Solution Summary
We created an adapter pattern that allows our classifier to communicate with the external VLLM server via HTTP while maintaining the same interface as the local processor.
Key Principle: The classifier doesn't need to know HOW it talks to the LLM, just that it can send prompts and get structured results back.

🏗️ Architecture
Two-Server Design
┌─────────────────────────────────────────────────────────────┐
│  Classification Server (Port 9000)                          │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Application Logic                                   │   │
│  │  • HierarchicalClassifier                           │   │
│  │  • Capabilities (classification, recommendations,   │   │
│  │    alerts, stem analysis, trends)                   │   │
│  │  • Request batching & orchestration                 │   │
│  │  • Policy enforcement                               │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │                                      │
│                       │ Interface: process_with_schema()    │
│                       │                                      │
│         ┌─────────────┴──────────────┐                      │
│         │                            │                      │
│  ┌──────▼──────────┐   ┌────────────▼─────────────┐       │
│  │ Flexible        │   │ VLLMServer               │       │
│  │ SchemaProcessor │   │ Processor (NEW!)         │       │
│  │                 │   │                          │       │
│  │ Direct vLLM     │   │ HTTP Client              │       │
│  │ Python API      │   │ AsyncOpenAI              │       │
│  └─────────────────┘   └────────────┬─────────────┘       │
│                                      │                      │
└──────────────────────────────────────┼──────────────────────┘
                                       │
                                       │ HTTP POST
                                       │ /v1/chat/completions
                                       │
┌──────────────────────────────────────▼──────────────────────┐
│  VLLM Server (Port 8054)                                    │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  • Model Loading (GPT-OSS 120B, etc.)                       │
│  • GPU Memory Management                                    │
│  • Raw LLM Inference                                        │
│  • OpenAI-Compatible API                                    │
│  • Structured Output (via text_format parameter)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
Division of Responsibilities
ComponentResponsibilitiesOwnerVLLM ServerModel loading, GPU management, raw inferenceColleague's codeClassification ServerBusiness logic, hierarchical traversal, capabilities, prompt engineeringOur codeVLLMServerProcessorHTTP communication adapterOur code (NEW)

🔧 Changes Made
1. New File: lib/llm_parallelization/src/llm_parallelization/vllm_server_processor.py
Purpose: Adapter that translates between our classifier's interface and the external VLLM server's HTTP API.
Key Features:

Implements same interface as FlexibleSchemaProcessor
Uses AsyncOpenAI client for HTTP communication
Handles async-to-sync bridging (since our classifier is synchronous)
Manages connection lifecycle per batch
Provides concurrency control via semaphore

Methods:
pythonclass VLLMServerProcessor:
    def process_with_schema(prompts, schema, ...)
        # Send prompts to server, get structured results
    
    def parse_results_with_schema(schema, ...)
        # Return already-parsed results
    
    def terminate()
        # Cleanup (no-op for server-based)
Why it works: Our classifier calls these methods, not caring whether they hit local vLLM or a remote server.
2. Modified: classifier/processor.py
Changes:

Added import for VLLMServerProcessor
Added backend selection parameters to __init__:

backend: "local" or "vllm-server"
vllm_server_url: URL of external server
max_concurrent: Concurrency limit for server requests


Added conditional initialization logic:

pythonif backend == "local":
    self.llm_processor = FlexibleSchemaProcessor(...)
elif backend == "vllm-server":
    self.llm_processor = VLLMServerProcessor(...)
Impact: Minimal - just added parameters and conditional initialization. All downstream code unchanged.
3. Modified: classifier_server.py
Changes:

Added CLI options:

--backend: Choice between "local" and "vllm-server"
--vllm-server-url: Server URL (default: http://localhost:8054/v1)
--max-concurrent: Max concurrent requests (default: 5)


Pass these parameters to ProcessorPool

Example usage:
bash# Local mode (original behavior)
python classifier_server.py --config topics.json --gpu-list 0,1,2,3

# Server mode (new behavior)
python classifier_server.py --config topics.json \
    --backend vllm-server \
    --vllm-server-url http://localhost:8054/v1
4. Modified: classifier_server_manager.py
Changes:

Added same parameters to ProcessorPool.__init__
Pass them through to ClassificationProcessor

Impact: Just parameter pass-through, no logic changes.
5. Modified: batch_classify.py (Optional - for CLI usage)
Same changes as classifier_server.py to enable server backend for batch processing.

✨ Why This Design Works
1. Interface Abstraction
Both processors implement the same interface:
python# Classifier doesn't care which processor it uses
self.processor.process_with_schema(prompts, schema)
results = self.processor.parse_results_with_schema(schema)
```

This is the **Dependency Inversion Principle** - depend on abstractions, not concrete implementations.

### 2. **Separation of Concerns**
```
Classification Logic (What to classify)
    ↓
Processor Interface (How to communicate)
    ↓
Implementation (Local vLLM or Remote server)
Each layer only knows about the layer below it, not the implementation details.
3. Minimal Changes
Because we had good separation already:

HierarchicalClassifier: No changes
Capabilities: No changes
Orchestrator: No changes
Prompts: No changes
Policies: No changes

Only the processor initialization changed - everything else is untouched.
4. Backward Compatibility
Default behavior is preserved:
bash# Still works exactly as before
python classifier_server.py --config topics.json
New functionality is opt-in via --backend flag.

🚀 Usage
Setup
1. Start the VLLM Server (Colleague's Code)
bash# Using their script
python vllm_server_gpt_oss.py start

# Wait for it to be ready (check logs)
tail -f vllm_server_stdout.log

# Verify it's running
curl http://localhost:8054/v1/models
2. Start the Classification Server
bashpython classifier_server.py \
    --config /path/to/topics.json \
    --backend vllm-server \
    --vllm-server-url http://localhost:8054/v1 \
    --max-concurrent 10 \
    --port 9000
Making Requests
bash# Single capability
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The training was excellent"],
    "capabilities": ["classification"]
  }'

# Multiple capabilities
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The instructor needs to improve their teaching style"],
    "capabilities": ["classification", "recommendations", "stem_polarity"]
  }'

# Pretty output with jq
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{"texts": ["..."], "capabilities": ["classification"]}' \
  | jq '.' > results.json
Configuration Options
ParameterDescriptionDefaultNotes--backendProcessor typelocallocal or vllm-server--vllm-server-urlServer URLhttp://localhost:8054/v1Only for vllm-server backend--max-concurrentMax parallel requests5Only for vllm-server backend--gpu-listGPU IDs0,1,2,3,4,5,6,7Only for local backend

🧪 Testing
Quick Smoke Test
bash# Test classification
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The instructor was great"],
    "capabilities": ["classification"]
  }' | jq '.results[0].classification_result.classification_paths'

# Expected: Array of classification paths
Comprehensive Test Script
See test_classifier.py in the project root for a comprehensive test suite that:

Tests multiple capabilities
Saves results to timestamped files
Provides summary statistics
Validates output structure

bashpython test_classifier.py
Performance Comparison
bash# Local mode
time python batch_classify.py \
    --input-file test_data.csv \
    --backend local \
    --enable-classification

# Server mode
time python batch_classify.py \
    --input-file test_data.csv \
    --backend vllm-server \
    --enable-classification
```

---

## 🔧 Troubleshooting

### Issue: "Connection error" when calling server

**Symptoms:**
```
Batch processing error: Inference failed for prompt 0: Connection error.
Causes & Solutions:

VLLM server not running

bash   # Check if server is up
   curl http://localhost:8054/v1/models
   
   # If not, start it
   python vllm_server_gpt_oss.py start

Wrong URL format

bash   # ❌ Wrong - don't use 0.0.0.0 for client connections
   --vllm-server-url http://0.0.0.0:8054/v1
   
   # ✅ Correct - use localhost or 127.0.0.1
   --vllm-server-url http://localhost:8054/v1

Port already in use

bash   # Find what's using the port
   lsof -i :9000
   
   # Use a different port
   --port 9001
```

### Issue: First capability works, second fails

**Symptoms:**
```
✓ classification complete
Batch processing error: 'VLLMServerProcessor' object has no attribute...
Solution: Ensure you have the complete vllm_server_processor.py file with all three methods:

process_with_schema()
parse_results_with_schema()
terminate()

Issue: Slow performance
Tuning options:

Increase concurrency

bash   --max-concurrent 20  # More parallel requests

Reduce batch timeout (for server mode)

bash   --batch-timeout 0.05  # Process batches faster
```

3. **Check VLLM server settings**
   - Ensure adequate GPU memory allocation
   - Check server logs for bottlenecks

### Issue: Model name mismatch

**Symptoms:**
```
Model 'openai/gpt-oss-120b' not found
Solution:
bash# Check what models the server has loaded
curl http://localhost:8054/v1/models

# Use the correct model name
python classifier_server.py \
    --llm "casperhansen/llama-3.3-70b-instruct-awq" \
    ...

🔮 Future Improvements
1. Connection Pooling
Currently, we create a fresh AsyncOpenAI client for each batch. Could implement connection pooling for better performance:
pythonclass VLLMServerProcessor:
    def __init__(self, ...):
        self.client_pool = []  # Pool of reusable clients
2. Retry Logic
Add automatic retry for transient failures:
pythonasync def _single_inference_with_retry(self, prompt, schema, idx, client):
    for attempt in range(3):
        try:
            return await self._single_inference(...)
        except Exception as e:
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
3. Health Checks
Add automatic health checking of VLLM server:
pythondef is_server_healthy(self) -> bool:
    try:
        response = requests.get(f"{self.server_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
4. Load Balancing
Support multiple VLLM servers for horizontal scaling:
pythonclass VLLMServerProcessor:
    def __init__(self, server_urls: List[str], ...):
        self.servers = server_urls
        self.round_robin_index = 0
5. Metrics & Monitoring
Add Prometheus metrics:
pythonfrom prometheus_client import Counter, Histogram

inference_counter = Counter('vllm_inferences_total', 'Total inferences')
inference_duration = Histogram('vllm_inference_duration_seconds', 'Inference duration')
6. Async Server Mode
Make the classification server fully async for better concurrency:
python@app.post("/classify")
async def classify_texts(request: ClassificationRequest):
    # Already async - use asyncio.gather for parallel processing
    results = await process_batch_async(...)

📚 Related Documentation

VLLM Server Setup: See colleague's vllm_server_gpt_oss.py documentation
Capability System: See classifier/capabilities/README.md
Prompt Engineering: See classifier/prompts.py docstrings
API Reference: See FastAPI auto-docs at http://localhost:9000/docs


🎓 Key Learnings

Abstraction is powerful - By having a clean interface, we could swap implementations with minimal changes
Separation of concerns - Model serving and application logic are independent
Microservices pattern - Two specialized services working together via HTTP
Backward compatibility - New features can be additive without breaking existing functionality

# Dynamic Config Classification Server - Usage Guide

## Overview

The refactored classification server now supports **dynamic config loading**, allowing you to:
- ✅ Run different batch jobs with different configs without restarting the server
- ✅ Use a default config for quick tests
- ✅ Override config per request
- ✅ Run config-less requests for standalone capabilities

---

## Quick Start

### 1️⃣ Start VLLM Servers

```bash
# Start 3 servers (GPUs 7,6 / 5,4 / 3,2)
python vllm_server_manager.py start --gpu-list 7,5,3

# Wait ~2-3 minutes for models to load

# Check status
python vllm_server_manager.py status
```

---

### 2️⃣ Start Classification Server

#### Option A: With Default Config
```bash
python classifier_server.py \
    --backend vllm-server \
    --vllm-server-url "http://localhost:8054/v1,http://localhost:8055/v1,http://localhost:8056/v1" \
    --default-config /path/to/default_topics.json \
    --port 9000
```

#### Option B: Without Default Config
```bash
python classifier_server.py \
    --backend vllm-server \
    --vllm-server-url "http://localhost:8054/v1,http://localhost:8055/v1,http://localhost:8056/v1" \
    --port 9000
```
*(Config required per request for classification)*

---

### 3️⃣ Use the Server

#### A) Quick curl test (uses default config)
```bash
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The training was excellent"],
    "capabilities": ["classification", "recommendations"]
  }' | jq '.'
```

#### B) curl with custom config
```bash
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The training was excellent"],
    "capabilities": ["classification", "recommendations"],
    "config_path": "/path/to/custom_config.json"
  }' | jq '.'
```

#### C) Batch processing with client script
```bash
# Using EEC config
python batch_classify_client.py \
    --input-file /path/to/eec_data.csv \
    --input-column "text" \
    --config /path/to/eec_config.json \
    --server-url http://localhost:9000 \
    --enable-stem-recommendations \
    --enable-stem-polarity \
    --max-stem-definitions 2 \
    --chunk-size 500 \
    --output-dir /path/to/eec_results \
    -v

# Later, using SCE config (same server!)
python batch_classify_client.py \
    --input-file /path/to/sce_data.csv \
    --input-column "comment" \
    --config /path/to/sce_config.json \
    --server-url http://localhost:9000 \
    --enable-classification \
    --enable-recommendations \
    --save-path sce_results.json \
    -v

# Or use server's default config
python batch_classify_client.py \
    --input-file /path/to/data.csv \
    --server-url http://localhost:9000 \
    --enable-classification \
    --save-path results.json \
    -v
```

---

### 4️⃣ Shutdown

```bash
# Stop classification server
# Press Ctrl+C in the terminal where it's running

# Stop all VLLM servers
python vllm_server_manager.py stop
```

---

## Config Priority System

The server uses this priority order when determining which config to use:

1. **Request-level config** (highest priority)
   - Specified via `config_path` in request payload
   - Use this to override default for specific requests

2. **Server default config** (fallback)
   - Specified via `--default-config` at server startup
   - Used when no config in request

3. **No config** (lowest priority)
   - Only works for capabilities that don't require hierarchy
   - Examples: `recommendations`, `alerts` (standalone mode)

---

## Batch Client Options

The `batch_classify_client.py` script mirrors the original `batch_classify.py` interface:

### Required Arguments
```bash
--server-url URL              # Classifier server URL
```

### Input Options
```bash
--input-file FILE             # Input file (.csv, .xlsx, .json, .txt)
--input-column COLUMN         # Column name (auto-detected if omitted)
```
*(Reads from stdin if --input-file not provided)*

### Config Options
```bash
--config PATH                 # Config file (optional if server has default)
--project-name NAME           # Project name for root prefix
```

### Output Options
```bash
--save-path PATH              # Single JSON output file
                              OR
--output-dir DIR              # Directory for chunked output
--chunk-size N                # Texts per chunk (with --output-dir)
```

### Capability Flags
```bash
--enable-classification       # Hierarchical classification (default: on)
--enable-recommendations      # Recommendation detection
--enable-alerts               # Alert detection
--enable-stem-recommendations # Stem-level recommendations
--enable-stem-polarity        # Stem-level polarity
--enable-stem-trends          # Stem-level trend analysis
--enable-trends               # Global trend analysis
```

### Standalone Modes
```bash
--recommendations-only        # Only recommendations (no classification)
--alerts-only                 # Only alerts (no classification)
```

### Other Options
```bash
--request-timeout SECONDS     # Request timeout (default: 300)
-v, -vv, -vvv                 # Verbosity levels
```

---

## Example Workflows

### Example 1: EEC Classification
```bash
# Start server with default EEC config
python classifier_server_refactored.py \
    --backend vllm-server \
    --vllm-server-url "http://localhost:8054/v1,http://localhost:8055/v1,http://localhost:8056/v1" \
    --default-config /path/to/eec_config.json \
    --port 9000

# Process EEC data (uses default config)
python batch_classify_client.py \
    --input-file eec_data.csv \
    --input-column "text" \
    --server-url http://localhost:9000 \
    --project-name "eec" \
    --enable-stem-recommendations \
    --enable-stem-polarity \
    --chunk-size 500 \
    --output-dir eec_results/ \
    -v
```

### Example 2: Multiple Configs on Same Server
```bash
# Start server WITHOUT default config
python classifier_server_refactored.py \
    --backend vllm-server \
    --vllm-server-url "http://localhost:8054/v1,http://localhost:8055/v1,http://localhost:8056/v1" \
    --port 9000

# Process EEC data
python batch_classify_client.py \
    --input-file eec_data.csv \
    --config /path/to/eec_config.json \
    --server-url http://localhost:9000 \
    --save-path eec_results.json \
    -v

# Process SCE data (same server, different config!)
python batch_classify_client.py \
    --input-file sce_data.csv \
    --config /path/to/sce_config.json \
    --server-url http://localhost:9000 \
    --save-path sce_results.json \
    -v

# Process CEE data (yet another config!)
python batch_classify_client.py \
    --input-file cee_data.csv \
    --config /path/to/cee_config.json \
    --server-url http://localhost:9000 \
    --save-path cee_results.json \
    -v
```

### Example 3: Recommendations-Only Mode
```bash
# No config needed for standalone capabilities
python batch_classify_client.py \
    --input-file feedback.csv \
    --server-url http://localhost:9000 \
    --recommendations-only \
    --save-path recommendations.json \
    -v
```

---

## Server Endpoints

### `GET /health`
Check server status
```bash
curl http://localhost:9000/health | jq '.'
```

Response:
```json
{
  "status": "healthy",
  "processor_loaded": true,
  "gpu_count": 6,
  "capabilities_available": ["classification", "recommendations", "alerts", ...],
  "requests_processed": 42,
  "uptime_seconds": 3600.5,
  "default_config": "/path/to/default.json"
}
```

### `GET /stats`
Get server statistics
```bash
curl http://localhost:9000/stats | jq '.'
```

Response:
```json
{
  "total_requests": 42,
  "total_texts_processed": 1250,
  "average_processing_time": 2.34,
  "requests_by_capability": {
    "classification": 30,
    "recommendations": 12
  },
  "uptime_seconds": 3600.5,
  "configs_cached": 3
}
```

### `GET /capabilities`
List available capabilities
```bash
curl http://localhost:9000/capabilities | jq '.'
```

### `POST /classify`
Classify texts
```bash
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["text1", "text2"],
    "capabilities": ["classification"],
    "config_path": "/optional/path/to/config.json",
    "project_name": "optional_project"
  }' | jq '.'
```

---

## Migration from Old System

### Old Way (Fixed Config)
```bash
# Had to restart server to change config ❌
python classifier_server.py \
    --config /path/to/config.json \
    --backend vllm-server \
    --port 9000
```

### New Way (Dynamic Config)
```bash
# Start once, use many configs ✅
python classifier_server_refactored.py \
    --backend vllm-server \
    --default-config /path/to/default.json \
    --port 9000

# Then use different configs per request
python batch_classify_client.py --config /path/to/config1.json ...
python batch_classify_client.py --config /path/to/config2.json ...
python batch_classify_client.py --config /path/to/config3.json ...
```

---

## Performance Notes

### Config Caching
- Configs are loaded once and cached
- Subsequent requests with same config are instant
- Check cache size via `/stats` endpoint

### Chunking Strategy
- Client-side chunking for progress tracking
- Server-side batching for GPU optimization
- Recommended chunk size: 50-500 texts

### Throughput
- Typical: 10-50 texts/second (depends on capabilities)
- Monitor via `--verbose` flag
- Track via `/stats` endpoint

---

## Troubleshooting

### Server won't start
```bash
# Check VLLM servers are running
python vllm_server_manager.py status

# Check ports are available
lsof -i :9000
```

### Config not found
```bash
# Verify path is absolute
--config /absolute/path/to/config.json

# Or use relative from where you run the command
--config ./configs/my_config.json
```

### Request timeout
```bash
# Increase timeout for large batches
python batch_classify_client.py \
    --request-timeout 600 \
    ...
```

### Memory issues
```bash
# Check config cache size
curl http://localhost:9000/stats | jq '.configs_cached'

# Restart server to clear cache if needed
```

---

## Files Summary

| File | Purpose |
|------|---------|
| `classifier_server_refactored.py` | Main server with dynamic config support |
| `classifier_server_manager_refactored.py` | Backend manager with config caching |
| `batch_classify_client.py` | Batch processing client script |
| `vllm_server_manager.py` | VLLM multi-server manager |

---

## Next Steps

1. **Test the refactored server**
   ```bash
   python classifier_server_refactored.py --help
   ```

2. **Test the client**
   ```bash
   python batch_classify_client.py --help
   ```

3. **Run a small test**
   ```bash
   echo "The training was great" | python batch_classify_client.py \
       --server-url http://localhost:9000 \
       --enable-recommendations-only \
       -v
   ```

4. **Process your data**
   ```bash
   python batch_classify_client.py \
       --input-file your_data.csv \
       --config your_config.json \
       --server-url http://localhost:9000 \
       --save-path results.json \
       -v
   ```

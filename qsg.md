🚀 Quick Start Guide
1️⃣ Start VLLM Servers
bash# Start 3 servers (GPUs 7,6 / 5,4 / 3,2)
python vllm_server_manager.py start --gpu-list 7,5,3

# Wait ~2-3 minutes for models to load
# Check status
python vllm_server_manager.py status
2️⃣ Start Classification Server
bashpython classifier_server.py \
    --config /path/to/topics.json \
    --backend vllm-server \
    --vllm-server-url "http://localhost:8054/v1,http://localhost:8055/v1,http://localhost:8056/v1" \
    --port 9000
3️⃣ Send Request
bash# Quick test
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The training was excellent"],
    "capabilities": ["classification", "recommendations"]
  }' | jq '.'
4️⃣ Shutdown
bash# Stop classification server
# Press Ctrl+C in the terminal where it's running

# Stop all VLLM servers
python vllm_server_manager.py stop

#!/bin/bash
# Auto-detect server port and test it

echo "🔍 Finding classification server..."

# Try common ports
PORTS=(9000 8000 8080 8888 7000 9001 8081)
SERVER_PORT=""

for PORT in "${PORTS[@]}"; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✓ Found server on port $PORT"
        SERVER_PORT=$PORT
        break
    fi
done

if [ -z "$SERVER_PORT" ]; then
    echo "❌ Server not found on common ports"
    echo "Is the server running?"
    echo ""
    echo "Start it with:"
    echo "  python classifier_server.py --config topics.json --gpu-list 0,1,2,3 --port 9000"
    exit 1
fi

echo ""
echo "==================== SERVER INFO ===================="
curl -s "http://localhost:$SERVER_PORT/health" | python3 -m json.tool
echo ""

echo "==================== TEST REQUEST ===================="
echo "Classifying: 'I really think that we should practice more examples in class'"
echo ""

curl -s -X POST "http://localhost:$SERVER_PORT/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["I really think that we should practice more examples in class"],
    "capabilities": ["classification", "recommendations", "alerts"]
  }' | python3 -m json.tool

echo ""
echo "==================== DONE ===================="
echo ""
echo "Server is running on: http://localhost:$SERVER_PORT"
echo ""
echo "Use this in your code/curl commands:"
echo "  http://localhost:$SERVER_PORT"

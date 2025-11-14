#!/bin/bash
# Test script for Classification Server
# Quick commands to test the server functionality

BASE_URL="http://localhost:8000"

echo "=========================================="
echo "Classification Server Test Script"
echo "=========================================="
echo ""

# Check if server is running
echo "1. Health Check"
echo "---"
curl -s "$BASE_URL/health" | jq '.' || echo "Server not responding. Is it running?"
echo ""
echo ""

# List capabilities
echo "2. List Capabilities"
echo "---"
curl -s "$BASE_URL/capabilities" | jq '.'
echo ""
echo ""

# Basic classification
echo "3. Basic Classification"
echo "---"
curl -s -X POST "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The instructor was excellent and very knowledgeable."],
    "capabilities": ["classification"]
  }' | jq '.results[0].classification_result.classification_paths'
echo ""
echo ""

# Multiple texts with recommendations
echo "4. Multiple Texts + Recommendations"
echo "---"
curl -s -X POST "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "We should add more hands-on exercises to the training.",
      "The course was great overall."
    ],
    "capabilities": ["classification", "recommendations"]
  }' | jq '.processing_time, .batch_info'
echo ""
echo ""

# With alerts
echo "5. Alert Detection"
echo "---"
curl -s -X POST "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["My supervisor makes inappropriate comments."],
    "capabilities": ["classification", "alerts"]
  }' | jq '.results[0].alerts'
echo ""
echo ""

# Server statistics
echo "6. Server Statistics"
echo "---"
curl -s "$BASE_URL/stats" | jq '.'
echo ""
echo ""

echo "=========================================="
echo "Test Complete!"
echo "=========================================="

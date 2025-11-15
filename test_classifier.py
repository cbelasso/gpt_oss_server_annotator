#!/usr/bin/env python3
"""
Test the classification server and save results nicely.
"""

from datetime import datetime
import json
from pathlib import Path

import requests

# Configuration
SERVER_URL = "http://localhost:9000/classify"
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test data
test_cases = [
    {
        "name": "instructor_complaint",
        "texts": [
            "it sucks that the university hasnt gotten a better instructor for this COEN 233 course for the past 5 years. Its really frustrating. You would think that by now they wouldve gotten the message after everyones feeedback. I think they should take our feedback more seriously"
        ],
        "capabilities": ["classification", "recommendations", "stem_trend"],
    },
    {
        "name": "positive_feedback",
        "texts": [
            "The training was excellent and the instructor was very knowledgeable. I learned a lot!"
        ],
        "capabilities": ["classification", "recommendations", "stem_polarity"],
    },
]


def run_test(test_case):
    """Run a single test case and save results."""
    print(f"\n{'=' * 60}")
    print(f"Running test: {test_case['name']}")
    print(f"{'=' * 60}")
    print(f"Capabilities: {', '.join(test_case['capabilities'])}")

    # Make request
    response = requests.post(
        SERVER_URL,
        json={"texts": test_case["texts"], "capabilities": test_case["capabilities"]},
        headers={"Content-Type": "application/json"},
    )

    # Check response
    if response.status_code == 200:
        result = response.json()

        # Print summary
        print(f"✓ Success! Processing time: {result['processing_time']:.2f}s")
        print(f"  Texts processed: {result['batch_info']['text_count']}")

        # Save to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"{test_case['name']}_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Results saved to: {output_file}")

        # Print quick preview
        print("\n  Quick Preview:")
        for text_result in result["results"]:
            if "classification_result" in text_result:
                paths = text_result["classification_result"].get("classification_paths", [])
                print(f"    Classification paths: {len(paths)} found")
                for path in paths[:3]:  # Show first 3
                    print(f"      • {path}")
            if "recommendations" in text_result:
                recs = text_result["recommendations"]
                print(f"    Recommendations: {len(recs)} found")
            if "stem_trends" in text_result:
                trends = text_result.get("stem_trends", {})
                print(f"    Stem trends: {len(trends)} stems analyzed")

        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")
        return False


def main():
    """Run all test cases."""
    print("Classification Server Test Suite")
    print(f"Server: {SERVER_URL}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = []
    for test_case in test_cases:
        success = run_test(test_case)
        results.append((test_case["name"], success))

    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary:")
    print(f"{'=' * 60}")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

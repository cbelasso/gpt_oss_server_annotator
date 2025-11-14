"""
Simple test client that shows FULL JSON results.
Use this to see complete classification outputs.
"""

import asyncio
import json
import sys

import aiohttp


async def classify_and_print(
    texts,
    capabilities=None,
    base_url="http://localhost:9000",  # Change port if needed
):
    """
    Classify texts and print full JSON response.
    """
    if capabilities is None:
        capabilities = ["classification"]

    payload = {
        "texts": texts,
        "capabilities": capabilities,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/classify",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status == 200:
                result = await response.json()

                # Print full JSON with nice formatting
                print("\n" + "=" * 80)
                print("FULL CLASSIFICATION RESULTS")
                print("=" * 80)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("=" * 80 + "\n")

                return result
            else:
                error = await response.text()
                print(f"Error {response.status}: {error}")
                return None


async def main():
    """Run test examples."""

    # CHANGE THIS to match your server port!
    BASE_URL = "http://localhost:9000"  # <-- UPDATE THIS

    print("Testing Classification Server")
    print(f"Server URL: {BASE_URL}")
    print("\n")

    # Test 1: Basic classification
    print("TEST 1: Basic Classification")
    await classify_and_print(
        texts=["The instructor was excellent and explained everything clearly."],
        capabilities=["classification"],
        base_url=BASE_URL,
    )

    # Test 2: With recommendations
    print("\nTEST 2: Classification + Recommendations")
    await classify_and_print(
        texts=["We should add more hands-on exercises to the training."],
        capabilities=["classification", "recommendations"],
        base_url=BASE_URL,
    )

    # Test 3: With alerts
    print("\nTEST 3: Classification + Alerts")
    await classify_and_print(
        texts=["My supervisor makes inappropriate comments about my appearance."],
        capabilities=["classification", "alerts"],
        base_url=BASE_URL,
    )

    # Test 4: Everything
    print("\nTEST 4: All Capabilities")
    await classify_and_print(
        texts=["The course was good but could be better organized."],
        capabilities=["classification", "recommendations", "alerts"],
        base_url=BASE_URL,
    )

    # Test 5: Multiple texts
    print("\nTEST 5: Multiple Texts")
    await classify_and_print(
        texts=[
            "The instructor was great.",
            "We need better equipment.",
            "The material was outdated.",
        ],
        capabilities=["classification", "recommendations"],
        base_url=BASE_URL,
    )


if __name__ == "__main__":
    # Check if user wants to specify port
    if len(sys.argv) > 1:
        port = sys.argv[1]
        print(f"Using port: {port}")

    asyncio.run(main())

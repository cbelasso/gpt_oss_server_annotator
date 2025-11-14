"""
Example client for the Classification Server

Demonstrates how to make requests to the server with different capability combinations.
"""

import asyncio
import json
import time
from typing import List

import aiohttp


class ClassificationClient:
    """Async client for the classification server."""

    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url

    async def health_check(self):
        """Check server health."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()

    async def get_capabilities(self):
        """Get list of available capabilities."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/capabilities") as response:
                return await response.json()

    async def get_stats(self):
        """Get server statistics."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/stats") as response:
                return await response.json()

    async def classify(
        self,
        texts: List[str],
        capabilities: List[str] = None,
        project_name: str = None,
    ):
        """
        Classify texts with specified capabilities.

        Args:
            texts: List of texts to classify
            capabilities: List of capabilities (default: ["classification"])
            project_name: Optional project name

        Returns:
            Classification results
        """
        if capabilities is None:
            capabilities = ["classification"]

        payload = {
            "texts": texts,
            "capabilities": capabilities,
        }

        if project_name:
            payload["project_name"] = project_name

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/classify",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Request failed: {response.status} - {error}")


async def example_basic_classification():
    """Example: Basic classification only."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Classification")
    print("=" * 80)

    client = ClassificationClient()

    texts = [
        "The instructor was excellent and explained everything clearly.",
        "We need more hands-on exercises and practical examples.",
        "The training room was too cold and uncomfortable.",
    ]

    result = await client.classify(texts=texts, capabilities=["classification"])

    print(f"\nProcessing time: {result['processing_time']:.2f}s")
    print(f"Texts processed: {result['batch_info']['text_count']}")

    for i, text_result in enumerate(result["results"]):
        print(f"\n--- Text {i + 1} ---")
        print(f"Text: {text_result['text'][:60]}...")
        if "classification_result" in text_result:
            paths = text_result["classification_result"].get("classification_paths", [])
            print(f"Paths: {paths}")


async def example_multi_capability():
    """Example: Classification + Recommendations + Alerts."""
    print("\n" + "=" * 80)
    print("Example 2: Multiple Capabilities")
    print("=" * 80)

    client = ClassificationClient()

    texts = [
        "The instructor should provide more examples during lectures.",
        "My supervisor makes inappropriate comments about my appearance.",
    ]

    result = await client.classify(
        texts=texts, capabilities=["classification", "recommendations", "alerts"]
    )

    print(f"\nProcessing time: {result['processing_time']:.2f}s")

    for i, text_result in enumerate(result["results"]):
        print(f"\n--- Text {i + 1} ---")
        print(f"Text: {text_result['text'][:60]}...")

        if "recommendations" in text_result:
            recs = text_result["recommendations"]
            print(f"Recommendations found: {len(recs)}")

        if "alerts" in text_result:
            alerts = text_result["alerts"]
            print(f"Alerts found: {len(alerts)}")


async def example_stem_analysis():
    """Example: Classification with stem recommendations and polarity."""
    print("\n" + "=" * 80)
    print("Example 3: Stem Analysis")
    print("=" * 80)

    client = ClassificationClient()

    texts = [
        "The teaching style was excellent but we need more interactive activities.",
    ]

    result = await client.classify(
        texts=texts,
        capabilities=[
            "classification",
            "stem_recommendations",
            "stem_polarity",
        ],
    )

    print(f"\nProcessing time: {result['processing_time']:.2f}s")

    for text_result in result["results"]:
        print(f"\nText: {text_result['text']}")

        if "stem_recommendations" in text_result:
            print("\nStem Recommendations:")
            print(json.dumps(text_result["stem_recommendations"], indent=2))

        if "stem_polarity" in text_result:
            print("\nStem Polarity:")
            print(json.dumps(text_result["stem_polarity"], indent=2))


async def example_concurrent_requests():
    """Example: Multiple concurrent requests with different capabilities."""
    print("\n" + "=" * 80)
    print("Example 4: Concurrent Requests (Intelligent Batching Demo)")
    print("=" * 80)

    client = ClassificationClient()

    # Simulate different users making different requests simultaneously
    tasks = [
        # User 1: Just wants classification
        client.classify(
            texts=["The course was very informative."], capabilities=["classification"]
        ),
        # User 2: Wants recommendations
        client.classify(
            texts=["We should add more videos to the training."],
            capabilities=["classification", "recommendations"],
        ),
        # User 3: Wants alerts
        client.classify(
            texts=["My manager threatens me when I make mistakes."],
            capabilities=["classification", "alerts"],
        ),
        # User 4: Wants everything
        client.classify(
            texts=["The training was good but could be better organized."],
            capabilities=["classification", "recommendations", "alerts"],
        ),
    ]

    start = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start

    print(f"\nTotal time for 4 concurrent requests: {total_time:.2f}s")
    print("\nThe server batched these requests intelligently!")
    print("It executed each capability once for all texts that needed it.")


async def example_batch_request():
    """Example: Single request with many texts."""
    print("\n" + "=" * 80)
    print("Example 5: Batch Request")
    print("=" * 80)

    client = ClassificationClient()

    # Many texts in one request
    texts = [
        "The instructor was excellent.",
        "We need better equipment.",
        "The course material was outdated.",
        "Great hands-on exercises!",
        "Too much theory, not enough practice.",
    ] * 2  # 10 texts total

    result = await client.classify(
        texts=texts, capabilities=["classification", "recommendations"]
    )

    print(f"\nProcessing time: {result['processing_time']:.2f}s")
    print(f"Texts processed: {len(result['results'])}")
    print(f"Average time per text: {result['processing_time'] / len(texts):.3f}s")


async def main():
    """Run all examples."""
    client = ClassificationClient()

    # Check server health
    print("Checking server health...")
    try:
        health = await client.health_check()
        print(f"✓ Server is {health['status']}")
        print(f"  Processor loaded: {health['processor_loaded']}")
        print(f"  GPUs: {health['gpu_count']}")
        print(f"  Capabilities: {health['capabilities_available']}")
    except Exception as e:
        print(f"✗ Server is not responding: {e}")
        print("\nMake sure to start the server first:")
        print("  python classifier_server.py --config topics.json --gpu-list 0,1,2,3")
        return

    # Run examples
    await example_basic_classification()
    await example_multi_capability()
    # await example_stem_analysis()  # Uncomment if you want stem analysis
    await example_concurrent_requests()
    await example_batch_request()

    # Show final stats
    print("\n" + "=" * 80)
    print("Server Statistics")
    print("=" * 80)
    stats = await client.get_stats()
    print(f"\nTotal requests: {stats['total_requests']}")
    print(f"Total texts processed: {stats['total_texts_processed']}")
    print(f"Average processing time: {stats['average_processing_time']:.2f}s")
    print("Requests by capability:")
    for cap, count in stats["requests_by_capability"].items():
        print(f"  - {cap}: {count}")


if __name__ == "__main__":
    asyncio.run(main())

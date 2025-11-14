"""
Ultra-simple test - shows full JSON output.
Modify the PORT, texts, and capabilities as needed.
"""

import asyncio
import json

import aiohttp

# ============ CONFIGURATION ============
PORT = 9000  # <-- CHANGE THIS to match your server port
BASE_URL = f"http://localhost:{PORT}"

# Your test data
TEXTS = [
    "I really think that we should practice more examples in class. I think that the professor is a bit of a motherfucker. I think he should die because of the way that he harassed that girl in school."
]

CAPABILITIES = ["classification", "recommendations", "alerts"]
# =======================================


async def test():
    payload = {"texts": TEXTS, "capabilities": CAPABILITIES}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/classify", json=payload, timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status == 200:
                result = await response.json()

                # Print beautiful JSON
                print("\n" + "=" * 80)
                print("RESULTS")
                print("=" * 80)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("=" * 80)
            else:
                print(f"Error: {response.status}")
                print(await response.text())


if __name__ == "__main__":
    asyncio.run(test())

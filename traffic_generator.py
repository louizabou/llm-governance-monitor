import asyncio
import random
import httpx

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "What are the benefits of cloud computing?",
    "How does machine learning work?",
]

async def send_request(prompt):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/chat",
                json={"message": prompt}
            )
            return response.status_code == 200
        except:
            return False

async def main():
    print("Generating traffic...")
    for i in range(10):
        prompt = random.choice(PROMPTS)
        success = await send_request(prompt)
        status = "✅" if success else "❌"
        print(f"  [{i+1}/10] {status} - {prompt[:40]}...")
        await asyncio.sleep(1)
    print("✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())
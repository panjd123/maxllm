"""
Basic test for MaxLLM - Text completion
"""
import asyncio
import sys
import os

# Add parent directory to path for importing maxllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from maxllm import async_openai_complete, get_call_status


async def test_basic_completion():
    """Test basic text completion"""
    print("=" * 60)
    print("Test 1: Basic Text Completion")
    print("=" * 60)

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt="Write a haiku about programming",
        system_prompt="You are a creative poet."
    )

    print(f"\nPrompt: Write a haiku about programming")
    print(f"Response:\n{response}")
    print()


async def test_with_history():
    """Test completion with conversation history"""
    print("=" * 60)
    print("Test 2: Completion with History")
    print("=" * 60)

    history = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice!"}
    ]

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt="What's my name?",
        history_messages=history
    )

    print(f"Conversation history: {history}")
    print(f"New prompt: What's my name?")
    print(f"Response: {response}")
    print()


async def test_cache():
    """Test caching functionality"""
    print("=" * 60)
    print("Test 3: Cache Testing")
    print("=" * 60)

    prompt = "Count from 1 to 5"

    print(f"First request (should call API): {prompt}")
    response1 = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt
    )
    print(f"Response: {response1}\n")

    print(f"Second request (should use cache): {prompt}")
    response2 = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt
    )
    print(f"Response: {response2}")
    print(f"Responses match: {response1 == response2}")
    print()


async def main():
    print("\n" + "=" * 60)
    print("MaxLLM Basic Tests")
    print("=" * 60 + "\n")

    # Run tests
    await test_basic_completion()
    await test_with_history()
    await test_cache()

    # Show statistics
    print("=" * 60)
    print("Call Statistics")
    print("=" * 60)
    status = get_call_status()
    print(status)
    print()


if __name__ == "__main__":
    asyncio.run(main())

"""
Batch processing test for MaxLLM
"""
import asyncio
import sys
import os

# Add parent directory to path for importing maxllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from maxllm import batch_complete, get_call_status


async def test_batch_simple():
    """Test simple batch processing"""
    print("=" * 60)
    print("Test 1: Simple Batch Processing")
    print("=" * 60)

    prompts = [
        "Write a haiku about the number 1",
        "Write a haiku about the number 2",
        "Write a haiku about the number 3",
        "Write a haiku about the number 4",
        "Write a haiku about the number 5",
    ]

    print(f"Processing {len(prompts)} prompts in batch...")
    print()

    results = await batch_complete(
        prompts=prompts,
        model="gpt-4o-mini",
        desc="Generating haikus",
        concurrency=3  # Process 3 at a time
    )

    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"Prompt {i}: {prompt}")
        print(f"Result {i}:\n{result}")
        print("-" * 60)

    print()


async def test_batch_with_params():
    """Test batch processing with additional parameters"""
    print("=" * 60)
    print("Test 2: Batch with Additional Parameters")
    print("=" * 60)

    # Each item can be a dict with parameters
    tasks = [
        {
            "prompt": "Explain quantum computing",
            "system_prompt": "You are a physics teacher. Explain in simple terms.",
            "max_tokens": 100
        },
        {
            "prompt": "Explain machine learning",
            "system_prompt": "You are a computer science teacher. Explain in simple terms.",
            "max_tokens": 100
        },
        {
            "prompt": "Explain blockchain",
            "system_prompt": "You are a technology expert. Explain in simple terms.",
            "max_tokens": 100
        },
    ]

    print(f"Processing {len(tasks)} tasks with custom parameters...")
    print()

    results = await batch_complete(
        prompts=tasks,
        model="gpt-4o-mini",
        desc="Explaining concepts",
        concurrency=2
    )

    for i, (task, result) in enumerate(zip(tasks, results), 1):
        print(f"Task {i}: {task['prompt']}")
        print(f"System: {task['system_prompt']}")
        print(f"Result {i}:\n{result}")
        print("-" * 60)

    print()


async def test_batch_error_handling():
    """Test error handling in batch processing"""
    print("=" * 60)
    print("Test 3: Error Handling with Placeholder")
    print("=" * 60)

    prompts = [
        "Write a short sentence",
        "",  # This might cause an error
        "Write another short sentence",
    ]

    print(f"Processing {len(prompts)} prompts (including one that may fail)...")
    print()

    results = await batch_complete(
        prompts=prompts,
        model="gpt-4o-mini",
        desc="Testing error handling",
        concurrency=2,
        placeholder="[ERROR: Failed to process]"
    )

    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"Prompt {i}: '{prompt}'")
        print(f"Result {i}: {result}")
        print("-" * 60)

    print()


async def main():
    print("\n" + "=" * 60)
    print("MaxLLM Batch Processing Tests")
    print("=" * 60 + "\n")

    # Run tests
    await test_batch_simple()
    await test_batch_with_params()
    # await test_batch_error_handling()  # Uncomment to test error handling

    # Show statistics
    print("=" * 60)
    print("Call Statistics")
    print("=" * 60)
    status = get_call_status()
    print(status)
    print()


if __name__ == "__main__":
    asyncio.run(main())

"""
JSON mode and structured output test for MaxLLM
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
from pydantic import BaseModel
from typing import List


class Keywords(BaseModel):
    """Schema for keyword extraction"""
    keywords: List[str]


class Person(BaseModel):
    """Schema for person information"""
    name: str
    age: int
    occupation: str


async def test_json_mode():
    """Test basic JSON mode output"""
    print("=" * 60)
    print("Test 1: JSON Mode (returns string)")
    print("=" * 60)

    prompt = """Extract keywords from this text and return as JSON with a 'keywords' field.

Text: Machine learning is a subset of artificial intelligence that enables computers to learn from data."""

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt,
        json_mode=True
    )

    print(f"Prompt: Extract keywords...")
    print(f"Response type: {type(response)}")
    print(f"Response:\n{response}")
    print()


async def test_json_format():
    """Test structured output with Pydantic model"""
    print("=" * 60)
    print("Test 2: Structured Output with Pydantic (returns dict)")
    print("=" * 60)

    prompt = """Extract keywords from this text:

Machine learning is a subset of artificial intelligence that enables computers to learn from data."""

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt,
        json_format=Keywords
    )

    print(f"Prompt: Extract keywords...")
    print(f"Response type: {type(response)}")
    print(f"Response:\n{response}")
    print(f"Keywords: {response.get('keywords', [])}")
    print()


async def test_complex_schema():
    """Test with a more complex schema"""
    print("=" * 60)
    print("Test 3: Complex Schema - Person Info")
    print("=" * 60)

    prompt = """Extract person information from this text and return as JSON:

John Smith is a 35-year-old software engineer working at Tech Corp."""

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt,
        json_format=Person
    )

    print(f"Prompt: Extract person info...")
    print(f"Response type: {type(response)}")
    print(f"Response:\n{response}")
    print(f"Name: {response.get('name')}")
    print(f"Age: {response.get('age')}")
    print(f"Occupation: {response.get('occupation')}")
    print()


async def main():
    print("\n" + "=" * 60)
    print("MaxLLM JSON/Structured Output Tests")
    print("=" * 60 + "\n")

    # Run tests
    await test_json_mode()
    await test_json_format()
    await test_complex_schema()

    # Show statistics
    print("=" * 60)
    print("Call Statistics")
    print("=" * 60)
    status = get_call_status()
    print(status)
    print()


if __name__ == "__main__":
    asyncio.run(main())

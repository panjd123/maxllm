"""
Embedding generation test for MaxLLM
"""
import asyncio
import sys
import os

# Add parent directory to path for importing maxllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from maxllm import async_openai_complete, batch_embedding, get_call_status


async def test_single_embedding():
    """Test single text embedding"""
    print("=" * 60)
    print("Test 1: Single Text Embedding")
    print("=" * 60)

    text = "Machine learning is a subset of artificial intelligence"

    print(f"Text: {text}")

    embedding = await async_openai_complete(
        model="text-embedding-3-small",
        input=text
    )

    print(f"Embedding type: {type(embedding)}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print()


async def test_batch_embedding_api():
    """Test batch embedding using async_openai_complete"""
    print("=" * 60)
    print("Test 2: Batch Embedding (API)")
    print("=" * 60)

    texts = [
        "Hello world",
        "Machine learning is amazing",
        "Python is a great programming language"
    ]

    print(f"Texts to embed: {len(texts)}")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    embeddings = await async_openai_complete(
        model="text-embedding-3-small",
        input=texts
    )

    print(f"\nEmbeddings type: {type(embeddings)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Each embedding dimensions: {len(embeddings[0])}")
    print(f"First embedding preview: {embeddings[0][:5]}...")
    print()


async def test_batch_embedding_helper():
    """Test batch embedding using helper function"""
    print("=" * 60)
    print("Test 3: Batch Embedding (Helper Function)")
    print("=" * 60)

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the world",
        "Python programming is fun and powerful",
        "Data science combines statistics and programming",
        "Machine learning models learn from data"
    ]

    print(f"Texts to embed: {len(texts)}")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text[:50]}...")

    embeddings = await batch_embedding(
        texts=texts,
        model="text-embedding-3-small",
        desc="Generating embeddings",
        concurrency=3
    )

    print(f"\nEmbeddings type: {type(embeddings)}")
    print(f"Number of embeddings: {len(embeddings)}")
    if embeddings and embeddings[0]:
        print(f"Each embedding dimensions: {len(embeddings[0])}")
        print(f"First embedding preview: {embeddings[0][:5]}...")
    print()


async def test_embedding_similarity():
    """Test embedding similarity calculation"""
    print("=" * 60)
    print("Test 4: Embedding Similarity")
    print("=" * 60)

    texts = [
        "I love programming",
        "I enjoy coding",
        "The weather is nice today"
    ]

    print(f"Generating embeddings for similarity comparison...")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    embeddings = await async_openai_complete(
        model="text-embedding-3-small",
        input=texts
    )

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))
        return dot_product / (magnitude_a * magnitude_b)

    print(f"\nSimilarity scores:")
    print(f"  '{texts[0]}' vs '{texts[1]}': {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"  '{texts[0]}' vs '{texts[2]}': {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"  '{texts[1]}' vs '{texts[2]}': {cosine_similarity(embeddings[1], embeddings[2]):.4f}")
    print()


async def main():
    print("\n" + "=" * 60)
    print("MaxLLM Embedding Tests")
    print("=" * 60 + "\n")

    # Run tests
    await test_single_embedding()
    await test_batch_embedding_api()
    await test_batch_embedding_helper()
    await test_embedding_similarity()

    # Show statistics
    print("=" * 60)
    print("Call Statistics")
    print("=" * 60)
    status = get_call_status()
    print(status)
    print()


if __name__ == "__main__":
    asyncio.run(main())

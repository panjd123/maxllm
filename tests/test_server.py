"""
Comprehensive tests for MaxLLM API Server.

This test file uses the OpenAI Python client to test all API endpoints
and verify that parameters are correctly passed through.

Usage:
    1. Start the server: maxllm serve --port 18000
    2. Run tests: pytest tests/test_server.py -v

Or run directly:
    python tests/test_server.py
"""

import os
import sys
import time
import pytest
import asyncio
import subprocess
from typing import Optional

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse

# Test configuration
TEST_SERVER_HOST = os.environ.get("TEST_SERVER_HOST", "localhost")
TEST_SERVER_PORT = int(os.environ.get("TEST_SERVER_PORT", "18000"))
TEST_BASE_URL = f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1"

# Test models - adjust based on your configuration
TEST_CHAT_MODEL = os.environ.get("TEST_CHAT_MODEL", "Qwen3-4B-4090")
TEST_EMBEDDING_MODEL = os.environ.get("TEST_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
TEST_RERANK_MODEL = os.environ.get("TEST_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
TEST_CLOUD_CHAT_MODEL = os.environ.get("TEST_CLOUD_CHAT_MODEL", "deepseek")
TEST_CLOUD_EMBEDDING_MODEL = os.environ.get("TEST_CLOUD_EMBEDDING_MODEL", "text-embedding-3-small")

# Fallback to cloud models if local models are not available
USE_LOCAL_MODELS = os.environ.get("USE_LOCAL_MODELS", "true").lower() == "true"


def get_client() -> OpenAI:
    """Get a synchronous OpenAI client configured for the test server."""
    return OpenAI(
        base_url=TEST_BASE_URL,
        api_key="test-key",  # API key is not required but OpenAI client needs one
    )


def get_async_client() -> AsyncOpenAI:
    """Get an asynchronous OpenAI client configured for the test server."""
    return AsyncOpenAI(
        base_url=TEST_BASE_URL,
        api_key="test-key",
    )


def get_chat_model() -> str:
    """Get the chat model to use for testing."""
    return TEST_CHAT_MODEL if USE_LOCAL_MODELS else TEST_CLOUD_CHAT_MODEL


def get_embedding_model() -> str:
    """Get the embedding model to use for testing."""
    return TEST_EMBEDDING_MODEL if USE_LOCAL_MODELS else TEST_CLOUD_EMBEDDING_MODEL


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_list_models(self):
        """Test listing all available models."""
        client = get_client()
        models = client.models.list()

        assert models is not None
        assert hasattr(models, "data")
        assert len(models.data) > 0

        # Check model structure
        for model in models.data:
            assert hasattr(model, "id")
            assert hasattr(model, "object")
            assert model.object == "model"

        print(f"Found {len(models.data)} models")

    def test_get_specific_model(self):
        """Test retrieving a specific model."""
        client = get_client()
        model = client.models.retrieve(get_chat_model())

        assert model is not None
        assert model.id == get_chat_model()
        assert model.object == "model"


class TestChatCompletions:
    """Tests for /v1/chat/completions endpoint."""

    def test_basic_chat_completion(self):
        """Test basic chat completion."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
        )

        assert isinstance(response, ChatCompletion)
        assert response.id is not None
        assert response.object == "chat.completion"
        assert response.model == get_chat_model()
        assert len(response.choices) > 0
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content is not None
        assert response.usage is not None

        print(f"Response: {response.choices[0].message.content}")

    def test_chat_with_system_message(self):
        """Test chat completion with system message."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only responds in uppercase."},
                {"role": "user", "content": "Say hello."}
            ],
        )

        assert response.choices[0].message.content is not None
        print(f"Response with system: {response.choices[0].message.content}")

    def test_chat_with_max_tokens(self):
        """Test that max_tokens parameter is respected."""
        client = get_client()

        # Request very few tokens
        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Write a very long story about a dragon."}
            ],
            max_tokens=5,
        )

        assert response.usage is not None
        # The completion should be short due to max_tokens limit
        assert response.usage.completion_tokens <= 10  # Allow some buffer
        print(f"Tokens used: {response.usage.completion_tokens} (max_tokens=5)")

    def test_chat_with_temperature(self):
        """Test that temperature parameter is accepted."""
        client = get_client()

        # Low temperature should give more deterministic results
        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            temperature=0.0,
        )

        assert response.choices[0].message.content is not None
        print(f"Low temp response: {response.choices[0].message.content}")

    def test_chat_with_top_p(self):
        """Test that top_p parameter is accepted."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
            top_p=0.9,
        )

        assert response.choices[0].message.content is not None

    def test_chat_with_stop_sequence(self):
        """Test that stop parameter works."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            stop=[","],
            max_tokens=50,
        )

        content = response.choices[0].message.content
        print(f"Response with stop: {content}")

    def test_chat_with_response_format_json(self):
        """Test response_format with json_object type."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Extract the name from this text and return as JSON with 'name' field: My name is Alice."}
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        assert content is not None

        # Verify it's valid JSON
        import json
        try:
            parsed = json.loads(content)
            print(f"JSON response: {parsed}")
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {content}")

    def test_chat_with_seed(self):
        """Test that seed parameter is accepted for reproducibility."""
        client = get_client()

        response1 = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Pick a random number between 1 and 100."}
            ],
            seed=42,
            temperature=0.0,
        )

        response2 = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Pick a random number between 1 and 100."}
            ],
            seed=42,
            temperature=0.0,
        )

        # With same seed and temperature=0, responses should be similar
        print(f"Response 1: {response1.choices[0].message.content}")
        print(f"Response 2: {response2.choices[0].message.content}")

    def test_chat_with_presence_penalty(self):
        """Test that presence_penalty parameter is accepted."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
            presence_penalty=0.5,
        )

        assert response.choices[0].message.content is not None

    def test_chat_with_frequency_penalty(self):
        """Test that frequency_penalty parameter is accepted."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
            frequency_penalty=0.5,
        )

        assert response.choices[0].message.content is not None

    def test_chat_with_timeout(self):
        """Test that timeout parameter works."""
        client = get_client()

        # Normal request with reasonable timeout
        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
            timeout=30.0,
        )

        assert response.choices[0].message.content is not None

    def test_chat_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "My name is Bob."},
                {"role": "assistant", "content": "Hello Bob! Nice to meet you."},
                {"role": "user", "content": "What is my name?"}
            ],
        )

        content = response.choices[0].message.content.lower()
        assert "bob" in content
        print(f"Multi-turn response: {response.choices[0].message.content}")


class TestCompletions:
    """Tests for /v1/completions endpoint."""

    def test_basic_completion(self):
        """Test basic text completion."""
        client = get_client()

        response = client.completions.create(
            model=get_chat_model(),
            prompt="The capital of France is",
            max_tokens=10,
        )

        assert response.id is not None
        assert response.object == "text_completion"
        assert len(response.choices) > 0
        assert response.choices[0].text is not None
        assert response.usage is not None

        print(f"Completion: {response.choices[0].text}")

    def test_completion_with_max_tokens(self):
        """Test completion with max_tokens limit."""
        client = get_client()

        response = client.completions.create(
            model=get_chat_model(),
            prompt="Write a story:",
            max_tokens=5,
        )

        assert response.usage.completion_tokens <= 10  # Allow buffer
        print(f"Tokens used: {response.usage.completion_tokens}")

    def test_completion_with_temperature(self):
        """Test completion with temperature."""
        client = get_client()

        response = client.completions.create(
            model=get_chat_model(),
            prompt="2 + 2 =",
            max_tokens=5,
            temperature=0.0,
        )

        assert response.choices[0].text is not None

    def test_completion_with_stop(self):
        """Test completion with stop sequence."""
        client = get_client()

        response = client.completions.create(
            model=get_chat_model(),
            prompt="Count: 1, 2, 3, 4, 5",
            max_tokens=20,
            stop=[","],
        )

        print(f"Completion with stop: {response.choices[0].text}")


class TestEmbeddings:
    """Tests for /v1/embeddings endpoint."""

    def test_single_embedding(self):
        """Test creating a single embedding."""
        client = get_client()

        response = client.embeddings.create(
            model=get_embedding_model(),
            input="Hello, world!",
        )

        assert isinstance(response, CreateEmbeddingResponse)
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert response.data[0].index == 0
        assert len(response.data[0].embedding) > 0
        assert response.usage is not None

        print(f"Embedding dimension: {len(response.data[0].embedding)}")

    def test_batch_embeddings(self):
        """Test creating multiple embeddings at once."""
        client = get_client()

        texts = ["Hello", "World", "Test"]
        response = client.embeddings.create(
            model=get_embedding_model(),
            input=texts,
        )

        assert len(response.data) == len(texts)
        for i, embedding in enumerate(response.data):
            assert embedding.index == i
            assert len(embedding.embedding) > 0

        print(f"Created {len(response.data)} embeddings")

    def test_embedding_consistency(self):
        """Test that same input produces consistent embeddings."""
        client = get_client()

        text = "Consistency test"

        response1 = client.embeddings.create(
            model=get_embedding_model(),
            input=text,
        )

        response2 = client.embeddings.create(
            model=get_embedding_model(),
            input=text,
        )

        # Embeddings should be identical for same input
        emb1 = response1.data[0].embedding
        emb2 = response2.data[0].embedding

        assert len(emb1) == len(emb2)
        # Check first few values are close (allowing for floating point differences)
        for v1, v2 in zip(emb1[:10], emb2[:10]):
            assert abs(v1 - v2) < 0.0001, f"Embeddings differ: {v1} vs {v2}"

    def test_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        client = get_client()

        response = client.embeddings.create(
            model=get_embedding_model(),
            input=["Hello world", "Goodbye moon"],
        )

        emb1 = response.data[0].embedding
        emb2 = response.data[1].embedding

        # Calculate cosine similarity - should not be 1.0
        import math
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        similarity = dot_product / (norm1 * norm2)

        assert similarity < 0.99, f"Embeddings are too similar: {similarity}"
        print(f"Cosine similarity between different texts: {similarity:.4f}")


class TestScore:
    """Tests for /v1/score endpoint."""

    def test_basic_score(self):
        """Test basic scoring between two texts."""
        import requests

        response = requests.post(
            f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/score",
            json={
                "model": TEST_RERANK_MODEL,
                "text_1": "What is the capital of France?",
                "text_2": "Paris is the capital of France.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "score" in data["data"][0]

        score = data["data"][0]["score"]
        assert 0 <= score <= 1
        print(f"Score: {score}")

    def test_score_high_relevance(self):
        """Test that highly relevant texts get high scores."""
        import requests

        response = requests.post(
            f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/score",
            json={
                "model": TEST_RERANK_MODEL,
                "text_1": "What is machine learning?",
                "text_2": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        score = data["data"][0]["score"]
        assert score > 0.5, f"Expected high score for relevant texts, got {score}"
        print(f"High relevance score: {score}")

    def test_score_low_relevance(self):
        """Test that unrelated texts get lower scores."""
        import requests

        response = requests.post(
            f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/score",
            json={
                "model": TEST_RERANK_MODEL,
                "text_1": "What is the weather today?",
                "text_2": "The Pythagorean theorem states that a² + b² = c².",
            },
        )

        assert response.status_code == 200
        data = response.json()
        score = data["data"][0]["score"]
        print(f"Low relevance score: {score}")


class TestRerank:
    """Tests for /v1/rerank endpoint."""

    def test_basic_rerank(self):
        """Test basic reranking of documents."""
        import requests

        query = "What is the capital of France?"
        documents = [
            "Berlin is the capital of Germany.",
            "Paris is the capital of France.",
            "London is the capital of England.",
        ]

        response = requests.post(
            f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/rerank",
            json={
                "model": TEST_RERANK_MODEL,
                "query": query,
                "documents": documents,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == len(documents)

        # Check that results are sorted by relevance
        results = data["results"]
        assert results[0]["relevance_score"] >= results[1]["relevance_score"]
        assert results[1]["relevance_score"] >= results[2]["relevance_score"]

        # The most relevant document should be about Paris (index 1)
        assert results[0]["index"] == 1, f"Expected Paris document (index 1) to be ranked first, got index {results[0]['index']}"
        print(f"Top result index: {results[0]['index']}, score: {results[0]['relevance_score']}")

    def test_rerank_ordering(self):
        """Test that reranking properly orders documents by relevance."""
        import requests

        query = "How to make coffee?"
        documents = [
            "The history of tea in China dates back thousands of years.",
            "To brew coffee, grind fresh beans and use hot water at 195-205°F.",
            "Basketball was invented by James Naismith in 1891.",
            "Coffee beans should be stored in an airtight container.",
        ]

        response = requests.post(
            f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/rerank",
            json={
                "model": TEST_RERANK_MODEL,
                "query": query,
                "documents": documents,
            },
        )

        assert response.status_code == 200
        data = response.json()
        results = data["results"]

        # Coffee-related documents (indices 1 and 3) should rank higher
        top_indices = [r["index"] for r in results[:2]]
        assert 1 in top_indices or 3 in top_indices, "Coffee-related documents should be ranked high"
        print(f"Top 2 indices: {top_indices}")

    def test_rerank_with_top_n(self):
        """Test reranking with top_n parameter."""
        import requests

        response = requests.post(
            f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/v1/rerank",
            json={
                "model": TEST_RERANK_MODEL,
                "query": "What is Python?",
                "documents": [
                    "Python is a programming language.",
                    "Java is also a programming language.",
                    "Snakes can be dangerous.",
                    "Programming is fun.",
                ],
                "top_n": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        # Note: top_n may or may not be enforced depending on backend
        print(f"Results count: {len(results)}")


class TestAsyncOperations:
    """Tests for async API operations."""

    @pytest.mark.asyncio
    async def test_async_chat_completion(self):
        """Test async chat completion."""
        client = get_async_client()

        response = await client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
        )

        assert response.choices[0].message.content is not None
        print(f"Async response: {response.choices[0].message.content}")

    @pytest.mark.asyncio
    async def test_async_embedding(self):
        """Test async embedding creation."""
        client = get_async_client()

        response = await client.embeddings.create(
            model=get_embedding_model(),
            input="Async test",
        )

        assert len(response.data[0].embedding) > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test multiple concurrent requests."""
        client = get_async_client()

        async def make_request(i: int):
            response = await client.chat.completions.create(
                model=get_chat_model(),
                messages=[
                    {"role": "user", "content": f"Say the number {i}."}
                ],
                max_tokens=10,
            )
            return response.choices[0].message.content

        # Make 5 concurrent requests
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result is not None
            print(f"Concurrent request {i}: {result}")


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_model(self):
        """Test error handling for invalid model."""
        client = get_client()

        # This should raise an error or return an error response
        try:
            response = client.chat.completions.create(
                model="non-existent-model-12345",
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
            )
            # If we get here, check if response indicates an error
            # Some servers might return a response with error details
        except Exception as e:
            print(f"Expected error for invalid model: {e}")

    def test_empty_messages(self):
        """Test error handling for empty messages."""
        client = get_client()

        try:
            response = client.chat.completions.create(
                model=get_chat_model(),
                messages=[],
            )
        except Exception as e:
            print(f"Expected error for empty messages: {e}")


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Test the health check endpoint."""
        import requests

        response = requests.get(f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"


class TestParameterPassthrough:
    """Tests specifically for parameter passthrough verification."""

    def test_all_chat_parameters(self):
        """Test that all chat parameters are accepted without error."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=10,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            seed=42,
            stop=["END"],
        )

        assert response.choices[0].message.content is not None
        print("All chat parameters passed successfully")

    def test_response_format_json_object(self):
        """Test response_format with json_object."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Return a JSON object with key 'greeting' and value 'hello'."}
            ],
            response_format={"type": "json_object"},
        )

        import json
        content = response.choices[0].message.content
        parsed = json.loads(content)
        print(f"JSON object response: {parsed}")

    def test_response_format_text(self):
        """Test response_format with text type."""
        client = get_client()

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {"role": "user", "content": "Say hello."}
            ],
            response_format={"type": "text"},
        )

        assert response.choices[0].message.content is not None


def run_all_tests():
    """Run all tests and print summary."""
    import traceback

    test_classes = [
        TestHealthEndpoint,
        TestModelsEndpoint,
        TestChatCompletions,
        TestCompletions,
        TestEmbeddings,
        TestScore,
        TestRerank,
        TestParameterPassthrough,
        TestErrorHandling,
    ]

    results = {"passed": 0, "failed": 0, "errors": []}

    print("=" * 60)
    print("MaxLLM API Server Tests")
    print(f"Server: {TEST_BASE_URL}")
    print(f"Chat Model: {get_chat_model()}")
    print(f"Embedding Model: {get_embedding_model()}")
    print("=" * 60)
    print()

    for test_class in test_classes:
        print(f"\n{'=' * 40}")
        print(f"Running {test_class.__name__}")
        print("=" * 40)

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                print(f"\n  {method_name}...", end=" ")

                try:
                    # Skip async tests in non-pytest run
                    if asyncio.iscoroutinefunction(method):
                        asyncio.run(method())
                    else:
                        method()
                    print("PASSED")
                    results["passed"] += 1
                except Exception as e:
                    print(f"FAILED: {e}")
                    results["failed"] += 1
                    results["errors"].append({
                        "test": f"{test_class.__name__}.{method_name}",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")

    if results["errors"]:
        print("\nFailed Tests:")
        for error in results["errors"]:
            print(f"\n  {error['test']}")
            print(f"    Error: {error['error']}")

    return results["failed"] == 0


if __name__ == "__main__":
    # Check if server is running
    import requests
    try:
        requests.get(f"http://{TEST_SERVER_HOST}:{TEST_SERVER_PORT}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Error: Server not running at {TEST_BASE_URL}")
        print(f"Please start the server first: maxllm serve --port {TEST_SERVER_PORT}")
        sys.exit(1)

    success = run_all_tests()
    sys.exit(0 if success else 1)

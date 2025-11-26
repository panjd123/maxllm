"""
OpenAI-compatible API server for maxllm.

This module provides a FastAPI-based server that exposes OpenAI-compatible endpoints,
using the maxllm library for actual model completion.

Endpoints:
- GET /models - List available models
- GET /v1/models - List available models (OpenAI-compatible)
- POST /v1/chat/completions - Chat completions
- POST /v1/completions - Text completions
- POST /v1/embeddings - Embeddings
- POST /v1/rerank - Rerank documents
- POST /v1/score - Score text pairs
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import logging

from ._maxllm import (
    async_openai_complete,
    find_best_model_config,
    _litellm_model_list,
    MockScoreResponse,
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="MaxLLM API Server",
    description="OpenAI-compatible API server powered by maxllm",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class MaxLLMCacheControl(BaseModel):
    """MaxLLM-specific cache control parameters."""
    force: Optional[bool] = None  # bypass cache (no read, no write)
    request_flag: Optional[Any] = None  # custom flag for cache key differentiation


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Accept any additional fields for cache control
    model_config = ConfigDict(extra="allow")


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Accept any additional fields for cache control
    model_config = ConfigDict(extra="allow")


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None
    # Accept any additional fields for cache control
    model_config = ConfigDict(extra="allow")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    # Accept any additional fields for cache control
    model_config = ConfigDict(extra="allow")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    object: str = "list"
    results: List[RerankResult]
    model: str
    usage: Usage


class ScoreRequest(BaseModel):
    model: str
    text_1: Union[str, List[str]]
    text_2: Union[str, List[str]]
    # Accept any additional fields for cache control
    model_config = ConfigDict(extra="allow")


class ScoreData(BaseModel):
    index: int
    score: float


class ScoreResponse(BaseModel):
    object: str = "list"
    data: List[ScoreData]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "maxllm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============== Helper Functions ==============

def generate_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def get_timestamp() -> int:
    return int(time.time())


def get_available_models() -> List[str]:
    """Get list of available model names from config."""
    models = set()
    for item in _litellm_model_list:
        if "model_name" in item:
            models.add(item["model_name"])
    return sorted(list(models))


def extract_cache_control(
    request: BaseModel,
    x_maxllm_force: Optional[str] = None,
    x_maxllm_request_flag: Optional[str] = None,
) -> tuple[bool, Optional[Any]]:
    """
    Extract cache control parameters from request model with robust field detection.

    This function handles multiple ways cache control parameters can be passed:
    1. Direct fields in the request (force, request_flag)
    2. Headers (X-MaxLLM-Force, X-MaxLLM-Request-Flag)
    3. Fields in extra_body (legacy)

    Priority: direct fields > headers > extra_body

    Headers:
        - X-MaxLLM-Force: "true" or "false" (bypass cache)
        - X-MaxLLM-Request-Flag: any string (custom cache key flag)

    Returns:
        (force: bool, request_flag: Optional[Any])
    """
    force = False
    request_flag = None

    # Convert request to dict to access all fields including extra ones
    request_dict = request.model_dump() if hasattr(request, 'model_dump') else dict(request)

    # First check direct fields (highest priority)
    if 'force' in request_dict:
        force = bool(request_dict['force'])
    if 'request_flag' in request_dict:
        request_flag = request_dict['request_flag']

    # Then check headers (lower priority than direct fields)
    if x_maxllm_force is not None and 'force' not in request_dict:
        force = x_maxllm_force.lower() in ("true", "1", "yes")
    if x_maxllm_request_flag is not None and 'request_flag' not in request_dict:
        request_flag = x_maxllm_request_flag

    # Finally check extra_body (legacy, lowest priority)
    if 'extra_body' in request_dict and request_dict['extra_body']:
        extra_body = request_dict['extra_body']

        if 'force' in extra_body and 'force' not in request_dict:
            force = bool(extra_body['force'])
        if 'request_flag' in extra_body and 'request_flag' not in request_dict:
            request_flag = extra_body['request_flag']

    return force, request_flag


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {"message": "MaxLLM API Server", "version": "1.0.0"}


@app.get("/models", response_model=ModelListResponse)
@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    models = get_available_models()
    return ModelListResponse(
        data=[ModelInfo(id=model) for model in models]
    )


@app.get("/models/{model_id}")
@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model's information."""
    config = find_best_model_config(model_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelInfo(id=model_id)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    x_maxllm_force: Optional[str] = Header(None, alias="X-MaxLLM-Force"),
    x_maxllm_request_flag: Optional[str] = Header(None, alias="X-MaxLLM-Request-Flag"),
):
    """Create a chat completion."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")

    # Extract cache control parameters using robust method
    force, request_flag = extract_cache_control(request, x_maxllm_force, x_maxllm_request_flag)

    # Convert messages to dict format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Build kwargs for async_openai_complete
    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = request.max_completion_tokens
    if request.stop is not None:
        kwargs["stop"] = request.stop
    if request.presence_penalty is not None:
        kwargs["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = request.frequency_penalty
    if request.seed is not None:
        kwargs["seed"] = request.seed
    if request.tools is not None:
        kwargs["tools"] = request.tools
    if request.tool_choice is not None:
        kwargs["tool_choice"] = request.tool_choice

    try:
        response = await async_openai_complete(
            model=request.model,
            messages=messages,
            response_format=request.response_format,
            raw=True,
            force=force,
            request_flag=request_flag,
            **kwargs,
        )

        # Extract content from response
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason or "stop"
        else:
            content = str(response)
            finish_reason = "stop"

        # Extract usage
        if hasattr(response, "usage"):
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            # Estimate tokens if not available
            usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return ChatCompletionResponse(
            id=generate_id("chatcmpl"),
            created=get_timestamp(),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    x_maxllm_force: Optional[str] = Header(None, alias="X-MaxLLM-Force"),
    x_maxllm_request_flag: Optional[str] = Header(None, alias="X-MaxLLM-Request-Flag"),
):
    """Create a text completion."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")

    # Extract cache control parameters using robust method
    force, request_flag = extract_cache_control(request, x_maxllm_force, x_maxllm_request_flag)

    # Build kwargs
    kwargs = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.stop is not None:
        kwargs["stop"] = request.stop
    if request.logprobs is not None:
        kwargs["logprobs"] = request.logprobs
    if request.echo:
        kwargs["echo"] = request.echo

    # Handle single or multiple prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

    try:
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx, prompt in enumerate(prompts):
            response = await async_openai_complete(
                model=request.model,
                prompt=prompt,
                call_method="completions",
                raw=True,
                force=force,
                request_flag=request_flag,
                **kwargs,
            )

            if hasattr(response, "choices") and len(response.choices) > 0:
                text = response.choices[0].text
                finish_reason = response.choices[0].finish_reason or "stop"
                logprobs_data = getattr(response.choices[0], "logprobs", None)
            else:
                text = str(response)
                finish_reason = "stop"
                logprobs_data = None

            choices.append(
                CompletionChoice(
                    text=text,
                    index=idx,
                    logprobs=logprobs_data,
                    finish_reason=finish_reason,
                )
            )

            if hasattr(response, "usage"):
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

        return CompletionResponse(
            id=generate_id("cmpl"),
            created=get_timestamp(),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(
    request: EmbeddingRequest,
    x_maxllm_force: Optional[str] = Header(None, alias="X-MaxLLM-Force"),
    x_maxllm_request_flag: Optional[str] = Header(None, alias="X-MaxLLM-Request-Flag"),
):
    """Create embeddings."""
    # Extract cache control parameters using robust method
    force, request_flag = extract_cache_control(request, x_maxllm_force, x_maxllm_request_flag)

    try:
        # Handle single or multiple inputs
        inputs = request.input if isinstance(request.input, list) else [request.input]

        response = await async_openai_complete(
            model=request.model,
            prompt=inputs if len(inputs) > 1 else inputs[0],
            call_method="embeddings",
            raw=True,
            force=force,
            request_flag=request_flag,
        )

        # Extract embeddings from response
        if hasattr(response, "data"):
            embeddings_data = [
                EmbeddingData(
                    embedding=emb.embedding,
                    index=idx,
                )
                for idx, emb in enumerate(response.data)
            ]
        else:
            # Response is already a list of embeddings
            if isinstance(response, list):
                if isinstance(response[0], list):
                    # List of embeddings
                    embeddings_data = [
                        EmbeddingData(embedding=emb, index=idx)
                        for idx, emb in enumerate(response)
                    ]
                else:
                    # Single embedding
                    embeddings_data = [EmbeddingData(embedding=response, index=0)]
            else:
                raise ValueError("Unexpected response format from embedding model")

        # Extract usage
        if hasattr(response, "usage"):
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens,
            )
        else:
            usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return EmbeddingResponse(
            data=embeddings_data,
            model=request.model,
            usage=usage,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    x_maxllm_force: Optional[str] = Header(None, alias="X-MaxLLM-Force"),
    x_maxllm_request_flag: Optional[str] = Header(None, alias="X-MaxLLM-Request-Flag"),
):
    """Rerank documents based on query relevance."""
    # Extract cache control parameters using robust method
    force, request_flag = extract_cache_control(request, x_maxllm_force, x_maxllm_request_flag)

    try:
        kwargs = {}
        if request.top_n is not None:
            kwargs["top_n"] = request.top_n
        if request.return_documents:
            kwargs["return_documents"] = request.return_documents

        response = await async_openai_complete(
            model=request.model,
            query=request.query,
            documents=request.documents,
            call_method="rerank",
            raw=True,
            force=force,
            request_flag=request_flag,
            **kwargs,
        )

        # Parse response - handle MockScoreResponse from cache
        if isinstance(response, MockScoreResponse):
            response = response.data

        if isinstance(response, dict):
            results_data = response.get("results", response.get("data", []))
        else:
            results_data = response

        results = []
        for idx, item in enumerate(results_data):
            if isinstance(item, dict):
                results.append(
                    RerankResult(
                        index=item.get("index", idx),
                        relevance_score=item.get("relevance_score", item.get("score", 0)),
                        document=item.get("document") if request.return_documents else None,
                    )
                )
            else:
                results.append(
                    RerankResult(index=idx, relevance_score=float(item))
                )

        # Estimate usage
        total_tokens = sum(len(doc.split()) for doc in request.documents) + len(request.query.split())

        return RerankResponse(
            results=results,
            model=request.model,
            usage=Usage(prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/score", response_model=ScoreResponse)
@app.post("/score", response_model=ScoreResponse)
async def score(
    request: ScoreRequest,
    x_maxllm_force: Optional[str] = Header(None, alias="X-MaxLLM-Force"),
    x_maxllm_request_flag: Optional[str] = Header(None, alias="X-MaxLLM-Request-Flag"),
):
    """Score text pairs for similarity/relevance (Jina AI compatible)."""
    # Extract cache control parameters using robust method
    force, request_flag = extract_cache_control(request, x_maxllm_force, x_maxllm_request_flag)

    try:
        # Normalize inputs to lists
        text_1_list = request.text_1 if isinstance(request.text_1, list) else [request.text_1]
        text_2_list = request.text_2 if isinstance(request.text_2, list) else [request.text_2]

        # Handle batch requests by making parallel individual calls
        if len(text_1_list) > 1 or len(text_2_list) > 1:
            # Ensure lists are same length (pair-wise scoring)
            if len(text_1_list) != len(text_2_list):
                raise HTTPException(
                    status_code=400,
                    detail=f"text_1 and text_2 must have same length for batch scoring. Got {len(text_1_list)} and {len(text_2_list)}"
                )

            # Make parallel calls for each pair
            async def score_single(idx: int, t1: str, t2: str):
                response = await async_openai_complete(
                    model=request.model,
                    text_1=t1,
                    text_2=t2,
                    call_method="score",
                    raw=True,
                    force=force,
                    request_flag=request_flag,
                )
                # Parse response
                if isinstance(response, MockScoreResponse):
                    response = response.data
                if isinstance(response, dict):
                    data = response.get("data", [])
                    if not data and "score" in response:
                        score_val = response["score"]
                    elif data:
                        score_val = data[0].get("score", data[0].get("similarity", 0))
                    else:
                        score_val = 0.0
                elif isinstance(response, (int, float)):
                    score_val = float(response)
                else:
                    score_val = 0.0
                return ScoreData(index=idx, score=score_val)

            tasks = [score_single(idx, t1, t2) for idx, (t1, t2) in enumerate(zip(text_1_list, text_2_list))]
            score_data = await asyncio.gather(*tasks)
        else:
            # Single pair scoring
            response = await async_openai_complete(
                model=request.model,
                text_1=text_1_list[0],
                text_2=text_2_list[0],
                call_method="score",
                raw=True,
                force=force,
                request_flag=request_flag,
            )

            # Parse response - handle MockScoreResponse from cache
            if isinstance(response, MockScoreResponse):
                response = response.data

            if isinstance(response, dict):
                data = response.get("data", [])
                if not data and "score" in response:
                    data = [{"score": response["score"]}]
            elif isinstance(response, (int, float)):
                data = [{"score": response}]
            elif isinstance(response, list):
                if len(response) > 0 and isinstance(response[0], dict):
                    data = response
                else:
                    data = [{"score": s} for s in response]
            else:
                data = [{"score": float(response)}]

            score_data = [
                ScoreData(index=idx, score=item.get("score", item.get("similarity", 0)))
                for idx, item in enumerate(data)
            ]

        # Estimate usage
        total_tokens = sum(len(t.split()) for t in text_1_list + text_2_list)

        return ScoreResponse(
            data=score_data,
            model=request.model,
            usage=Usage(prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def create_app() -> FastAPI:
    """Create and return the FastAPI app instance."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

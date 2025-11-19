import asyncio
import copy
import time
from tqdm import tqdm
import csv
import os
import datetime
from typing import Any, List, Optional
import tiktoken
from pydantic import BaseModel, ValidationError
from collections import deque
from typing import Literal
import inspect
import os.path as osp
import json
from openai.types import Completion, CompletionUsage
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse as EmbeddingResponse,
)
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.responses import ResponseUsage, Response
import yaml
import requests
from copy import deepcopy

import openai
from openai.lib._parsing._completions import type_to_response_format_param as type_to_response_format

from openai.types import ReasoningEffort
from openai import NotGiven
from aiolimiter import AsyncLimiter
from diskcache import Cache
import pickle
import hashlib

import re
import json
import json5
import json_repair
from typing import Any, List, Optional, Union
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict
import aiohttp

try:
    from jrag.logger import get_module_logger  # type: ignore

    logger = get_module_logger(__file__, __package__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

path = os.getenv("MAXLLM_DOTENV_PATH", None) or find_dotenv(usecwd=True)

if path:
    logger.info(f"Loading .env file from: {path}")
else:
    logger.warning(".env file not found.")
    logger.info(f"Looking for .env file in current directory: {os.getcwd()}")

load_dotenv(path)

MAXLLM_FOLDER = osp.join(osp.expanduser("~"), ".maxllm")
MAXLLM_LOGS_FOLDER = osp.join(MAXLLM_FOLDER, "logs")
MAXLLM_CACHE_FOLDER = osp.join(MAXLLM_FOLDER, "cache")
MAXLLM_BATCH_FOLDER = osp.join(MAXLLM_FOLDER, "batch")
MAXLLM_DEFAULT_CONFIG_FILE = osp.join(MAXLLM_FOLDER, "maxllm.yaml")
os.makedirs(MAXLLM_FOLDER, exist_ok=True)
os.makedirs(MAXLLM_LOGS_FOLDER, exist_ok=True)
os.makedirs(MAXLLM_CACHE_FOLDER, exist_ok=True)
os.makedirs(MAXLLM_BATCH_FOLDER, exist_ok=True)


def _lower_keys(obj):
    if isinstance(obj, dict):
        return {k.lower(): _lower_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_lower_keys(i) for i in obj]
    else:
        return obj

class AnyModel(BaseModel):
    pass

def _parse_json_safely(
    raw_text: str, json_format: Optional[BaseModel] = None
) -> Union[List[Any], dict]:
    """
    简单稳健的 JSON 解析函数。
    支持 JSON → JSON5 → json_repair → block fallback → array fallback。
    """

    def safe_json_parse(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            return json5.loads(text)
        except Exception:
            pass
        try:
            repaired = json_repair.repair_json(text)
            return json.loads(repaired)
        except Exception:
            return {}

    def extract_json_block(text: str) -> str:
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else text

    def extract_array_fallback(text: str) -> List[str]:
        match = re.search(r"\[([\s\S]*?)\]", text)
        if not match:
            return []
        items = re.split(r",\s*", match.group(1))
        return [i.strip(" \"'") for i in items if i.strip()]

    def validate_list(items: Any) -> List[Any]:
        if not isinstance(items, list):
            return []
        cleaned = []
        for i in items:
            if isinstance(i, str) and i.strip():
                cleaned.append(i.strip())
            elif isinstance(i, dict):
                cleaned.append(i)
        return cleaned

    def validate(data: Any, json_format: BaseModel) -> Any:
        if json_format and len(json_format.model_fields) == 1:
            only_key = list(json_format.model_fields.keys())[0]
            if isinstance(data, dict):
                if only_key not in data:
                    data = {only_key: data}
            else:
                data = {only_key: data}
        data = _lower_keys(data)
        return data

    # 去掉 ```json 包裹
    content = re.sub(r"```(?:json5|json)?|```", "", raw_text).strip()

    # 1. 直接解析
    data = safe_json_parse(content)

    if data:
        return validate(data, json_format)

    json_block = extract_json_block(content)
    data = safe_json_parse(json_block)

    if data:
        return validate(data, json_format)

    return {}


def _create_call_status():
    return {
        "total_calls": 0,
        "cache_hit": 0,
        "cache_miss": 0,
        "hit_input_tokens": 0,
        "hit_input_cached_tokens": 0,
        "hit_output_tokens": 0,
        "hit_output_reasoning_tokens": 0,
        "miss_input_tokens": 0,
        "miss_input_cached_tokens": 0,
        "miss_output_tokens": 0,
        "miss_output_reasoning_tokens": 0,
        "validation_errors": 0,
    }


_call_status = defaultdict(_create_call_status)

global_request_flag = False
global_recache_flag = os.getenv("RECACHE_FLAG", "0") in ["1", "true", "True"]


def set_request_flag(flag: Any):
    global global_request_flag
    global_request_flag = flag


# ==============================================================================
# 1. Cost and Logging Setup (Inspired by your code)
# ==============================================================================

# Model prices in USD per 1 Million tokens
MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cache_rate": 0.5},
    "gpt-5": {"input": 1.25, "output": 10, "cache_rate": 0.1},
    "gpt-5-2025-08-07": {"input": 1.25, "output": 10, "cache_rate": 0.1},
    "gpt-5-mini": {"input": 0.25, "output": 2, "cache_rate": 0.1},
    "gpt-5-nano": {"input": 0.05, "output": 0.4, "cache_rate": 0.1},
    "gpt-4.1": {"input": 2, "output": 8, "cache_rate": 0.25},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "cache_rate": 1},
    "glm-4.6": {"input": 0.06, "output": 0.24, "cache_rate": 1},  # fuck onechats
    "glm-4.6-thinking": {
        "input": 0.06,
        "output": 0.24,
        "cache_rate": 1,
    },  # fuck onechats
    "grok-4-fast": {"input": 0.2, "output": 0.5, "cache_rate": 1},  # fuck onechats
    "onechats/deepseek-v3.2-exp-thinking": {
        "input": 0.2,
        "output": 0.3,
        "cache_rate": 1,
    },  # fuck onechats
    "onechats/deepseek-v3.2-exp": {
        "input": 0.2,
        "output": 0.3,
        "cache_rate": 1,
    },  # fuck onechats
    "grok-4-fast-non-reasoning": {"input": 0.2, "output": 0.5, "cache_rate": 1},
    "gpt-oss-120b": {"input": 0.01, "output": 0.05, "cache_rate": 1},
}


class AsyncOpenAICSVLogger:
    """An async-safe CSV logger for OpenAI API calls."""

    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, mode="a", newline="", encoding="utf-8")
        self.fieldnames = [
            "timestamp",
            "model",
            "system_prompt_preview",
            "prompt_preview",
            "completion_preview",
            "input_tokens",
            "output_tokens",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "total_tokens",
            "cost_input_usd",
            "cost_output_usd",
            "cost_total_usd",
            "latency_sec",
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        # Write header only if the file is new/empty
        if self.file.tell() == 0:
            self.writer.writeheader()

    async def async_log(self, row: dict):
        """Asynchronously logs a row to the CSV file."""
        # The lock prevents race conditions from concurrent tasks writing at the same time.
        # Note: The file I/O itself is blocking. For extremely high throughput,
        # a library like 'aiofiles' could be used, but this is sufficient for most cases.
        self.writer.writerow(row)
        self.file.flush()  # Ensure data is written to disk immediately

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()  # Ensure data is written to disk immediately

    def close(self):
        """Closes the file handle."""
        self.file.close()


class SlidingWindowAverage:
    def __init__(self, window_size, estimate_avg=500):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be positive")

        self.window = deque(maxlen=window_size)
        self.estimate_avg = estimate_avg
        self._current_sum = 0.0

    def add(self, value):
        if len(self.window) == self.window.maxlen:
            self._current_sum -= self.window[0]

        self.window.append(value)
        self._current_sum += value

    def average(self):
        if len(self.window) == 0:
            return self.estimate_avg
        avg = self._current_sum / len(self.window)
        return avg


# Create a single, shared logger instance
_global_csv_logger = AsyncOpenAICSVLogger(
    osp.join(MAXLLM_LOGS_FOLDER, "openai_calls.csv")
)

MAX_RETRY = 5
MAX_NO_RATELIMIT_RETRY = 3

RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    ValidationError,
    # openai.APIStatusError,
    # openai.APITimeoutError,
)


def diff_call_status(
    before: dict,
    after: dict,
):
    diff = {}
    for key in before:
        diff[key] = after[key] - before[key]
    return diff


def get_call_status(model: Optional[str] = None):
    global _call_status
    return_status = deepcopy(dict(_call_status))
    
    if model and model not in return_status:
        return f"Not found call status for model: {model}"
    elif model:
        return_status = {model: return_status[model]}
        
    for model_name in return_status:
        return_status[model_name]["cost_usd"] = 0.0
        return_status[model_name]["cached_usd"] = 0.0
        model_price = MODEL_PRICES.get(
            model_name, {"input": 0.0, "output": 0.0, "cache_rate": 0.1}
        )
        return_status[model_name]["cost_usd"] += (
            (
                return_status[model_name]["miss_input_tokens"]
                - return_status[model_name]["miss_input_cached_tokens"]
            )
            / 1_000_000
        ) * model_price["input"]
        return_status[model_name]["cost_usd"] += (
            ((return_status[model_name]["miss_input_cached_tokens"]) / 1_000_000)
            * model_price["input"]
            * model_price["cache_rate"]
        )
        return_status[model_name]["cost_usd"] += (
            return_status[model_name]["miss_output_tokens"] / 1_000_000
        ) * model_price["output"]

        return_status[model_name]["cached_usd"] += (
            (
                return_status[model_name]["hit_input_tokens"]
                - return_status[model_name]["hit_input_cached_tokens"]
            )
            / 1_000_000
        ) * model_price["input"]
        return_status[model_name]["cached_usd"] += (
            ((return_status[model_name]["hit_input_cached_tokens"]) / 1_000_000)
            * model_price["input"]
            * model_price["cache_rate"]
        )
        return_status[model_name]["cached_usd"] += (
            return_status[model_name]["hit_output_tokens"] / 1_000_000
        ) * model_price["output"]
        
    if model:
        return json.dumps(return_status[model], indent=2, ensure_ascii=False)
    else:
        return json.dumps(return_status, indent=2, ensure_ascii=False)


def _get_response_tokens(response):
    usage: ResponseUsage | CompletionUsage | EmbeddingUsage = response.usage
    if isinstance(usage, ResponseUsage):
        input_tokens = usage.input_tokens
        input_cached_tokens = usage.input_tokens_details.cached_tokens
        output_tokens = usage.output_tokens
        output_reasoning_tokens = usage.output_tokens_details.reasoning_tokens
    elif isinstance(usage, CompletionUsage):
        input_tokens = usage.prompt_tokens
        input_cached_tokens = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details
            else 0
        )
        output_tokens = usage.completion_tokens
        output_reasoning_tokens = (
            usage.completion_tokens_details.reasoning_tokens
            if usage.completion_tokens_details
            else 0
        )
    elif isinstance(usage, EmbeddingUsage):
        input_tokens = usage.prompt_tokens
        input_cached_tokens = 0
        output_tokens = 0
        output_reasoning_tokens = 0
    else:
        logger.error(f"Unknown usage type: {type(usage)}")
    return (
        input_tokens,
        input_cached_tokens,
        output_tokens,
        output_reasoning_tokens,
        usage.total_tokens,
    )


# Find config file in the following order:
# 1. Environment variable MAXLLM_CONFIG_PATH
# 2. Current working directory (maxllm.yaml)
# 3. Default config file path (~/.maxllm/maxllm.yaml)
def get_maxllm_config_path():
    if os.environ.get("MAXLLM_CONFIG_PATH"):
        return os.environ.get("MAXLLM_CONFIG_PATH")
    elif osp.exists("maxllm.yaml"):
        return "maxllm.yaml"
    elif osp.exists(MAXLLM_DEFAULT_CONFIG_FILE):
        return MAXLLM_DEFAULT_CONFIG_FILE
    else:
        return None

_config_path = get_maxllm_config_path()

# Try to load config file, use defaults if not found
if _config_path and osp.exists(_config_path):
    with open(_config_path, "r") as f:
        _litellm_config = yaml.safe_load(f)
    logger.info(f"Loaded config from: {_config_path}")
else:
    logger.warning(
        "Config file not found. Using default configuration. "
        "Create 'maxllm.yaml' in current directory or set MAXLLM_CONFIG_PATH environment variable to customize."
    )
    _litellm_config = {}

_litellm_model_list = _litellm_config.get(
    "model_list",
    [
        {
            "model_name": "*",
            "litellm_params": {
                "model": "*",
                "api_base": os.environ.get("OPENAI_API_BASE"),
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
        }
    ],
)

_rate_limit_config = _litellm_config.get(
    "rate_limit", {"default": [{"model_name": "*", "rpm": 500, "tpm": 200000}]}
)


def _match_score(pattern: str, target: str) -> int:
    pattern_escaped = re.escape(pattern).replace("\\*", "(.*)")
    if re.fullmatch(pattern_escaped, target):
        score = len(pattern) - pattern.count("*") * 2
        return score
    return -999


def _find_best_match(target_name: str, model_list):
    best_item = None
    best_score = -99
    for item in model_list:
        if "model_name" not in item:
            continue
        score = _match_score(item["model_name"], target_name)
        if score > best_score:
            best_score = score
            best_item = copy.deepcopy(item)
    return best_item


def get_rate_limit(
    model: str,
    rpm: int = None,
    tpm: int = None,
    is_local_model=False,
    is_embedding_model=None,
    rate_limit_config=_rate_limit_config,
):
    """
    get the rate limit (requests per minute and tokens per minute) for a given model.
    If rpm or tpm is given, use the minimum of the given value and the configured upper-bound value.
    """
    # config_rpm, config_tpm

    defaults = rate_limit_config.get("default", [])
    custom_limits = rate_limit_config.get("custom_limits", [])
    best_item = _find_best_match(model, custom_limits)
    if not best_item:
        logger.warning(f"No custom rate limit found for model {model}, using default.")
        wildcard_name = ""
        if is_local_model:
            wildcard_name = "local_"
        if is_embedding_model:
            wildcard_name += "embedding"
        else:
            wildcard_name += "completion"
        best_item = _find_best_match(wildcard_name, defaults)

    assert best_item is not None, f"No rate limit config found for model {model}"

    config_rpm_upper_bound = best_item.get("rpm_upper_bound", 10000000)  # 10M
    config_rpm = best_item.get("rpm", 500)
    config_tpm_upper_bound = best_item.get("tpm_upper_bound", 10000000)  # 10M
    config_tpm = best_item.get("tpm", 30000)
    if rpm is not None:
        rpm = min(rpm, config_rpm_upper_bound)
    else:
        rpm = config_rpm
    if tpm is not None:
        tpm = min(tpm, config_tpm_upper_bound)
    else:
        tpm = config_tpm

    return rpm, tpm


def find_best_model_config(target_name: str, model_list: list = _litellm_model_list):
    best_item = _find_best_match(target_name, model_list)
    if best_item["litellm_params"]["model"] is None:
        best_item["litellm_params"]["model"] = target_name
    if best_item and best_item["litellm_params"].get("model", "").count("*") > 0:
        if best_item["model_name"].count("*") == best_item["litellm_params"].get(
            "model", ""
        ).count("*"):
            pattern = re.escape(best_item["model_name"]).replace("\\*", "(.*)")
            match = re.fullmatch(pattern, target_name)
            groups = match.groups()

            new_model = best_item["litellm_params"]["model"]
            if new_model:
                for g in groups:
                    new_model = new_model.replace("*", g, 1)
            else:
                new_model = target_name
            best_item["litellm_params"]["model"] = new_model

            new_unique_name = best_item["litellm_params"].get(
                "model_unique_name", new_model
            )
            for g in groups:
                new_unique_name = new_unique_name.replace("*", g, 1)
            best_item["litellm_params"]["model_unique_name"] = new_unique_name

            logger.info(
                f"Resolved model name for {target_name}: request_model_name: {new_model}, unique_model_name: {new_unique_name}"
            )
        else:
            logger.error(
                f"Model name pattern and litellm_params.model pattern star count mismatch for {best_item}"
            )
    return best_item


def _get_json_compatibility(model_name: str) -> bool:
    json_mode_compatible = True
    json_format_compatible = True
    return json_mode_compatible, json_format_compatible


import tiktoken
from transformers import AutoTokenizer


def _get_encoding(model_name: str):
    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except KeyError:
        try:
            encoder = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            logger.warning(f"Using default encoding for model {model_name} due to error loading tokenizer.")
        encoder = tiktoken.get_encoding("cl100k_base")
    return encoder


class AutoLoopBigSemaphore:
    def __init__(self, value: int):
        self._init_value = value
        self._loop = None
        self._value = value
        self._waiters = []  # list of (need, future)
        self._warning_reuse = False

    def _ensure_loop(self):
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
            return

        if self._loop.is_closed():
            self._loop = asyncio.get_event_loop()
            self._waiters = []
            if not self._warning_reuse:
                logger.warning(
                    f"BigSemaphore reused across loops; reset value from {self._value} to {self._init_value}"
                )
                self._warning_reuse = True
            self._value = self._init_value

    async def acquire(self, n=1):
        if n <= 0:
            return
        self._ensure_loop()

        fut = self._loop.create_future()
        self._waiters.append((n, fut))
        self._try_release_waiters()
        await fut

    def release(self, n=1):
        if n <= 0:
            return
        self._ensure_loop()
        self._value += n
        self._try_release_waiters()

    def _try_release_waiters(self):
        i = 0
        while i < len(self._waiters):
            need, fut = self._waiters[i]
            if fut.done():
                self._waiters.pop(i)
                continue
            if self._value >= need:
                self._value -= need
                fut.set_result(True)
                self._waiters.pop(i)
            else:
                i += 1

    async def __aenter__(self):
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release(1)

    # --------- 可传参上下文管理器 ---------

    class _Ctx:
        def __init__(self, sem, n):
            self.sem = sem
            self.n = n

        async def __aenter__(self):
            await self.sem.acquire(self.n)

        async def __aexit__(self, exc_type, exc, tb):
            self.sem.release(self.n)

    def use(self, n=1):
        return AutoLoopBigSemaphore._Ctx(self, n)

class VLLMSleepModeManager:
    vllm_api_base: str
    def __init__(self, vllm_api_base: str):
        self.vllm_api_base = vllm_api_base
        
    def is_sleep(self) -> bool:
        url = osp.join(self.vllm_api_base, "is_sleeping")
        try:
            with requests.get(url) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("is_sleeping", False)
                else:
                    text = resp.text
                    logger.error(
                        f"VLLM model is_sleeping request failed: {resp.status_code} - {text}"
                    )
        except Exception as e:
            logger.error(f"VLLM model is_sleeping request error: {e}")
        return False
    
    async def ais_sleep(self) -> bool:
        url = osp.join(self.vllm_api_base, "is_sleeping")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("is_sleeping", False)
                    else:
                        text = await resp.text()
                        logger.error(
                            f"VLLM model is_sleeping request failed: {resp.status} - {text}"
                        )
        except Exception as e:
            logger.error(f"VLLM model is_sleeping request error: {e}")
        return False
    
    def sleep(self):
        url = osp.join(self.vllm_api_base, "sleep?level=1")
        try:
            with requests.post(url) as resp:
                text = resp.text
                logger.info(
                    f"VLLM model sleep response: {resp.status_code} - {text}"
                )
        except Exception as e:
            logger.error(f"VLLM model sleep error: {e}")
            
    async def asleep(self):
        url = osp.join(self.vllm_api_base, "sleep?level=1")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as resp:
                    text = await resp.text()
                    logger.info(
                        f"VLLM model sleep response: {resp.status} - {text}"
                    )
        except Exception as e:
            logger.error(f"VLLM model sleep error: {e}")
    
    def wake_up(self):
        url = osp.join(self.vllm_api_base, "wake_up")
        try:
            with requests.post(url) as resp:
                text = resp.text
                logger.info(
                    f"VLLM model wake_up response: {resp.status_code} - {text}"
                )
        except Exception as e:
            logger.error(f"VLLM model wake_up error: {e}")
            
    async def awake_up(self):
        url = osp.join(self.vllm_api_base, "wake_up")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as resp:
                    text = await resp.text()
                    logger.info(
                        f"VLLM model wake_up response: {resp.status} - {text}"
                    )
        except Exception as e:
            logger.error(f"VLLM model wake_up error: {e}")
            
    def auto_wake_up(self):
        is_sleep = self.is_sleep()
        if is_sleep:
            self.wake_up()

    async def aauto_wake_up(self):
        is_sleep = await self.ais_sleep()
        if is_sleep:
            await self.awake_up()

class RateLimitCompleter:
    is_embedding_model: bool
    openai_parse_compatible: bool
    json_format_compatible: bool
    is_local_model: bool
    enable_call_status: bool
    api_base: str
    api_key: str
    vllm_api_base: str
    model: str
    vllm_sleep_mode_manager: VLLMSleepModeManager

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        rpm: int = None,
        tpm: int = None,
        concurrency: Optional[int] = 0,
        hard_tpm_ratio=0.8,
        lazy_wake_up=False,
    ):
        best_model_config = find_best_model_config(model_name)

        if best_model_config is not None:
            api_key = best_model_config["litellm_params"].get("api_key", None)
            if api_key.startswith("os.environ/"):  # os.environ/BIANXIE_API_KEY
                env_var = api_key.split("/", 1)[1]
                api_key = os.environ.get(env_var, None)
            api_base = best_model_config["litellm_params"].get(
                "api_base", os.environ.get("OPENAI_API_BASE", None)
            )
            rpm = best_model_config["litellm_params"].get("rpm", rpm)
            tpm = best_model_config["litellm_params"].get("tpm", tpm)
            kvcache_tokens = best_model_config["litellm_params"].get(
                "kvcache_tokens", 1e8
            )  # 默认不限制
            model = best_model_config["litellm_params"].get("model", model_name)
            model_unique_name = best_model_config["litellm_params"].get(
                "model_unique_name", model
            )
            concurrency = best_model_config["litellm_params"].get(
                "concurrency", concurrency
            )
            is_local_model = best_model_config["litellm_params"].get("local", False)
            logger.info(f"Using model config for {model_name}: {best_model_config}")
        else:
            model = model_name
            logger.warning(
                f"No matching model config found for {model_name}, using default OPENAI_API_BASE and OPENAI_API_KEY."
            )
            api_key = os.environ.get("OPENAI_API_KEY", None)
            api_base = os.environ.get("OPENAI_API_BASE", None)
            is_local_model = False
            kvcache_tokens = 1e8  # 默认不限制

        self.api_key = api_key
        self.api_base = api_base

        self.is_embedding_model = "embedding" in model.lower()

        self.openai_parse_compatible = False  # 和当前 cache 逻辑冲突，禁用

        self.json_mode_compatible, self.json_format_compatible = (
            _get_json_compatibility(model_unique_name)
        )

        self.is_local_model = is_local_model

        if self.is_local_model:
            vllm_api_base = api_base.replace("/v1", "")
            self.vllm_sleep_mode_manager = VLLMSleepModeManager(vllm_api_base)
            if not lazy_wake_up:
                self.vllm_sleep_mode_manager.auto_wake_up()

        self.enable_call_status = True

        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        if concurrency == 0:
            if self.is_embedding_model:
                self.semaphore = AutoLoopBigSemaphore(1000)
            else:
                self.semaphore = AutoLoopBigSemaphore(500)
        else:
            if concurrency is None:
                self.semaphore = AutoLoopBigSemaphore(99999)
            else:
                self.semaphore = AutoLoopBigSemaphore(concurrency)
        no_safe = os.environ.get("NO_SAFE", None) in ["1", "true", "True"]
        if no_safe:
            logger.warning("Rate limit set to NO_SAFE mode.")

        rpm, tpm = get_rate_limit(
            model_unique_name, rpm, tpm, is_local_model, self.is_embedding_model
        )

        logger.info(f"Rate limit for model {model_unique_name}: {rpm} RPM, {tpm} TPM")

        self.model = model
        self.rpm = rpm
        self.tpm = tpm
        self.kvcache_tokens = kvcache_tokens
        self.rpm_limiter = AsyncLimiter(rpm, 60.0)
        self.soft_tpm_limiter = AsyncLimiter(tpm, 60.0)
        self.kvcache_token_semaphore = AutoLoopBigSemaphore(int(0.98 * kvcache_tokens))  # 0.98
        self.tpm_limiter = AsyncLimiter(tpm * hard_tpm_ratio, 60.0)
        self.encoding = _get_encoding(model)
        self.input_tokens_delta_deque = SlidingWindowAverage(
            window_size=30, estimate_avg=100
        )
        self.output_tokens_deque = SlidingWindowAverage(
            window_size=30, estimate_avg=500
        )
        self.token_limit_reached = False
        self.request_limit_reached = False
        self.cache_dir = osp.join(MAXLLM_CACHE_FOLDER, model_unique_name)
        self.model_unique_name = model_unique_name
        os.makedirs(self.cache_dir, exist_ok=True)
        CACHE_SIZE_LIMIT = 16 * (2**30)  # 16GB
        self.cache = Cache(self.cache_dir, size_limit=CACHE_SIZE_LIMIT)  # 16GB
        cache_volume = self.cache.volume()
        if cache_volume > 0.5 * CACHE_SIZE_LIMIT:
            logger.error(
                f"Cache size for model {model_unique_name} exceeds 50% of limit ({CACHE_SIZE_LIMIT/(2**30)} GB). Consider increasing or cleaning up."
            )
            exit(0)
        elif cache_volume > 0.2 * CACHE_SIZE_LIMIT:
            logger.info(
                f"Cache size for model {model_unique_name}: {cache_volume/(2**30):.2f} GB. Consider cleaning up if it exceeds 50% of limit ({CACHE_SIZE_LIMIT/(2**30)} GB)."
            )

    def estimate_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def log_rate_limit_details(
        self, exception, num_attempt
    ) -> Literal["requests", "tokens", "requests/tokens"]:
        try:
            error_details = exception.response.json().get("error", {})
            message = error_details.get("message", "No message available")
            err_type = error_details.get("type", "Unknown type")
            logger.warning(
                f"RateLimitError triggered."
                f" - Model: {self.model}"
                f" - Type: {err_type}"
                f" - Message: {message}"
                f" - Retry {num_attempt}"
            )
            return err_type
        except Exception:
            logger.warning(
                f"RateLimitError triggered, but failed to parse details."
                f" - Retry {num_attempt}"
            )
            return "requests/tokens"

    async def async_complete(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: List[dict] = [],
        messages: List[dict] = [],
        json_mode: bool = False,
        json_format: Optional[BaseModel] = None,
        call_method: Literal[
            "auto", "chat.completions", "embeddings", "responses.parse", "completions"
        ] = "auto",
        raw: bool = False,
        force: bool = False,
        request_flag: Any = None,
        **kwargs,
    ) -> Any:
        async with self.semaphore:
            global _call_status
            model_call_status = _call_status[self.model_unique_name]
            if self.enable_call_status:
                model_call_status["total_calls"] += 1

            # 1. Construct the messages list from inputs
            if self.is_embedding_model:
                if not prompt:
                    for key in ["input", "inputs", "text", "texts"]:
                        if key in kwargs:
                            prompt = kwargs.pop(key)
                            break

            if not prompt and not messages:
                raise ValueError("'prompt' or 'messages' must be provided.")

            # 2. Construct the messages list from inputs
            if not messages:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})
            else:
                if prompt or system_prompt or history_messages:
                    logger.warning(
                        "'messages' is provided, 'prompt', 'system_prompt', and 'history_messages' will be ignored."
                    )

            # 2.1 Construct all params for caching
            json_format_schema = (
                json_format.model_json_schema()
                if inspect.isclass(json_format) and issubclass(json_format, BaseModel)
                else json_format
            )

            if call_method == "auto":
                if json_format and self.openai_parse_compatible:
                    call_method = "responses.parse"
                elif not self.is_embedding_model:
                    if "logprobs" in kwargs or "echo" in kwargs:
                        call_method = "completions"
                    else:
                        call_method = "chat.completions"
                else:
                    call_method = "embeddings"

            cache_kwargs = kwargs.copy()
            cache_params = (
                messages,
                json_mode,
                json_format_schema,
                call_method,
                tuple(sorted(cache_kwargs.items())),
            )
            request_flag = request_flag or global_request_flag
            if request_flag:
                cache_params += (request_flag,)

            no_read_cache_flag = global_recache_flag or force
            no_write_cache_flag = force

            if no_read_cache_flag and no_write_cache_flag:
                cache_key = None
            else:
                cache_key = hashlib.md5(pickle.dumps(cache_params)).hexdigest()

            if not no_read_cache_flag and cache_key in self.cache:
                cached_response = self.cache[cache_key]
                (
                    input_tokens,
                    input_cached_tokens,
                    output_tokens,
                    output_reasoning_tokens,
                    _,
                ) = _get_response_tokens(cached_response)
                if self.enable_call_status:
                    model_call_status["cache_hit"] += 1
                    model_call_status["hit_input_tokens"] += input_tokens
                    model_call_status["hit_input_cached_tokens"] += input_cached_tokens
                    model_call_status["hit_output_tokens"] += output_tokens
                    model_call_status[
                        "hit_output_reasoning_tokens"
                    ] += output_reasoning_tokens

                if raw:
                    return cached_response

                if json_format and self.openai_parse_compatible:
                    content = cached_response.output_parsed.model_dump()
                elif not self.is_embedding_model:  # chat.completions
                    if getattr(cached_response, "choices", None):
                        content = cached_response.choices[0].message.content
                    else:
                        choices = cached_response.data.get("choices")
                        if choices:
                            content = choices[0]["message"]["content"]
                    if json_format:
                        content = _parse_json_safely(content, json_format=json_format)
                    # completions will also fall here
                else:
                    content = (
                        [emb.embedding for emb in cached_response.data]
                        if isinstance(prompt, list)
                        else cached_response.data[0].embedding
                    )
                return content

            # 3. rate limiting

            # 3.1 request limiting
            if self.is_embedding_model:
                if isinstance(prompt, list):
                    estimate_input_tokens_base = sum(
                        self.estimate_tokens(p) for p in prompt
                    )
                else:
                    estimate_input_tokens_base = self.estimate_tokens(prompt)
            else:
                prompt_text = " ".join(msg.get("content", "") for msg in messages)
                estimate_input_tokens_base = self.estimate_tokens(
                    prompt_text
                ) + 3 * len(messages)

            await self.rpm_limiter.acquire(1)

            if self.rpm_limiter._level + 1 > self.rpm_limiter.max_rate * 0.9:
                if not self.request_limit_reached:
                    self.request_limit_reached = True
                    # logger.info(
                    #     f"Request limit reached: {self.rpm_limiter._level} / {self.rpm_limiter.max_rate}"
                    # )
            elif self.rpm_limiter._level + 1 < self.rpm_limiter.max_rate * 0.5:
                if self.request_limit_reached:
                    self.request_limit_reached = False
                    # logger.info(
                    #     f"Request limit no longer reached: {self.rpm_limiter._level} / {self.rpm_limiter.max_rate}"
                    # )

            # 3.2 token limiting
            estimate_input_tokens = estimate_input_tokens_base
            estimate_output_tokens = 100
            tokens_needed = (
                estimate_input_tokens + estimate_output_tokens
            )  # output tokens

            await self.soft_tpm_limiter.acquire(tokens_needed)

            estimate_input_tokens = (
                estimate_input_tokens_base + self.input_tokens_delta_deque.average()
            )
            estimate_output_tokens = self.output_tokens_deque.average()
            tokens_needed = (
                estimate_input_tokens + estimate_output_tokens
            )  # output tokens

            await self.tpm_limiter.acquire(tokens_needed)

            if (
                self.tpm_limiter._level + tokens_needed
                > self.tpm_limiter.max_rate * 0.9
            ):
                if not self.token_limit_reached:
                    self.token_limit_reached = True
                    # logger.info(
                    #     f"Token limit reached: {self.tpm_limiter._level} / {self.tpm_limiter.max_rate}"
                    # )
            elif (
                self.tpm_limiter._level + tokens_needed
                < self.tpm_limiter.max_rate * 0.5
            ):
                if self.token_limit_reached:
                    self.token_limit_reached = False
                    # logger.info(
                    #     f"Token limit no longer reached: {self.tpm_limiter._level} / {self.tpm_limiter.max_rate}"
                    # )


            # 3.2.1 kvcache level token limit
            # await self.kvcache_token_semaphore.acquire(tokens_needed)
            async with self.kvcache_token_semaphore.use(tokens_needed):
                # 4. Call the API (retry logic)
                for num_attempt in range(1, MAX_RETRY + 1):
                    tic = time.monotonic()
                    rate_limit_type = ""
                    try:
                        if json_format and self.openai_parse_compatible:
                            response = await self.async_client.responses.parse(
                                model=self.model,
                                input=messages,
                                text_format=json_format,
                                **kwargs,
                            )
                        elif not self.is_embedding_model:
                            if json_mode:
                                if self.json_mode_compatible:
                                    kwargs["response_format"] = {"type": "json_object"}
                                else:
                                    pass # DO NOTHING
                                # elif self.json_format_compatible:
                                #     kwargs["response_format"] = type_to_response_format(
                                #         AnyModel
                                #     )
                            elif json_format:
                                if self.json_format_compatible:
                                    kwargs["response_format"] = type_to_response_format(
                                        json_format
                                    )
                                elif self.json_mode_compatible:
                                    kwargs["response_format"] = {"type": "json_object"}
                                else:
                                    pass  # kwargs["response_format"] = {"type": "text"}
                            if "max_output_tokens" in kwargs:
                                kwargs["max_tokens"] = kwargs.pop("max_output_tokens")

                            if call_method == "completions":
                                response = await self.async_client.completions.create(
                                    model=self.model, prompt=prompt, **kwargs
                                )
                            else:
                                response = (
                                    await self.async_client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        **kwargs,
                                    )
                                )
                        else:
                            response = await self.async_client.embeddings.create(
                                model=self.model,
                                input=prompt,
                                **kwargs,
                            )

                        if raw:
                            content = response
                        elif json_format and self.openai_parse_compatible:
                            content = response.output_parsed.model_dump()
                        elif not self.is_embedding_model:  # chat.completions
                            if getattr(response, "choices", None):
                                content = response.choices[0].message.content
                            elif isinstance(getattr(response, "data", None), dict):
                                choices = response.data.get("choices")
                                if choices:
                                    content = choices[0]["message"]["content"]
                            if json_format:
                                content = _parse_json_safely(
                                    content, json_format=json_format
                                )
                                json_format.model_validate(content)
                        else:
                            content = (
                                [emb.embedding for emb in response.data]
                                if isinstance(prompt, list)
                                else response.data[0].embedding
                            )

                    except openai.RateLimitError as e:
                        if num_attempt >= MAX_RETRY:
                            raise e
                        rate_limit_type = self.log_rate_limit_details(e, num_attempt)
                    except RETRYABLE_EXCEPTIONS as e:
                        if num_attempt >= MAX_NO_RATELIMIT_RETRY:
                            raise e
                        if e is not ValidationError:
                            logger.warning(
                                f"Retryable error: {type(e).__name__} - {e} - Retry {num_attempt}"
                            )
                        else:
                            _call_status[self.model_unique_name][
                                "validation_errors"
                            ] += 1
                    except Exception as e:
                        # logger.error(f"Non-retryable error: {type(e).__name__} - {e}")
                        raise e
                    else:
                        break

                    # rate limit backoff
                    if "requests" in rate_limit_type:
                        self.rpm_limiter._level += (2**num_attempt) - 1
                    if "tokens" in rate_limit_type:
                        self.tpm_limiter._level += (
                            (2**num_attempt) - 1
                        ) * tokens_needed

                    # retry after backoff
                    await self.rpm_limiter.acquire(1)
                    await self.tpm_limiter.acquire(tokens_needed)
                    # await self.kvcache_token_semaphore.acquire(tokens_needed) # already acquired

            toc = time.monotonic()
            latency = toc - tic

            (
                input_tokens,
                input_cached_tokens,
                output_tokens,
                output_reasoning_tokens,
                total_tokens,
            ) = _get_response_tokens(response)

            # Calculate cost
            price_info = MODEL_PRICES.get(self.model, {"input": 0, "output": 0})
            cost_input = (input_tokens / 1_000_000) * price_info["input"]
            cost_output = (output_tokens / 1_000_000) * price_info["output"]
            cost_total = cost_input + cost_output

            self.input_tokens_delta_deque.add(input_tokens - estimate_input_tokens)
            self.output_tokens_deque.add(output_tokens)

            prompt_preview = prompt or ""
            if isinstance(prompt_preview, list):
                prompt_preview = prompt_preview[0] if len(prompt_preview) > 0 else ""

            if self.enable_call_status:
                model_call_status["cache_miss"] += 1
                model_call_status["miss_input_tokens"] += input_tokens
                model_call_status["miss_input_cached_tokens"] += input_cached_tokens
                model_call_status["miss_output_tokens"] += output_tokens
                model_call_status[
                    "miss_output_reasoning_tokens"
                ] += output_reasoning_tokens

            # Prepare log row
            log_row = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": self.model,
                "system_prompt_preview": (system_prompt or "").replace("\n", " ")[:80],
                "prompt_preview": prompt_preview.replace("\n", " ")[:80],
                "completion_preview": (str(content) or "").replace("\n", " ")[:80],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_input_tokens": f"{estimate_input_tokens}",
                "estimated_output_tokens": f"{estimate_output_tokens}",
                "total_tokens": total_tokens,
                "cost_input_usd": round(cost_input, 6),
                "cost_output_usd": round(cost_output, 6),
                "cost_total_usd": round(cost_total, 6),
                "latency_sec": round(latency, 4),
            }

            await _global_csv_logger.async_log(log_row)

            # 5. Cache the response
            if not no_write_cache_flag:
                self.cache[cache_key] = response

        return content

    def complete(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: List[dict] = [],
        messages: List[dict] = [],
        json_mode: bool = False,
        json_format: Optional[BaseModel] = None,
        call_method: Literal[
            "auto", "chat.completions", "embeddings", "responses.parse", "completions"
        ] = "auto",
        raw: bool = False,
        force: bool = False,
        request_flag: Any = None,
        **kwargs,
    ) -> Any:
        global _call_status
        model_call_status = _call_status[self.model_unique_name]

        if self.enable_call_status:
            model_call_status["total_calls"] += 1

        # 1. Construct the messages list from inputs
        if self.is_embedding_model:
            if not prompt:
                for key in ["input", "inputs", "text", "texts"]:
                    if key in kwargs:
                        prompt = kwargs.pop(key)
                        break

        if self.is_local_model:
            kwargs["timeout"] = 99999  # no timeout for local model

        if not prompt and not messages:
            raise ValueError("'prompt' or 'messages' must be provided.")

        # 2. Construct the messages list from inputs
        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
        elif prompt or system_prompt or history_messages:
            logger.warning(
                "'messages' is provided, 'prompt', 'system_prompt', and 'history_messages' will be ignored."
            )

        # 2.1 Construct all params for caching
        json_format_schema = (
            json_format.model_json_schema()
            if inspect.isclass(json_format) and issubclass(json_format, BaseModel)
            else json_format
        )

        if call_method == "auto":
            if json_format and self.openai_parse_compatible:
                call_method = "responses.parse"
            elif not self.is_embedding_model:
                if "logprobs" in kwargs or "echo" in kwargs:
                    call_method = "completions"
                else:
                    call_method = "chat.completions"
            else:
                call_method = "embeddings"

        cache_kwargs = kwargs.copy()
        kwargs.pop("raw", None)
        cache_params = (
            messages,
            json_mode,
            json_format_schema,
            call_method,
            tuple(sorted(cache_kwargs.items())),
        )
        request_flag = request_flag or global_request_flag
        if request_flag:
            cache_params += (request_flag,)

        no_read_cache_flag = global_recache_flag or force
        no_write_cache_flag = force

        if no_read_cache_flag and no_write_cache_flag:
            cache_key = None
        else:
            cache_key = hashlib.md5(pickle.dumps(cache_params)).hexdigest()

        if not no_read_cache_flag and cache_key in self.cache:
            cached_response = self.cache[cache_key]
            (
                input_tokens,
                input_cached_tokens,
                output_tokens,
                output_reasoning_tokens,
                _,
            ) = _get_response_tokens(cached_response)
            if self.enable_call_status:
                model_call_status["cache_hit"] += 1
                model_call_status["hit_input_tokens"] += input_tokens
                model_call_status["hit_input_cached_tokens"] += input_cached_tokens
                model_call_status["hit_output_tokens"] += output_tokens
                model_call_status[
                    "hit_output_reasoning_tokens"
                ] += output_reasoning_tokens

            if raw:
                return cached_response

            if json_format and self.openai_parse_compatible:
                content = cached_response.output_parsed.model_dump()
            elif not self.is_embedding_model:  # chat.completions
                content = cached_response.choices[0].message.content
                if json_format:
                    content = _parse_json_safely(content, json_format=json_format)
                # completions will also fall here
            else:
                content = (
                    [emb.embedding for emb in cached_response.data]
                    if isinstance(prompt, list)
                    else cached_response.data[0].embedding
                )
            return content

        # 3. rate limiting

        # 3.1 request limiting
        if self.is_embedding_model:
            if isinstance(prompt, list):
                estimate_input_tokens_base = sum(
                    self.estimate_tokens(p) for p in prompt
                )
            else:
                estimate_input_tokens_base = self.estimate_tokens(prompt)
        else:
            prompt_text = " ".join(msg.get("content", "") for msg in messages)
            estimate_input_tokens_base = self.estimate_tokens(prompt_text) + 3 * len(
                messages
            )

        # 3.2 token limiting
        estimate_input_tokens = estimate_input_tokens_base
        estimate_output_tokens = 100

        estimate_input_tokens = (
            estimate_input_tokens_base + self.input_tokens_delta_deque.average()
        )
        estimate_output_tokens = self.output_tokens_deque.average()

        # 4. Call the API (retry logic)
        for num_attempt in range(1, MAX_RETRY + 1):
            tic = time.monotonic()
            try:
                if json_format and self.openai_parse_compatible:
                    response = self.client.responses.parse(
                        model=self.model,
                        input=messages,
                        text_format=json_format,
                        **kwargs,
                    )
                elif not self.is_embedding_model:
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    elif json_format:
                        if self.json_format_compatible:
                            kwargs["response_format"] = type_to_response_format(
                                json_format
                            )
                        else:
                            kwargs["response_format"] = {"type": "json_object"}
                    if "max_output_tokens" in kwargs:
                        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")

                    if call_method == "completions":
                        response = self.client.completions.create(
                            model=self.model, prompt=prompt, **kwargs
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **kwargs,
                        )
                else:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=prompt,
                        **kwargs,
                    )

                if raw:
                    content = response
                elif json_format and self.openai_parse_compatible:
                    content = response.output_parsed.model_dump()
                elif not self.is_embedding_model:
                    content = response.choices[0].message.content
                    if json_format:
                        content = _parse_json_safely(content, json_format=json_format)
                        json_format.model_validate(content)
                else:
                    content = (
                        [emb.embedding for emb in response.data]
                        if isinstance(prompt, list)
                        else response.data[0].embedding
                    )

            except openai.RateLimitError as e:
                if num_attempt >= MAX_RETRY:
                    raise e
                rate_limit_type = self.log_rate_limit_details(e, num_attempt)
            except RETRYABLE_EXCEPTIONS as e:
                if num_attempt >= MAX_RETRY:
                    raise e
                logger.warning(
                    f"Retryable error: {type(e).__name__} - {e} - Retry {num_attempt}"
                )
            except Exception as e:
                # logger.error(f"Non-retryable error: {type(e).__name__} - {e}")
                raise e
            else:
                break

        toc = time.monotonic()
        latency = toc - tic

        (
            input_tokens,
            input_cached_tokens,
            output_tokens,
            output_reasoning_tokens,
            total_tokens,
        ) = _get_response_tokens(response)

        # Calculate cost
        price_info = MODEL_PRICES.get(self.model, {"input": 0, "output": 0})
        cost_input = (input_tokens / 1_000_000) * price_info["input"]
        cost_output = (output_tokens / 1_000_000) * price_info["output"]
        cost_total = cost_input + cost_output

        self.input_tokens_delta_deque.add(input_tokens - estimate_input_tokens)
        self.output_tokens_deque.add(output_tokens)

        prompt_preview = prompt or ""
        if isinstance(prompt_preview, list):
            prompt_preview = prompt_preview[0] if len(prompt_preview) > 0 else ""

        if self.enable_call_status:
            model_call_status["cache_miss"] += 1
            model_call_status["miss_input_tokens"] += input_tokens
            model_call_status["miss_input_cached_tokens"] += input_cached_tokens
            model_call_status["miss_output_tokens"] += output_tokens
            model_call_status["miss_output_reasoning_tokens"] += output_reasoning_tokens

        # Prepare log row
        log_row = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": self.model,
            "system_prompt_preview": (system_prompt or "").replace("\n", " ")[:80],
            "prompt_preview": prompt_preview.replace("\n", " ")[:80],
            "completion_preview": (str(content) or "").replace("\n", " ")[:80],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_input_tokens": f"{estimate_input_tokens}",
            "estimated_output_tokens": f"{estimate_output_tokens}",
            "total_tokens": total_tokens,
            "cost_input_usd": round(cost_input, 6),
            "cost_output_usd": round(cost_output, 6),
            "cost_total_usd": round(cost_total, 6),
            "latency_sec": round(latency, 4),
        }

        _global_csv_logger.log(log_row)

        # 5. Cache the response
        if not no_write_cache_flag:
            self.cache[cache_key] = response

        return content

    class BatchCompletion:
        model: str = "gpt-4o-mini"
        workspace: str
        cache: Cache
        current_id: int
        # job_id: Optional[str] = None
        # file_id: Optional[str] = None
        # state: Literal["enqueuing", "in_progress", "completed", "failed"]

        def load_status(self):
            if osp.exists(self.status_file):
                with open(self.status_file, "r") as f:
                    status = json.load(f)
                return status
            else:
                return {}

        @property
        def job_id(self):
            status = self.load_status()
            return status.get("job_id", None)

        @job_id.setter
        def job_id(self, value):
            status = self.load_status()
            status["job_id"] = value
            with open(self.status_file, "w") as f:
                json.dump(status, f)

        @property
        def state(self):
            status = self.load_status()
            return status.get("state", "enqueuing")

        @state.setter
        def state(self, value):
            status = self.load_status()
            status["state"] = value
            with open(self.status_file, "w") as f:
                json.dump(status, f)

        def __init__(self, model: str, cache: Cache, workspace: str = None):
            self.model = model
            self.cache = cache
            self.current_id = -1  # start from 0
            if workspace is not None:
                self.workspace = workspace
            else:
                self.workspace = osp.join(
                    MAXLLM_BATCH_FOLDER,
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

            self.cached_result_file = osp.join(self.workspace, "cached_results.jsonl")
            self.enqueued_file = osp.join(self.workspace, "enqueued_tasks.jsonl")
            self.batch_file = osp.join(self.workspace, "batch_tasks.jsonl")
            self.status_file = osp.join(self.workspace, "status.json")
            self.result_file = osp.join(self.workspace, "batch_results.jsonl")

            if not osp.exists(self.status_file):
                os.makedirs(self.workspace, exist_ok=True)
                self.state = "enqueuing"
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_base=os.environ.get("OPENAI_API_BASE"),
            )

        def enqueue(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: list[dict] = [],
            json_mode: bool = False,
            json_format: Optional[BaseModel] = None,
            force: bool = False,
            **kwargs: Any,
        ):
            if self.state != "enqueuing":
                raise ValueError("BatchCompletion is not in enqueuing state.")
            self.current_id += 1
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            json_format_schema = (
                json_format.model_json_schema()
                if inspect.isclass(json_format) and issubclass(json_format, BaseModel)
                else json_format
            )
            cache_params = (
                messages,
                json_mode,
                json_format_schema,
                tuple(sorted(kwargs.items())),
            )
            cache_key = hashlib.md5(pickle.dumps(cache_params)).hexdigest()

            params = {
                "custom_id": self.current_id,
                "cache_key": cache_key,
                "messages": messages,
                "json_mode": json_mode,
                "json_format_schema": json_format_schema,
                **kwargs,
            }
            with open(self.enqueued_file, "a") as f:
                f.write(json.dumps(params) + "\n")

            if not force and cache_key in self.cache:
                cached_response = self.cache[cache_key]
                with open(self.cached_result_file, "a") as f:
                    f.write(
                        json.dumps(
                            {"custom_id": self.current_id, "response": cached_response}
                        )
                        + "\n"
                    )
                return self.current_id

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            elif json_format:
                kwargs["response_format"] = type_to_response_format(json_format)

            task = {
                "custom_id": f"{self.current_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    **kwargs,
                },
            }

            with open(self.batch_file, "a") as f:
                f.write(json.dumps(task) + "\n")
            return self.current_id

        def upload(self):
            if self.current_id == 0:
                raise ValueError("No task enqueued.")
            if not osp.exists(self.batch_file):
                self.state = "cached"  # everything is cached
                return None
            if self.state != "enqueuing":
                raise ValueError("BatchCompletion is not in enqueuing state.")
            batch_file = self.client.files.create(
                file=open(self.batch_file, "rb"), purpose="batch"
            )
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            self.job_id = batch_job.id
            logger.info(f"Batch job created: {batch_job}")
            self.state = "in_progress"
            return batch_job

        def get_job(self):
            if self.state == "enqueuing":
                raise ValueError("BatchCompletion has not been uploaded yet.")
            if self.state == "cached":
                return None
            batch_job = self.client.batches.retrieve(self.job_id)
            return batch_job

        async def wait_until_done(self, poll_interval=30, verbose=False):
            if (
                self.state == "completed"
                or self.state == "failed"
                or self.state == "cached"
            ):
                return self.get_job()

            if self.state != "in_progress":
                raise ValueError("BatchCompletion is not in in_progress state.")

            while True:
                batch_job = self.get_job()
                if batch_job.status == "completed":
                    self.state = "completed"
                    logger.info("Batch job completed.")
                    break
                elif batch_job.status == "failed":
                    self.state = "failed"
                    raise ValueError("Batch job failed.")
                else:
                    if verbose:
                        logger.info(
                            f"Batch job status: {batch_job.status}. in_progress..."
                        )
                    await asyncio.sleep(poll_interval)
            return batch_job

        async def results(self):
            if self.state == "in_progress":
                logger.info("Waiting for batch job to complete...")
                batch_job = await self.wait_until_done()

            if self.state == "completed":
                batch_job = self.get_job()
                if not osp.exists(self.result_file):
                    result_file_id = batch_job.output_file_id
                    result = self.client.files.content(result_file_id).content
                    with open(self.result_file, "wb") as f:
                        f.write(result)
            elif self.state == "cached":
                pass
            else:
                raise ValueError("BatchCompletion is not in completed/cached state.")

            results = []
            queue_tasks = []
            with open(self.enqueued_file, "r") as f:
                for line in f:
                    item = _parse_json_safely(line.strip())
                    queue_tasks.append(item)

            if self.state == "completed":  # not cached
                with open(self.result_file, "r") as f:
                    for line in f:
                        item = _parse_json_safely(line.strip())
                        custom_id = int(item["custom_id"])
                        response = item["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                        task = queue_tasks[custom_id]
                        if task["json_format_schema"]:
                            try:
                                response = _parse_json_safely(response)
                            except Exception as e:
                                logger.error(
                                    f"Failed to parse JSON response for task {custom_id}: {e}"
                                )
                                response = None
                        if "cache_key" in task:
                            self.cache[task["cache_key"]] = response
                        results.append(
                            {
                                "custom_id": custom_id,
                                # "status": item["status"],
                                "response": response,
                            }
                        )

            if osp.exists(self.cached_result_file):
                with open(self.cached_result_file, "r") as f:
                    for line in f:
                        item = _parse_json_safely(line.strip())
                        results.append(item)

            results = sorted(results, key=lambda x: x["custom_id"])
            return results

        def tasks(self):
            with open(self.enqueued_file, "r") as f:
                tasks = [_parse_json_safely(line) for line in f.readlines()]
            return tasks


class ExceptionWithMeta(Exception):
    def __init__(self, original_exception, meta):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.meta = meta

    def __str__(self):
        return f"{self.original_exception} | Meta: {self.meta}"

    def __repr__(self):
        return f"ExceptionWithMeta({repr(self.original_exception)}, {repr(self.meta)})"


_completers: dict[str, RateLimitCompleter] = {}


def get_completer(model: str, lazy_wake_up=False) -> RateLimitCompleter:
    if model not in _completers:
        _completers[model] = RateLimitCompleter(model_name=model, lazy_wake_up=lazy_wake_up)
    return _completers[model]


async def _async_openai_complete_single_completer(
    model: str = "gpt-4o-mini",
    prompt=None,
    system_prompt=None,
    history_messages=[],
    messages: list[dict] = [],
    json_mode: bool = False,
    json_format: Optional[BaseModel] = None,
    call_method: Literal[
        "auto", "chat.completions", "embeddings", "responses.parse", "completions"
    ] = "auto",
    raw=False,
    meta=None,
    force=False,
    *args,
    **kwargs,
) -> str | dict | tuple[str | dict, Any] | Completion | Response | EmbeddingResponse:
    try:
        completer = get_completer(model)
        result = await completer.async_complete(
            prompt,
            system_prompt,
            history_messages,
            messages,
            json_mode,
            json_format,
            call_method=call_method,
            raw=raw,
            force=force,
            *args,
            **kwargs,
        )
    except Exception as e:
        if meta:
            raise ExceptionWithMeta(e, meta) from e
        else:
            raise e from e
    if meta is not None:
        return result, meta
    else:
        return result


class WeightedCompleterSelector:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        self.current = [0] * len(weights)
        self.weight_sum = sum(weights)
        self.model_unique_name = None
        model_unique_names = [
            find_best_model_config(model).get("litellm_params", {}).get("model_unique_name", model) for model in models
        ]
        model_unique_names_set = set(model_unique_names)
        model_unique_names = sorted(list(model_unique_names_set))
        self.model_unique_name = "+".join(model_unique_names)

    def next(self):
        if not self.models:
            return None

        # 让每个模型累计权重
        for i in range(len(self.weights)):
            self.current[i] += self.weights[i]

        # 选到当前权重最大的那个
        idx = max(range(len(self.current)), key=lambda i: self.current[i])

        # 平滑回调
        self.current[idx] -= self.weight_sum

        return self.models[idx]


selectors: dict[str, WeightedCompleterSelector] = {}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _create_selector_from_model_weights(model_weights_str: str):
    # 去除空白
    model_weights_str = model_weights_str.strip()

    # 如果没有 "+"，说明只有一个模型，直接返回 completer
    if "+" not in model_weights_str:
        return WeightedCompleterSelector([model_weights_str], [1])

    # 多模型：使用 "+" 分割
    parts = model_weights_str.split("+")

    models = []
    weights = []

    for part in parts:
        part = part.strip()

        # 找最后一个 ":" 作为权重分隔
        if ":" not in part:
            model = part
            config = find_best_model_config(model)
            weight = config.get("litellm_params", {}).get("weight", "1")
        else:
            model, weight = part.rsplit(":", 1)

        if not _is_number(weight):
            raise ValueError(f"权重必须是数字: {part}")

        models.append(model)
        weights.append(float(weight))

    # 返回加权调度器
    return WeightedCompleterSelector(models, weights)


def _get_selector(model_weights_str: str) -> WeightedCompleterSelector:
    if model_weights_str not in selectors:
        selectors[model_weights_str] = _create_selector_from_model_weights(
            model_weights_str
        )
    return selectors[model_weights_str]


async def async_openai_complete(
    model: str = "gpt-4o-mini",
    prompt=None,
    system_prompt=None,
    history_messages=[],
    messages: list[dict] = [],
    json_mode: bool = False,
    json_format: Optional[BaseModel] = None,
    call_method: Literal[
        "auto", "chat.completions", "embeddings", "responses.parse", "completions"
    ] = "auto",
    raw=False,
    meta=None,
    force=False,
    *args,
    **kwargs,
) -> str | dict | tuple[str | dict, Any] | Completion | Response | EmbeddingResponse:
    """
    A wrapper function to complete text using OpenAI API with rate limiting, caching, and error handling.

    :param model: The model to use for completion.
    :param prompt: The prompt text.
    :param system_prompt: The system prompt text.
    :param history_messages: List of previous messages for context, .g. [{"role": "user", "content": "Hello"}].
    :param messages: Full list of messages including system, user, and assistant messages, if provided, 'prompt', 'system_prompt', and 'history_messages' will be ignored.
    :param json_mode: Whether to expect a JSON response, output will be a STR.
    :param json_format: A Pydantic BaseModel class to parse the JSON response into, output will be a DICT.
    :param meta: Optional metadata to attach to exceptions for debugging, if set, returns (response, meta).

    :return response: The completed text or parsed JSON response, or a tuple of (response, meta) if meta is provided.
    :exception: Raises ExceptionWithMeta if an error occurs and meta is provided, otherwise raises the original exception.
    """
    selector = _get_selector(model)
    model = selector.next()
    return await _async_openai_complete_single_completer(
        model,
        prompt,
        system_prompt,
        history_messages,
        messages,
        json_mode,
        json_format,
        call_method,
        raw,
        meta,
        force,
        *args,
        **kwargs,
    )


def openai_complete(
    model: str = "gpt-4o-mini",
    prompt=None,
    system_prompt=None,
    history_messages=[],
    messages: list[dict] = [],
    json_mode: bool = False,
    json_format: Optional[BaseModel] = None,
    call_method: Literal[
        "auto", "chat.completions", "embeddings", "responses.parse", "completions"
    ] = "auto",
    raw=False,
    meta=None,
    force=False,
    *args,
    **kwargs,
) -> str | dict | tuple[str | dict, Any]:
    """
    A wrapper function to complete text using OpenAI API with rate limiting, caching, and error handling.

    :param model: The model to use for completion.
    :param prompt: The prompt text.
    :param system_prompt: The system prompt text.
    :param history_messages: List of previous messages for context, .g. [{"role": "user", "content": "Hello"}].
    :param messages: Full list of messages including system, user, and assistant messages, if provided, 'prompt', 'system_prompt', and 'history_messages' will be ignored.
    :param json_mode: Whether to expect a JSON response, output will be a STR.
    :param json_format: A Pydantic BaseModel class to parse the JSON response into, output will be a DICT.
    :param meta: Optional metadata to attach to exceptions for debugging, if set, returns (response, meta).

    :return response: The completed text or parsed JSON response, or a tuple of (response, meta) if meta is provided.
    :exception: Raises ExceptionWithMeta if an error occurs and meta is provided, otherwise raises the original exception.
    """
    try:
        completer = get_completer(model)
        result = completer.complete(
            prompt,
            system_prompt,
            history_messages,
            messages,
            json_mode,
            json_format,
            call_method=call_method,
            raw=raw,
            force=force,
            *args,
            **kwargs,
        )
    except Exception as e:
        if meta:
            raise ExceptionWithMeta(e, meta) from e
        else:
            raise e from e
    if meta is not None:
        return result, meta
    else:
        return result


def get_batch(
    model="gpt-4o-mini", workspace=None
) -> RateLimitCompleter.BatchCompletion:
    completer = get_completer(model)
    batch = RateLimitCompleter.BatchCompletion(
        model_name=model, cache=completer.cache, workspace=workspace
    )
    return batch


def create_batch(
    model="gpt-4o-mini", workspace=None
) -> RateLimitCompleter.BatchCompletion:
    if osp.exists(workspace):
        raise ValueError(f"Workspace {workspace} already exists.")
    return get_batch(model, workspace)

def warmup_models(models: list[str]):
    start_time = time.time()
    model_unique_names = []
    for model in models:
        selector = _get_selector(model)
        for model in selector.models:
            if model not in _completers:
                completer = get_completer(model, lazy_wake_up=True)
                completer.vllm_sleep_mode_manager.auto_wake_up()
        model_unique_names.append(selector.model_unique_name)
    end_time = time.time()
    logger.info(f"Warmup {model} took {end_time - start_time:.2f} seconds.")
    return model_unique_names
    

async def awarmup_models(models: list[str]):
    start_time = time.time()
    model_unique_names = []
    for model in models:
        selector = _get_selector(model)
        model_unique_names.append(selector.model_unique_name)
        tasks = []
        for model in selector.models:
            if model not in _completers:
                completer = get_completer(model, lazy_wake_up=True)
                tasks.append(completer.vllm_sleep_mode_manager.aauto_wake_up())
    await asyncio.gather(*tasks)
    end_time = time.time()
    logger.info(f"Async warmup {model} took {end_time - start_time:.2f} seconds.")
    return model_unique_names

async def _async_index_wrap(task, i, semaphore=None):
    if semaphore:
        async with semaphore:
            try:
                result = await task
            except Exception as e:
                raise ExceptionWithMeta(e, i) from e
            return result, i
    else:
        try:
            result = await task
        except Exception as e:
            raise ExceptionWithMeta(e, i) from e
        return result, i


async def batch_async_tqdm(
    tasks,
    concurrency: Optional[int] = None,
    smoothing=0.3,
    miniters=1,
    desc="Async task",
    result_handler=None,
    raise_exceptions=True,
    **tqdm_kwargs,
):
    if concurrency is None or concurrency <= 0:
        semaphore = None
    else:
        semaphore = AutoLoopBigSemaphore(concurrency)
    indexed_tasks = [
        _async_index_wrap(t, i, semaphore=semaphore) for i, t in enumerate(tasks)
    ]
    results = {}
    for task in tqdm(
        asyncio.as_completed(indexed_tasks),
        total=len(tasks),
        smoothing=smoothing,
        miniters=miniters,
        desc=desc,
        **tqdm_kwargs,
    ):
        try:
            result, i = await task
        except ExceptionWithMeta as e:
            result, i = e.original_exception, e.meta
            if raise_exceptions:
                raise e.original_exception from e
        results[i] = result
        if result_handler:
            result_handler(result, i)
    ordered_results = [results[i] for i in range(len(results))]
    return ordered_results


async def batch_async_shared_tqdm(
    tasks,
    pbar,
    concurrency=None,
    smoothing=0.3,
    miniters=1,
    desc="Async task",
    result_handler=None,
    **tqdm_kwargs,
):
    if len(tqdm_kwargs) > 0:
        logger.warning("tqdm_kwargs are ignored when pbar is provided.")
    if concurrency is None or concurrency <= 0:
        semaphore = None
    else:
        semaphore = AutoLoopBigSemaphore(concurrency)
    indexed_tasks = [
        _async_index_wrap(t, i, semaphore=semaphore) for i, t in enumerate(tasks)
    ]
    results = {}
    for task in asyncio.as_completed(indexed_tasks):
        result, i = await task
        results[i] = result
        if result_handler:
            result_handler(result, i)
        pbar.update(1)
    ordered_results = [results[i] for i in range(len(results))]
    return ordered_results


async def batch_embedding(
    texts: List[str], model, desc="Embedding texts", concurrency=120, placeholder=None
) -> List[Optional[List[float]]]:
    batch_tokens = 3_000_000
    batch_size = batch_tokens // 8192
    batched_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # tasks.append(async_openai_complete(model=model, inputs=batch, meta=i))
        batched_texts.append(batch)

    embedding_batches = [None] * len(batched_texts)

    pbar = tqdm(total=len(texts), desc=desc)

    async def process_batch(batch, i):
        try:
            response = await async_openai_complete(model=model, inputs=batch)
            embedding_batches[i] = response
        except Exception as e:
            logger.error(f"Batch embedding failed at index {i}: {e}")
            embedding_batches[i] = [placeholder] * len(batch)
        finally:
            pbar.update(len(batch))

    await batch_async_tqdm(
        [process_batch(batch, i) for i, batch in enumerate(batched_texts)],
        concurrency=concurrency,
        disable=True,
    )

    return sum(embedding_batches, [])


async def batch_complete(
    prompts: List[str],
    model,
    desc="Completing texts",
    concurrency=60,
    placeholder=None,
    **kwargs,
) -> List[Optional[str]]:
    responses = [None] * len(prompts)

    async def process_prompt(prompt, i):
        try:
            if isinstance(prompt, dict):
                response = await async_openai_complete(model=model, **prompt, **kwargs)
            elif isinstance(prompt, str):
                response = await async_openai_complete(
                    model=model, prompt=prompt, **kwargs
                )
        except Exception as e:
            logger.error(f"Completion failed at index {i}: {e}")
            response = placeholder
        responses[i] = response

    tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]
    await batch_async_tqdm(tasks, concurrency=concurrency, desc=desc)
    return responses


async def batch_example():
    batch = get_batch(model="gpt-4o-mini", workspace="test_batch/1")
    if batch.state == "enqueuing":
        batch.enqueue(prompt="Write a short poem about number 1")
        batch.enqueue(prompt="Write a short poem about number 2")
        job = batch.upload()
    job = batch.get_job()
    await batch.wait_until_done(verbose=True)
    results = await batch.results()
    print(results)

    batch = get_batch(model="gpt-4o-mini", workspace="test_batch/2")
    if batch.state == "enqueuing":
        batch.enqueue(prompt="Write a short poem about number 1")
        batch.enqueue(prompt="Write a short poem about number 2")
        batch.enqueue(prompt="Write a short poem about number 1234")
        job = batch.upload()
    job = batch.get_job()
    await batch.wait_until_done(verbose=True)
    results = await batch.results()
    print(results)


async def _benchmark(n_task=20, model="gpt-4o-mini"):
    try:
        logger.info("--- Starting OpenAI Request Stress Test (Async) ---")

        time_start = time.monotonic()
        tasks = [
            async_openai_complete(
                model=model, prompt=f"Write a short poem about number {i}", force=True
            )
            for i in range(n_task)
        ]
        time_task_created = time.monotonic()

        results = await batch_async_tqdm(tasks, desc="Processing tasks", unit="task")

        time_task_done = time.monotonic()
        logger.info("--- Test Finished ---")

        logger.info(
            f"Elapsed time for task creation: {time_task_created - time_start:.2f}s"
        )
        logger.info(
            f"Elapsed time for task processing: {time_task_done - time_task_created:.2f}s"
        )
        logger.info(
            [f"Result {i}: {result[:60]}..." for i, result in enumerate(results)]
        )

    finally:
        # Ensure the logger file is properly closed on exit
        # _global_csv_logger.close()
        pass


async def _examples():
    class KeywordResponse(BaseModel):
        keywords: List[str]

    prompt = "Extract keywords from the following text, output in JSON format with a single field 'keywords' as a list of strings.\n\nText: Machine learning is a field of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. It involves the use of data to train models that can make predictions or decisions based on new input data."

    print(KeywordResponse.model_json_schema())

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt,
        json_mode=True,
    )
    print(response)

    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt=prompt,
        json_format=KeywordResponse,
    )
    print(response)

    response = await async_openai_complete(
        model="text-embedding-3-small",
        input=prompt,
    )
    print(response)

    response = await async_openai_complete(
        model="text-embedding-3-small",
        input=[prompt, prompt + " Another input."],
    )
    print(response)


async def _main():
    # await _examples()
    await _benchmark(10, model="Qwen3-4B-3090+Qwen3-4B-4090")  # warmup
    await _benchmark(100, model="Qwen3-4B-3090")
    await _benchmark(100, model="Qwen3-4B-3090+Qwen3-4B-4090")


# python -m maxllm._maxllm
if __name__ == "__main__":
    asyncio.run(_main())
# MaxLLM

一个统一的 OpenAI API 客户端，提供强大的速率限制、缓存和批处理功能。

> 单文件，你也可以直接将 `maxllm.py` 拷贝到你的项目中使用,无需安装。

## 功能特性

- **统一 API 接口**: 支持 chat completions、embeddings、structured output 等多种 API 调用方式
- **智能速率限制**: 基于 RPM (每分钟请求数) 和 TPM (每分钟 token 数) 的智能速率控制
- **自动缓存**: 基于磁盘的响应缓存,避免重复请求
- **批处理支持**: 支持 OpenAI Batch API 进行大规模批量处理
- **并发控制**: 内置并发控制和进度条显示
- **成本追踪**: 自动记录 token 使用和成本统计
- **多模型负载均衡**: 支持在多个模型间按权重分配请求

## 安装

```bash
pip install maxllm
```

或从源码安装:

```bash
git clone https://github.com/panjd123/maxllm.git
cd maxllm
pip install -e .
# pip install "git+https://github.com/panjd123/maxllm.git"
```

## 配置

MaxLLM 使用 `maxllm.yaml` 配置文件来管理模型和速率限制。

### 配置文件位置

MaxLLM 会按以下顺序查找配置文件:
1. 环境变量 `MAXLLM_CONFIG_PATH` 指定的路径
2. 当前工作目录下的 `maxllm.yaml`
3. 用户主目录下的 `.maxllm/maxllm.yaml`

### 创建配置文件

在你的项目根目录创建 `maxllm.yaml` 文件:

```yaml
model_list:
  # 默认配置,匹配所有模型
  - model_name: "*"
    litellm_params:
      model: "*"
      api_base: https://api.openai.com/v1
      api_key: os.environ/OPENAI_API_KEY

  # 自定义模型配置
  - model_name: "gpt-4o-mini"
    litellm_params:
      model: "gpt-4o-mini"
      api_base: https://api.openai.com/v1
      api_key: os.environ/OPENAI_API_KEY
      rpm: 500
      tpm: 200000
      concurrency: 100

rate_limit:
  default:
    - model_name: "local_completion"
      rpm: 100
      tpm: 50000
    - model_name: "local_embedding"
      rpm: 1000
      tpm: 1000000
    - model_name: "*"
      rpm: 500
      tpm: 200000

  custom_limits:
    - model_name: "gpt-4o-mini"
      rpm: 500
      tpm: 200000
```

### 设置配置文件路径

方式1: 通过环境变量

```bash
export MAXLLM_CONFIG_PATH=/path/to/your/maxllm.yaml
```

方式2: 在代码中设置(在导入 maxllm 之前)

```python
import os
os.environ['MAXLLM_CONFIG_PATH'] = '/path/to/your/maxllm.yaml'

import maxllm
```

方式3: 在当前工作目录创建 `maxllm.yaml`

直接在项目根目录创建 `maxllm.yaml`,无需设置环境变量。

方式4: 在用户主目录创建配置文件（推荐）

在 `~/.maxllm/maxllm.yaml` 创建配置文件,适用于所有项目。

## 环境变量

MaxLLM 支持以下环境变量:

- `MAXLLM_CONFIG_PATH`: maxllm.yaml 配置文件路径
- `OPENAI_API_KEY`: OpenAI API 密钥
- `OPENAI_API_BASE`: OpenAI API 基础 URL
- `RECACHE_FLAG`: 设置为 "1" 强制重新缓存，这和禁用缓存有些不同，强制重新缓存是在请求后更新缓存，如果你想越过缓存，重新请求且不更新现有的缓存请考虑使用 `request_flag` 参数。

> `OPENAI_API_*` 变量实际上仅作为默认值使用，如果你的配置文件没有问题，不设置他们也没什么问题。

建议使用 `.env` 文件管理环境变量:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
MAXLLM_CONFIG_PATH=./maxllm.yaml
```

## 使用示例

### 基础文本生成

```python
import asyncio
from maxllm import async_openai_complete

async def main():
    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt="Write a short poem about AI",
        system_prompt="You are a creative poet."
    )
    print(response)

asyncio.run(main())
```

### JSON 模式输出

```python
from pydantic import BaseModel
from maxllm import async_openai_complete

class Keywords(BaseModel):
    keywords: list[str]

async def extract_keywords():
    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt="Extract keywords from: Machine learning is amazing",
        json_format=Keywords
    )
    print(response)  # {'keywords': ['machine learning', 'amazing']}
```

### 批量处理

```python
from maxllm import batch_complete

async def batch_process():
    prompts = [
        "Write a haiku about number 1",
        "Write a haiku about number 2",
        "Write a haiku about number 3"
    ]

    results = await batch_complete(
        prompts=prompts,
        model="gpt-4o-mini",
        desc="Generating haikus",
        concurrency=10
    )

    for i, result in enumerate(results):
        print(f"Prompt {i}: {result}")
```

### Embedding 生成

```python
from maxllm import async_openai_complete

async def get_embeddings():
    # 单个文本
    embedding = await async_openai_complete(
        model="text-embedding-3-small",
        input="Hello world"
    )

    # 批量文本
    embeddings = await async_openai_complete(
        model="text-embedding-3-small",
        input=["Hello", "World", "AI"]
    )
```

### 查看统计信息

```python
from maxllm import get_call_status

# 获取所有模型的调用统计
status = get_call_status()
print(status)  # JSON 格式的统计信息,包括缓存命中率、token 使用、成本等
```

### 使用批处理 API

```python
from maxllm import get_batch

async def batch_api_example():
    # 创建批处理任务
    batch = get_batch(model="gpt-4o-mini", workspace="my_batch_001")

    if batch.state == "enqueuing":
        # 添加任务
        batch.enqueue(prompt="Write about topic 1")
        batch.enqueue(prompt="Write about topic 2")
        batch.enqueue(prompt="Write about topic 3")

        # 上传到 OpenAI
        job = batch.upload()

    # 等待完成
    await batch.wait_until_done(verbose=True)

    # 获取结果
    results = await batch.results()
    for result in results:
        print(f"Task {result['custom_id']}: {result['response']}")
```

### 多模型负载均衡

```python
# 按 1:2 的权重在两个模型间分配请求
response = await async_openai_complete(
    model="gpt-4o-mini:1+gpt-4-turbo:2",
    prompt="Your prompt here"
)
```

## API 参考

### `async_openai_complete()`

异步完成文本生成。

**参数:**
- `model` (str): 模型名称
- `prompt` (str, optional): 用户提示
- `system_prompt` (str, optional): 系统提示
- `history_messages` (list, optional): 历史消息列表
- `messages` (list, optional): 完整消息列表(如提供,其他参数将被忽略)
- `json_mode` (bool): 是否启用 JSON 模式
- `json_format` (BaseModel, optional): Pydantic 模型,用于结构化输出
- `raw` (bool): 是否返回原始响应对象
- `force` (bool): 是否强制重新请求(忽略缓存)
- `meta` (any, optional): 元数据,用于错误追踪
- `**kwargs`: 传递给 OpenAI API 的其他参数

**返回:**
- 文本响应、字典或原始响应对象

### `batch_complete()`

批量处理多个提示。

**参数:**
- `prompts` (list): 提示列表
- `model` (str): 模型名称
- `desc` (str): 进度条描述
- `concurrency` (int): 并发数量
- `placeholder` (any): 失败时的占位符
- `**kwargs`: 传递给 `async_openai_complete()` 的其他参数

### `get_call_status()`

获取所有模型的调用统计信息。

**返回:**
- JSON 格式的统计信息字符串

## 高级配置

### 缓存管理

缓存存储在 `~/maxllm/cache/{model_name}` 目录下。可以通过以下方式控制缓存:

有四个变量/参数可以影响缓存行为:

- 全局请求标志：相同标志的请求会使用相同的缓存。可以通过 `maxllm.set_request_flag(flag_value)` 设置全局标志,也可以在每次请求时通过 `request_flag` 参数单独设置。
- `request_flag` 参数： 在每次请求时传递，如果 `bool(request_flag)` 为假，则使用全局请求标志，否则使用传递的值。
    这意味着 0, Flase, "", None 会被映射到一样的缓存条目上，而 1, True 会被映射到不同的缓存条目上。这个设计可能有点问题，主要是为了缓存不会随着版本变化而失效。
- 强制跳过缓存参数 (`force`)：这个参数用于在单次请求中跳过缓存检查和缓存写入。如果设置为 True，则该请求不会读取缓存，也不会将响应写入缓存。
- 环境变量 `RECACHE_FLAG`：如果为真，会跳过缓存读，但会在请求后更新缓存，通常适用于作者我的开发阶段写入了一些错误的缓存，重新缓存以纠正它们。

```python
import maxllm
import random

maxllm.set_request_flag(random.randint(1, 1_000_000)) # 默认标记为 False，相同的标记会命中同一个缓存

# 使用 force 参数强制跳过缓存，它的行为是不读缓存，也不写缓存
response = await async_openai_complete(
    model="gpt-4o-mini",
    prompt="Your prompt",
    force=True  # 跳过缓存
)
```

### 日志记录

所有 API 调用会自动记录到 `~/maxllm/logs/openai_calls.csv`,包含:
- 时间戳
- 模型名称
- 提示预览
- Token 使用
- 成本
- 延迟

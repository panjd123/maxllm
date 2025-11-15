# MaxLLM 快速开始

这是一个快速开始指南,帮助你在 5 分钟内开始使用 MaxLLM。

## 1. 安装

```bash
git clone https://github.com/panjd123/maxllm.git
pip install -e maxllm
```

## 2. 设置环境变量

复制环境变量示例文件:

```bash
cp .env.example .env
```

编辑 `.env` 文件,填入你的 OpenAI API 密钥:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```

## 3. 创建并编辑配置文件

> 这是本库开发的主要目的，提供统一的配置管理，如果你不需要这一步，本库也可以直接当作一个高效的并发调用 OpenAI API 的工具包来使用

创建并编辑 ~/.maxllm/maxllm.yaml 文件:

```yaml
model_list:
  - model_name: "gpt-4o-mini"
    litellm_params:
      model: "gpt-4o-mini"
      api_base: https://api.openai.com/v1
      api_key: os.environ/OPENAI_API_KEY
      rpm: 500
      tpm: 200000

rate_limit:
  default:
    - model_name: "*"
      rpm: 500
      tpm: 200000
```

## 4. 运行第一个示例

创建一个 Python 文件 `hello.py`:

```python
import asyncio
from maxllm import async_openai_complete

async def main():
    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt="Say hello to MaxLLM!"
    )
    print(response)

asyncio.run(main())
```

运行:

```bash
python hello.py
```

## 5. 运行测试

```bash
cd tests
python run_all_tests.py
```

或者运行单个测试:

```bash
python test_basic.py      # 基础功能
python test_json.py       # JSON 输出
python test_batch.py      # 批量处理
python test_embedding.py  # Embedding 生成
```

## 6. 更多示例

### 结构化输出

> 注意这里的行为和标准的 openai 不一样，MaxLLM 默认返回 dict，如果你需要 Pydantic 模型，可以手动再处理一下。

```python
from pydantic import BaseModel
from maxllm import async_openai_complete

class Person(BaseModel):
    name: str
    age: int

async def main():
    response = await async_openai_complete(
        model="gpt-4o-mini",
        prompt="Extract: John is 30 years old",
        json_format=Person
    )
    print(response)  # {'name': 'John', 'age': 30}
    openai_style_response = Person(**response)
```

### 批量处理

```python
from maxllm import batch_complete

async def main():
    prompts = ["Count to 3", "Count to 5", "Count to 7"]
    results = await batch_complete(
        prompts=prompts,
        model="gpt-4o-mini",
        concurrency=3
    )
    for result in results:
        print(result)
```

### 查看统计

```python
from maxllm import get_call_status

status = get_call_status()
print(status)  # JSON 格式的统计信息
```

## 常见问题

### Q: 如何调整速率限制?

A: 在 `maxllm.yaml` 中调整 `rpm` (每分钟请求数) 和 `tpm` (每分钟 token 数):

```yaml
rate_limit:
  custom_limits:
    - model_name: "gpt-4o-mini"
      rpm: 1000  # 调高速率
      tpm: 500000
```

## 下一步

查看完整文档: [README.md](README.md)

查看测试示例: [tests/](tests/)

# MaxLLM Tests

这个目录包含了 MaxLLM 库的测试脚本。

## 准备工作

### 1. 安装依赖

首先安装 MaxLLM (开发模式):

```bash
cd ..  # 回到项目根目录
pip install -e .
```

### 2. 配置环境变量

复制 `.env.example` 创建 `.env` 文件:

```bash
cp ../.env.example ../.env
```

然后编辑 `.env` 文件,填入你的 API 密钥:

```bash
OPENAI_API_KEY=your-actual-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```

### 3. (可选) 创建配置文件

如果需要自定义模型配置和速率限制,在项目根目录创建 `maxllm.yaml`:

```bash
cp ../maxllm.yaml.example ../maxllm.yaml  # 如果有示例文件
# 或者参考 README.md 创建配置文件
```

## 运行测试

### 运行所有测试

```bash
python run_all_tests.py
```

### 运行单个测试

```bash
# 基础功能测试
python test_basic.py

# JSON 模式和结构化输出测试
python test_json.py

# 批量处理测试
python test_batch.py

# Embedding 生成测试
python test_embedding.py
```

## 测试说明

### test_basic.py

测试基础功能:
- 基本的文本生成
- 带历史记录的对话
- 缓存功能

### test_json.py

测试 JSON 相关功能:
- JSON 模式输出 (返回字符串)
- 结构化输出 (使用 Pydantic,返回字典)
- 复杂 schema 处理

### test_batch.py

测试批量处理功能:
- 简单批量处理
- 带参数的批量处理
- 错误处理

### test_embedding.py

测试 Embedding 生成:
- 单个文本 embedding
- 批量 embedding (API)
- 批量 embedding (辅助函数)
- Embedding 相似度计算

## 注意事项

1. **API 费用**: 运行测试会调用真实的 API,可能产生费用。请确保了解相关费用。

2. **速率限制**: 如果遇到速率限制错误,可以:
   - 减少并发数 (`concurrency` 参数)
   - 在 `maxllm.yaml` 中调整 `rpm` 和 `tpm` 设置
   - 等待一段时间后重试

3. **缓存**: 测试会使用缓存。如果要强制重新请求:
   - 在 `.env` 中设置 `RECACHE_FLAG=1`
   - 或者删除 `.diskcache` 目录

4. **日志**: API 调用日志会保存在 `logs/openai_calls.csv`

## 查看统计信息

每个测试结束后会显示调用统计,包括:
- 总调用次数
- 缓存命中率
- Token 使用情况
- 成本估算

## 故障排除

### ModuleNotFoundError: No module named 'maxllm'

确保已经安装了 maxllm:
```bash
cd ..
pip install -e .
```

### API Key 错误

检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确。

### 找不到配置文件

创建 `.env` 文件或设置 `MAXLLM_CONFIG_PATH` 环境变量。

### Rate Limit 错误

降低并发数或在 `maxllm.yaml` 中调整速率限制配置。

# LangChain 学习项目（2026 版）

这个目录不再把 LangChain 当成“背 API 名词表”的项目，而是按现在更推荐的写法来组织：

- 先学 `Prompt -> Model -> Parser` 的 LCEL / Runnable 思路
- 再学顺序编排、并行编排、路由
- 再看“记忆”在新版本里如何落到消息历史和 LangGraph checkpointer
- 然后进入现代 RAG
- 最后用 `create_agent` 和工具调用把这些拼起来

如果你以前学过旧教程，你会发现这里刻意弱化了不少已经不太推荐的新手入口，比如：

- 到处使用 `LLMChain`
- 依赖 `initialize_agent(...)`
- 把 `ConversationBufferMemory` 当成默认方案
- 继续用 `RetrievalQA` 作为唯一的 RAG 教学入口

这些 API 不是完全不能用，但它们已经不再代表今天最值得先掌握的 LangChain 思维方式。

## 这一版项目重点更新了什么

1. 内容修正为正常中文，去掉原先的乱码。
2. 示例围绕当前主线 API 重写：
   - `prompt | model | parser`
   - `RunnableParallel`
   - `RunnableBranch`
   - `RunnableWithMessageHistory`
   - `create_retrieval_chain`
   - `create_agent`
3. 记忆部分强调“新项目优先考虑消息历史或 LangGraph checkpointer”。
4. RAG 示例改成更贴近当前实践的两段式结构：
   - 检索器
   - 文档组合问答链
5. Agent 示例改成工具调用与结构化输出，而不是旧式字符串解析 ReAct 为主。

## 环境要求

- Python 3.10+
- 一个可用的 OpenAI API Key

## 安装

```bash
pip install -r requirements.txt
```

然后复制环境变量模板：

```bash
cp .env.example .env
```

在 `.env` 中至少填入：

```env
OPENAI_API_KEY=your_api_key_here
```

默认模型建议使用：

- `OPENAI_MODEL=gpt-4.1-mini`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

如果你的账号已经开通更新的 GPT-5 系列模型，也可以自行切换，但本项目默认保留一个更通用、兼容性更好的设置。

## 目录说明

```text
langchain-learning/
|-- .env.example
|-- requirements.txt
|-- README.md
|-- 01_basics.py
|-- 02_chains.py
|-- 03_memory.py
|-- 04_rag.py
`-- 05_agents.py
```

### [01_basics.py](./01_basics.py)

你会学到：

- `PromptTemplate` 和 `ChatPromptTemplate`
- LCEL 的最小闭环
- `StrOutputParser`
- `with_structured_output(...)`

推荐先建立这个基本心智模型：

```python
chain = prompt | model | parser
```

这是今天很多 LangChain 代码的核心句型。

### [02_chains.py](./02_chains.py)

你会学到：

- 怎么把多个 runnable 串起来
- 怎么并行跑多个子任务
- 怎么做轻量路由

这里会刻意少讲老式 `SequentialChain`，多讲 LCEL 风格的数据流拼装。

### [03_memory.py](./03_memory.py)

你会学到：

- 为什么“记忆”本质上是状态管理
- `RunnableWithMessageHistory`
- `thread_id` / `session_id` 的意义
- LangGraph checkpointer 如何为 agent 提供短期记忆

这部分最重要的认知升级是：

> 现代 LangChain 里的 memory，不只是“塞一个 Memory 类”这么简单，而是围绕消息历史和 agent state 来组织。

### [04_rag.py](./04_rag.py)

你会学到：

- 文档加载
- 分块策略
- embedding 与向量检索
- `create_stuff_documents_chain`
- `create_retrieval_chain`

这部分会把“检索”与“回答”拆开讲清楚，帮助你形成一个更稳定的 RAG 框架感。

### [05_agents.py](./05_agents.py)

你会学到：

- `@tool`
- `create_agent`
- 工具调用型 agent
- 结构化输出型 agent

这一节强调的是：

> 现在更常见的是 tool-calling agent，而不是把 ReAct 当成唯一的 agent 入门方式。

## 推荐学习顺序

1. `python 01_basics.py`
2. `python 02_chains.py`
3. `python 03_memory.py`
4. `python 04_rag.py`
5. `python 05_agents.py`

## 旧写法与新写法速查

| 旧思路 | 现在更推荐的思路 |
|---|---|
| `LLMChain` | `prompt | model | parser` |
| `initialize_agent(...)` | `create_agent(...)` |
| `ConversationBufferMemory` 直接当默认记忆方案 | `RunnableWithMessageHistory` / LangGraph checkpointer |
| `RetrievalQA` 一把梭 | `retriever + create_stuff_documents_chain + create_retrieval_chain` |
| `text-embedding-ada-002` | `text-embedding-3-small` / `text-embedding-3-large` |

## 学的时候建议重点关注什么

1. 不要只盯着“某个类怎么 import”。
2. 多观察数据是怎么从上一步流到下一步的。
3. 多问自己：这个任务到底是“链式编排”“检索增强”还是“agent 决策”。
4. 先掌握一个清晰、能运行的最小版本，再去追求复杂架构。

## 常见问题

### 1. 没配 `OPENAI_API_KEY`

脚本会直接提示并跳过在线调用部分。先把 `.env` 配好。

### 2. 为什么这里默认不用很多旧教程里的类

因为它们对理解历史代码有帮助，但对学习“今天怎么写新项目”帮助有限。

### 3. 为什么示例尽量短

因为学习阶段最容易被“复杂但不必要的胶水代码”淹没。这里保留短小、可读、可迁移的版本。

## 参考方向

建议同时对照：

- LangChain 官方总览与教程
- LangChain 的迁移文档
- OpenAI 官方模型文档

本项目的示例风格已按这些较新的资料进行了更新。

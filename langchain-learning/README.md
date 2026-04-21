# LangChain 学习项目

这是一个结构化的 LangChain 学习项目，通过代码实例演示 LangChain 的核心功能与技巧，并附带详细的中文注释和知识点讲解。

## 项目结构

```
langchain-learning/
├── requirements.txt      # 项目依赖
├── .env.example         # 环境变量示例
├── README.md            # 项目说明文档（本文件）
├── 01_basics.py         # 基础模块：Prompt Templates, LLM, Output Parsers
├── 02_chains.py         # 链模块：LLMChain, SequentialChain, RouterChain
├── 03_memory.py         # 记忆模块：BufferMemory, WindowMemory, SummaryMemory
├── 04_rag.py            # RAG模块：Document Loaders, Splitters, Embeddings, RetrievalQA
└── 05_agents.py         # 代理与工具：自定义Tool, ReAct Agent
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 文件为 `.env`，并填入您的 API Key：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入您的 OpenAI API Key：

```env
OPENAI_API_KEY=your_actual_api_key_here
```

## 学习路径

建议按照以下顺序学习：

### 第一部分：基础模块（01_basics.py）

**学习目标：** 掌握 LangChain 的基本概念和组件

**核心内容：**
- PromptTemplate：创建可重用的提示词模板
- ChatPromptTemplate：针对对话场景的提示词模板
- LLM 和 ChatModel：大语言模型的直接调用
- Output Parsers：将模型输出解析为结构化数据

**运行示例：**
```bash
python 01_basics.py
```

**知识点：**
- SystemMessage vs HumanMessage：不同角色消息的区别
- 提示词参数化：如何使用占位符动态生成提示词
- 输出解析：如何将文本输出转换为 JSON 或其他结构化格式

---

### 第二部分：链模块（02_chains.py）

**学习目标：** 理解如何将多个组件组合成工作流

**核心内容：**
- LLMChain：基础链，将提示词和模型组合
- SimpleSequentialChain：顺序执行多个链
- SequentialChain：更复杂的顺序链，支持多输入输出
- RouterChain：根据输入动态选择执行路径

**运行示例：**
```bash
python 02_chains.py
```

**知识点：**
- 链的概念：如何将多个组件（Prompt、LLM、Parser 等）组合起来
- 输入/输出传递机制：前一个链的输出如何作为后一个链的输入
- 路由机制：如何根据输入内容的特征选择不同的处理逻辑

---

### 第三部分：记忆模块（03_memory.py）

**学习目标：** 掌握如何在对话中保持状态

**核心内容：**
- ConversationBufferMemory：保存完整的对话历史
- ConversationBufferWindowMemory：只保留最近几轮对话
- ConversationSummaryMemory：对话历史的摘要记忆

**运行示例：**
```bash
python 03_memory.py
```

**知识点：**
- 状态传递机制：Memory 如何保存和传递对话历史
- 不同 Memory 类型的适用场景：
  - BufferMemory：适合需要完整历史但对话较短的场景
  - WindowMemory：适合长对话，只关注最近内容的场景
  - SummaryMemory：适合长对话，需要保存关键信息的场景

---

### 第四部分：RAG 模块（04_rag.py）

**学习目标：** 理解检索增强生成（RAG）的原理和应用

**核心内容：**
- Document Loaders：加载文档（文本、PDF 等）
- Text Splitters：文本分割策略
- Embeddings 与 Vectorstores：向量化与向量存储
- RetrievalQA：基于检索的问答链

**运行示例：**
```bash
python 04_rag.py
```

**知识点：**
- RAG 原理：结合外部知识库和 LLM 的能力
- 向量检索：如何将文本转换为向量，通过相似度搜索找到相关内容
- 文本分割：将长文档切分成适合模型处理的块
- Embedding：将文本映射为高维向量，语义相似的文本向量距离近

---

### 第五部分：代理与工具（05_agents.py）

**学习目标：** 学习如何让 AI 自主决策和执行任务

**核心内容：**
- 自定义 Tool：创建和使用自定义工具
- ReAct Agent：使用 ReAct 模式的代理
- 工具调用机制和步骤追踪

**运行示例：**
```bash
python 05_agents.py
```

**知识点：**
- Agent 的概念：使用 LLM 决定采取什么行动的组件
- Tool（工具）：Agent 可以调用的功能，可以是 Python 函数
- ReAct 模式：Reasoning + Acting，通过思考-行动循环完成任务
- 步骤追踪：Agent 如何追踪完成复杂任务的多步操作

---

## 注意事项

1. **API Key 配置**：确保在 `.env` 文件中正确配置了 OpenAI API Key
2. **网络连接**：部分示例需要访问 OpenAI API，确保网络连接正常
3. **API 费用**：运行包含 API 调用的示例会产生费用，请注意控制使用量
4. **LangChain 版本**：本项目基于 LangChain 0.1.0+ 版本，如遇到兼容性问题，请更新依赖

## 学习建议

1. **逐个运行**：建议按照学习路径，逐个运行每个文件，理解每个示例的输出
2. **阅读注释**：代码中包含详细的中文注释，解释每一行关键代码的作用
3. **修改实验**：在理解代码的基础上，尝试修改参数或逻辑，观察变化
4. **实际应用**：结合实际需求，尝试使用所学知识构建简单的应用

## 常见问题

### Q: 提示 "未找到 OPENAI_API_KEY 环境变量"

**A:** 请确保：
1. 已创建 `.env` 文件
2. `.env` 文件中包含 `OPENAI_API_KEY=your_key_here`
3. 代码中正确加载了环境变量（`load_dotenv()`）

### Q: API 调用超时或失败

**A:** 可能的原因：
1. 网络连接问题
2. API Key 无效或余额不足
3. OpenAI 服务暂时不可用

### Q: 某些示例运行很慢

**A:** 这是正常现象，因为：
1. 需要调用 OpenAI API，网络请求需要时间
2. 某些示例涉及多次 API 调用（如 Agent 示例）
3. 向量化过程可能需要较长时间

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

## 许可证

本项目仅用于学习目的。

---

**祝学习愉快！**
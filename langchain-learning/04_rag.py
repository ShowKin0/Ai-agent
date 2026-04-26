"""
Modern RAG with LangChain
==========================

这一节使用更清晰的现代 RAG 结构：

1. 加载文档
2. 切分文档
3. 向量化并建立检索器
4. 用 create_stuff_documents_chain 组织回答链
5. 用 create_retrieval_chain 把检索和回答接起来

重点不是记住一堆类名，而是理解：
"检索"和"回答"是两个不同职责。
"""

# 导入注解特性，支持前向引用
from __future__ import annotations

# 导入 os 模块，用于环境变量操作
import os
# 导入 tempfile，用于创建临时文件
import tempfile
# 导入 Path，用于处理文件路径
from pathlib import Path

# 导入 load_dotenv，用于加载 .env 文件中的环境变量
from dotenv import load_dotenv

# 导入 create_retrieval_chain，用于创建完整的 RAG 链（检索+回答）
from langchain.chains import create_retrieval_chain
# 导入 create_stuff_documents_chain，用于创建文档回答链
from langchain.chains.combine_documents import create_stuff_documents_chain
# 导入 TextLoader，用于加载文本文件
from langchain_community.document_loaders import TextLoader
# 导入 Document，文档数据结构
from langchain_core.documents import Document
# 导入提示词模板
from langchain_core.prompts import ChatPromptTemplate
# 导入 InMemoryVectorStore，内存向量存储
from langchain_core.vectorstores import InMemoryVectorStore
# 导入 OpenAI 模型和嵌入模型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 导入 RecursiveCharacterTextSplitter，递归文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()


def print_title(title: str) -> None:
    """打印格式化的标题，用于在控制台输出中创建清晰的分隔"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_chat_model(temperature: float = 0.1) -> ChatOpenAI:
    """
    构建并返回一个配置好的 ChatOpenAI 实例

    参数:
        temperature: 控制模型生成随机性的参数，默认 0.1（较低，输出更确定性）

    返回:
        ChatOpenAI 实例
    """
    # 检查 API 密钥是否存在
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请先把 langchain-learning/.env.example 复制为 .env 并填写密钥。"
        )

    # 构建模型配置字典
    kwargs = {
        # 获取模型名称，默认 "gpt-4.1-mini"
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        # 设置温度参数，控制模型输出的随机性
        "temperature": temperature,
    }
    # 获取自定义 API 端点（可选）
    base_url = os.getenv("OPENAI_BASE_URL")
    # 如果存在自定义 URL，则添加到配置中
    if base_url:
        kwargs["base_url"] = base_url

    # 返回配置好的 ChatOpenAI 实例
    return ChatOpenAI(**kwargs)


def build_embeddings() -> OpenAIEmbeddings:
    """
    构建并返回一个配置好的 OpenAIEmbeddings 实例

    返回:
        OpenAIEmbeddings 实例，用于将文本转换为向量
    """
    # 检查 API 密钥是否存在
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请先把 langchain-learning/.env.example 复制为 .env 并填写密钥。"
        )

    # 构建嵌入模型配置字典
    kwargs = {
        # 获取嵌入模型名称，默认 "text-embedding-3-small"
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    # 获取自定义 API 端点（可选）
    base_url = os.getenv("OPENAI_BASE_URL")
    # 如果存在自定义 URL，则添加到配置中
    if base_url:
        kwargs["base_url"] = base_url

    # 返回配置好的 OpenAIEmbeddings 实例
    return OpenAIEmbeddings(**kwargs)


def example_document_loader() -> None:
    """
    演示 Document Loader 的用法
    Document Loader 用于从不同来源加载文档，如文本文件、PDF、网页等
    """
    print_title("示例 1：Document Loader")

    # 使用临时目录创建一个测试文件
    with tempfile.TemporaryDirectory() as temp_dir:
        # 构建临时文件路径
        file_path = Path(temp_dir) / "langchain_notes.txt"
        # 写入测试内容
        file_path.write_text(
            "LangChain 现在更推荐从 Runnable / LCEL 的角度理解工作流。\n"
            "RAG 的核心不是把所有知识塞进 prompt，而是先检索再生成。\n"
            "Agent 更适合处理需要动态决策、调用工具的任务。\n",
            encoding="utf-8",
        )

        # 创建文本加载器
        loader = TextLoader(str(file_path), encoding="utf-8")
        # 加载文档
        documents = loader.load()

    # 打印加载结果
    print(f"加载到 {len(documents)} 个文档。")
    print("文档内容：")
    print(documents[0].page_content)
    # 打印元数据（包含来源文件等信息）
    print(f"元数据：{documents[0].metadata}")


def build_demo_documents() -> list[Document]:
    """
    构建演示用的文档列表
    这些文档用于演示 RAG 的各个步骤

    返回:
        list[Document]: 文档列表
    """
    return [
        Document(
            page_content=(
                "LCEL 是 LangChain Expression Language。它把 prompt、model、parser、"
                "retriever 等模块统一成可组合的 runnable。"
            ),
            metadata={"topic": "lcel"},
        ),
        Document(
            page_content=(
                "RAG 的标准思路是先从外部知识库检索相关片段，再把片段连同问题一起交给模型回答。"
                "这样可以减少模型只凭参数记忆胡乱发挥。"
            ),
            metadata={"topic": "rag"},
        ),
        Document(
            page_content=(
                "Agent 适合那些步骤不固定的任务，比如先判断是否需要检索，再决定是否调用计算器或其他工具。"
            ),
            metadata={"topic": "agent"},
        ),
        Document(
            page_content=(
                "现代 LangChain 中，`create_retrieval_chain` 负责把 retriever 和回答链组合起来，"
                "职责比旧式一体化问答类更清晰。"
            ),
            metadata={"topic": "retrieval_chain"},
        ),
    ]


def example_text_splitting() -> list[Document]:
    """
    演示文本切分的用法
    将长文档切分成较小的块，每个块都会被单独向量化
    块的大小直接影响召回质量和上下文预算

    返回:
        list[Document]: 切分后的文档块列表
    """
    print_title("示例 2：文本切分")

    # 创建递归文本分割器
    # RecursiveCharacterTextSplitter 会尝试按段落、句子、单词的顺序递归分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,        # 每个块的最大字符数
        chunk_overlap=20,     # 相邻块之间的重叠字符数，确保语义连续性
    )
    # 分割文档
    chunks = splitter.split_documents(build_demo_documents())

    # 打印分割结果
    print(f"原始文档数：{len(build_demo_documents())}")
    print(f"切分后文档块数：{len(chunks)}")
    for index, chunk in enumerate(chunks[:4], start=1):
        print(f"\n块 {index}:")
        print(chunk.page_content)
        print(f"metadata={chunk.metadata}")

    print(
        "\n要点：chunk 不宜过大，也不宜过碎。它直接影响召回质量和上下文预算。"
    )
    return chunks


def example_similarity_search(chunks: list[Document]) -> InMemoryVectorStore:
    """
    演示向量检索的用法
    将文档块向量化后，可以根据查询的语义相似度检索相关文档

    参数:
        chunks: 文档块列表

    返回:
        InMemoryVectorStore: 内存向量存储
    """
    print_title("示例 3：向量检索")

    # 从文档块创建内存向量存储
    # 这个过程会：
    # 1. 使用嵌入模型将每个文档块转换为向量
    # 2. 将向量和文档内容存储到向量数据库中
    vector_store = InMemoryVectorStore.from_documents(
        chunks,
        embedding=build_embeddings(),
    )

    # 定义查询问题
    query = "为什么现在不建议把 RetrievalQA 当成唯一入口来理解 RAG？"
    # 执行相似度搜索，返回最相关的 k 个文档
    results = vector_store.similarity_search(query, k=3)

    # 打印检索结果
    print(f"查询：{query}")
    for index, doc in enumerate(results, start=1):
        print(f"\n命中 {index}:")
        print(doc.page_content)
        print(f"metadata={doc.metadata}")

    print("\n要点：检索阶段的任务是找"有用上下文"，不是直接替模型回答。")
    return vector_store


def example_retrieval_chain(vector_store: InMemoryVectorStore) -> None:
    """
    演示完整的两段式 RAG 流程
    第一段：从向量库检索相关文档
    第二段：根据检索到的文档生成回答

    参数:
        vector_store: 内存向量存储
    """
    print_title("示例 4：完整两段式 RAG")

    # 将向量存储转换为检索器
    # 检索器负责根据查询返回相关文档
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 创建 RAG 提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位谨慎的 RAG 助教。请只依据给定资料回答。"
                "如果资料不足，请明确说"根据当前检索结果还不能确定"。\n\n"
                "资料：\n{context}",
            ),
            ("human", "问题：{input}"),
        ]
    )

    # 创建文档回答链
    # create_stuff_documents_chain 会将检索到的所有文档"塞"到提示词中
    question_answer_chain = create_stuff_documents_chain(
        build_chat_model(temperature=0),  # 使用温度为 0 的模型
        prompt,
    )
    # 创建完整的 RAG 链
    # create_retrieval_chain 将检索器和回答链组合起来：
    #   1. 使用 retriever 检索相关文档
    #   2. 将检索到的文档和问题一起传给 question_answer_chain
    #   3. 返回包含 answer 和 context 的结果
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 定义问题
    question = "现代 LangChain 里，RAG 为什么更强调把检索和回答拆开？"
    # 调用 RAG 链
    result = rag_chain.invoke({"input": question})

    # 打印问题和回答
    print(f"问题：{question}")
    print("\n回答：")
    print(result["answer"])

    # 打印检索到的上下文
    print("\n检索到的上下文：")
    for index, doc in enumerate(result["context"], start=1):
        print(f"{index}. {doc.page_content}")

    print(
        "\n要点：这就是现代 RAG 的骨架。以后你接 Chroma、PGVector、Milvus，"
        "变化的主要是向量存储层，不是这个骨架本身。"
    )

    question_answer_chain = create_stuff_documents_chain(
        build_chat_model(temperature=0),
        prompt,
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    question = "现代 LangChain 里，RAG 为什么更强调把检索和回答拆开？"
    result = rag_chain.invoke({"input": question})

    print(f"问题：{question}")
    print("\n回答：")
    print(result["answer"])

    print("\n检索到的上下文：")
    for index, doc in enumerate(result["context"], start=1):
        print(f"{index}. {doc.page_content}")

    print(
        "\n要点：这就是现代 RAG 的骨架。以后你接 Chroma、PGVector、Milvus，"
        "变化的主要是向量存储层，不是这个骨架本身。"
    )


def main() -> None:
    """主函数，程序入口点"""
    # 打印主标题
    print_title("LangChain RAG 模块（现代版）")
    # 调用文档加载示例（不需要 API 密钥）
    example_document_loader()

    try:
        # 调用文本切分示例
        chunks = example_text_splitting()
        # 调用向量检索示例
        vector_store = example_similarity_search(chunks)
        # 调用完整 RAG 链示例
        example_retrieval_chain(vector_store)
    except RuntimeError as exc:
        # 捕获 RuntimeError（通常是 API 密钥缺失），并打印友好提示
        print(exc)
        print("已跳过需要在线模型的示例。")


# Python 惯用法：检查是否直接运行此脚本
if __name__ == "__main__":
    main()

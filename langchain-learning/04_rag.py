"""
Modern RAG with LangChain
=========================

这一节使用更清晰的现代 RAG 结构：

1. 加载文档
2. 切分文档
3. 向量化并建立检索器
4. 用 create_stuff_documents_chain 组织回答链
5. 用 create_retrieval_chain 把检索和回答接起来

重点不是记住一堆类名，而是理解：
“检索”和“回答”是两个不同职责。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_chat_model(temperature: float = 0.1) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请先把 langchain-learning/.env.example 复制为 .env 并填写密钥。"
        )

    kwargs = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "temperature": temperature,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def build_embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请先把 langchain-learning/.env.example 复制为 .env 并填写密钥。"
        )

    kwargs = {
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIEmbeddings(**kwargs)


def example_document_loader() -> None:
    print_title("示例 1：Document Loader")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "langchain_notes.txt"
        file_path.write_text(
            "LangChain 现在更推荐从 Runnable / LCEL 的角度理解工作流。\n"
            "RAG 的核心不是把所有知识塞进 prompt，而是先检索再生成。\n"
            "Agent 更适合处理需要动态决策、调用工具的任务。\n",
            encoding="utf-8",
        )

        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

    print(f"加载到 {len(documents)} 个文档。")
    print("文档内容：")
    print(documents[0].page_content)
    print(f"元数据：{documents[0].metadata}")


def build_demo_documents() -> list[Document]:
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
    print_title("示例 2：文本切分")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(build_demo_documents())

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
    print_title("示例 3：向量检索")

    vector_store = InMemoryVectorStore.from_documents(
        chunks,
        embedding=build_embeddings(),
    )

    query = "为什么现在不建议把 RetrievalQA 当成唯一入口来理解 RAG？"
    results = vector_store.similarity_search(query, k=3)

    print(f"查询：{query}")
    for index, doc in enumerate(results, start=1):
        print(f"\n命中 {index}:")
        print(doc.page_content)
        print(f"metadata={doc.metadata}")

    print("\n要点：检索阶段的任务是找“有用上下文”，不是直接替模型回答。")
    return vector_store


def example_retrieval_chain(vector_store: InMemoryVectorStore) -> None:
    print_title("示例 4：完整两段式 RAG")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位谨慎的 RAG 助教。请只依据给定资料回答。"
                "如果资料不足，请明确说“根据当前检索结果还不能确定”。\n\n"
                "资料：\n{context}",
            ),
            ("human", "问题：{input}"),
        ]
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
    print_title("LangChain RAG 模块（现代版）")
    example_document_loader()

    try:
        chunks = example_text_splitting()
        vector_store = example_similarity_search(chunks)
        example_retrieval_chain(vector_store)
    except RuntimeError as exc:
        print(exc)
        print("已跳过需要在线模型的示例。")


if __name__ == "__main__":
    main()

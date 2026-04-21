"""
LangChain 索引与检索（RAG）模块学习示例
======================================

本文件演示 LangChain 的检索增强生成（RAG）功能：
1. Document Loaders - 加载文档（文本、PDF 等）
2. Text Splitters - 文本分割策略
3. Embeddings 与 Vectorstores - 向量化与向量存储
4. RetrievalQA - 基于检索的问答链

核心知识点：
- RAG（Retrieval-Augmented Generation）：结合外部知识库和 LLM 的能力
- 向量检索：将文本转换为向量，通过相似度搜索找到相关内容
- 文本分割：将长文档切分成适合模型处理的块
- Embedding：将文本映射为高维向量，语义相似的文本向量距离近

运行前请确保：
1. 已安装依赖：pip install -r requirements.txt
2. 已配置 .env 文件（复制 .env.example 并填入 OPENAI_API_KEY）

注意：RAG 功能需要向量数据库，本示例使用 ChromaDB
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_classic import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_classic.adapters import Document
from langchain_classic.adapters import RetrievalQA


# ============================================================================
# 1. Document Loaders 示例
# ============================================================================

def example_text_loader():
    """
    演示 TextLoader 的用法
    
    TextLoader 用于从文本文件中加载内容。
    """
    print("=" * 60)
    print("示例 1：TextLoader 用法")
    print("=" * 60)
    
    # 创建一个示例文本文件
    sample_text = """
    LangChain 是一个强大的框架，用于开发由语言模型驱动的应用程序。
    
    核心概念：
    1. Chains（链）：将多个组件组合在一起
    2. Agents（代理）：使用 LLM 决定采取什么行动
    3. Memory（记忆）：在多次调用之间保持状态
    4. Prompts（提示词）：管理模型的输入
    
    LangChain 支持多种模型提供商，包括 OpenAI、Anthropic、Hugging Face 等。
    """
    
    # 保存示例文件
    sample_file = "sample_text.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    try:
        # 使用 TextLoader 加载文本文件
        loader = TextLoader(sample_file, encoding="utf-8")
        documents = loader.load()
        
        # 查看加载的文档
        print(f"加载了 {len(documents)} 个文档")
        print(f"文档内容:\n{documents[0].page_content[:200]}...")
        print(f"元数据: {documents[0].metadata}")
        print()
        
    finally:
        # 清理示例文件
        if os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"已清理临时文件: {sample_file}")
        print()


def example_document_manipulation():
    """
    演示如何手动创建和操作 Document 对象
    """
    print("=" * 60)
    print("示例 2：手动创建 Document 对象")
    print("=" * 60)
    
    # 手动创建 Document 对象
    docs = [
        Document(
            page_content="Python 是一种高级编程语言，语法简洁易学。",
            metadata={"source": "intro", "page": 1}
        ),
        Document(
            page_content="Python 支持多种编程范式，包括面向对象、函数式等。",
            metadata={"source": "features", "page": 2}
        ),
        Document(
            page_content="Python 有丰富的标准库和第三方库，适合快速开发。",
            metadata={"source": "features", "page": 3}
        )
    ]
    
    print(f"创建了 {len(docs)} 个文档对象")
    for i, doc in enumerate(docs, 1):
        print(f"\n文档 {i}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
    print()


# ============================================================================
# 2. Text Splitters 示例
# ============================================================================

def example_character_splitter():
    """
    演示 CharacterTextSplitter 的用法
    
    CharacterTextSplitter 按照指定的字符数分割文本。
    """
    print("=" * 60)
    print("示例 3：CharacterTextSplitter 用法")
    print("=" * 60)
    
    # 创建一个长文本
    long_text = """
    Python 是一种广泛使用的高级编程语言。它由 Guido van Rossum 于 1991 年首次发布。
    
    Python 的设计哲学强调代码的可读性和简洁的语法。相比于 C++ 或 Java，
    Python 让开发者能够用更少的代码行表达概念。
    
    Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
    它拥有一个完全动态的类型系统和自动内存管理功能。
    
    Python 的标准库庞大且功能丰富，被描述为"自带电池"（batteries included）。
    这意味着 Python 自带了许多常用的模块和包，无需额外安装。
    
    Python 在 Web 开发、数据科学、人工智能、自动化脚本等领域都有广泛应用。
    著名的框架包括 Django、Flask、Pandas、NumPy、TensorFlow 等。
    """
    
    # 创建分割器
    # chunk_size: 每个文本块的最大字符数
    # chunk_overlap: 相邻文本块之间的重叠字符数（保持上下文连续性）
    splitter = CharacterTextSplitter(
        separator="\n",  # 按换行符分割
        chunk_size=100,  # 每块最多 100 字符
        chunk_overlap=20,  # 重叠 20 字符
        length_function=len  # 计算长度的函数
    )
    
    # 分割文本
    chunks = splitter.split_text(long_text)
    
    print(f"原始文本长度: {len(long_text)} 字符")
    print(f"分割成 {len(chunks)} 个块")
    print(f"\n分割结果:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i} ({len(chunk)} 字符):")
        print(chunk.strip())
    print()


def example_recursive_splitter():
    """
    演示 RecursiveCharacterTextSplitter 的用法
    
    RecursiveCharacterTextSplitter 是 LangChain 推荐的分割器，
    它会递归地尝试不同的分隔符，以获得最佳的分割效果。
    """
    print("=" * 60)
    print("示例 4：RecursiveCharacterTextSplitter 用法")
    print("=" * 60)
    
    # 创建一个结构化的文本
    structured_text = """
    第一章：Python 简介
    
    Python 是一种高级编程语言，由 Guido van Rossum 创建。
    
    1.1 Python 的特点
    - 语法简洁
    - 易于学习
    - 跨平台
    - 丰富的库
    
    1.2 Python 的应用领域
    Web 开发、数据科学、人工智能等。
    
    第二章：安装 Python
    
    2.1 Windows 安装
    访问 python.org 下载安装包。
    
    2.2 Mac 安装
    使用 Homebrew：brew install python
    
    2.3 Linux 安装
    使用包管理器：sudo apt-get install python3
    """
    
    # 创建递归分割器
    # 分隔符优先级从高到低：\n\n -> \n -> 空格 -> ""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", " ", ""],
        chunk_size=150,
        chunk_overlap=30,
        length_function=len
    )
    
    # 分割文档
    chunks = splitter.split_text(structured_text)
    
    print(f"原始文本长度: {len(structured_text)} 字符")
    print(f"分割成 {len(chunks)} 个块")
    print(f"\n分割结果:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n块 {i} ({len(chunk)} 字符):")
        print(chunk.strip())
    print()


# ============================================================================
# 3. Embeddings 与 Vectorstores 示例
# ============================================================================

def example_embeddings():
    """
    演示 Embeddings（文本向量化）的用法
    
    Embeddings 将文本转换为高维向量，使得语义相似的文本在向量空间中距离较近。
    """
    print("=" * 60)
    print("示例 5：Embeddings 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        print("本示例需要有效的 OpenAI API Key")
        return
    
    try:
        # 创建 OpenAI Embeddings 对象
        # 默认使用 text-embedding-ada-002 模型
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # 要比较的文本
        texts = [
            "Python 是一种编程语言",
            "Java 是一种编程语言",
            "今天天气很好",
            "我喜欢吃苹果"
        ]
        
        print("正在计算文本向量...")
        
        # 将文本转换为向量
        # 每个文本会被转换为一个 1536 维的向量（使用 text-embedding-ada-002）
        text_embeddings = embeddings.embed_documents(texts)
        
        print(f"成功生成 {len(text_embeddings)} 个向量")
        print(f"每个向量维度: {len(text_embeddings[0])}")
        print()
        
        # 计算向量之间的相似度（使用余弦相似度）
        import numpy as np
        
        def cosine_similarity(v1, v2):
            """计算两个向量的余弦相似度"""
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            return dot_product / (norm1 * norm2)
        
        print("文本相似度分析:")
        print("-" * 60)
        
        # 比较第一个文本与其他文本的相似度
        base_text = texts[0]
        base_embedding = text_embeddings[0]
        
        for i, (text, embedding) in enumerate(zip(texts, text_embeddings)):
            if i == 0:
                continue
            similarity = cosine_similarity(base_embedding, embedding)
            print(f"'{base_text}' vs '{text}': {similarity:.4f}")
        print()
        
    except Exception as e:
        print(f"调用 Embeddings 时发生错误: {e}")
        print("提示：请检查 API Key 是否正确，网络连接是否正常")


def example_vectorstore():
    """
    演示 Vectorstore（向量存储）的用法
    
    Vectorstore 用于存储和检索向量，支持相似度搜索。
    """
    print("=" * 60)
    print("示例 6：Chroma Vectorstore 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建 Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # 准备文档
        docs = [
            Document(page_content="Python 是一种高级编程语言，语法简洁易学。"),
            Document(page_content="Java 是一种面向对象的编程语言。"),
            Document(page_content="C++ 是一种高性能的编程语言。"),
            Document(page_content="JavaScript 主要用于网页开发。"),
            Document(page_content="SQL 是用于数据库查询的语言。")
        ]
        
        print(f"正在创建向量存储，包含 {len(docs)} 个文档...")
        
        # 创建向量存储
        # Chroma 是一个轻量级的向量数据库，适合开发和测试
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"  # 持久化存储目录
        )
        
        print("向量存储创建成功！")
        print()
        
        # 演示相似度搜索
        query = "网页开发用的语言"
        print(f"查询: '{query}'")
        print("-" * 60)
        
        # 搜索最相关的文档
        # k 参数指定返回最相似的 k 个文档
        results = vectorstore.similarity_search(query, k=2)
        
        print(f"找到 {len(results)} 个相关文档:")
        for i, doc in enumerate(results, 1):
            print(f"\n文档 {i}:")
            print(f"内容: {doc.page_content}")
            print(f"相似度分数: {doc.metadata.get('score', 'N/A')}")
        print()
        
        # 演示带分数的搜索
        print("带相似度分数的搜索:")
        print("-" * 60)
        results_with_scores = vectorstore.similarity_search_with_score(query, k=2)
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n文档 {i}:")
            print(f"内容: {doc.page_content}")
            print(f"相似度分数: {score:.4f}")
        print()
        
        # 清理向量数据库
        print("清理向量数据库...")
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("已删除向量数据库目录")
        print()
        
    except Exception as e:
        print(f"创建 Vectorstore 时发生错误: {e}")


# ============================================================================
# 4. RetrievalQA 示例
# ============================================================================

def example_retrieval_qa():
    """
    演示 RetrievalQA（基于检索的问答）的用法
    
    RetrievalQA 将文档检索和 LLM 问答结合在一起：
    1. 从向量库中检索相关文档
    2. 将检索到的文档作为上下文
    3. 使用 LLM 基于上下文回答问题
    """
    print("=" * 60)
    print("示例 7：RetrievalQA 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建 LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0,
            model="gpt-3.5-turbo"
        )
        
        # 创建 Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # 准备知识库文档
        knowledge_base = [
            Document(page_content="LangChain 是一个用于开发由 LLM 驱动的应用程序的框架。"),
            Document(page_content="LangChain 的核心组件包括 Chains、Agents、Memory 和 Prompts。"),
            Document(page_content="Chains 是将多个组件（如 LLM、工具等）连接在一起的方式。"),
            Document(page_content="Agents 是使用 LLM 来决定采取什么行动的组件。"),
            Document(page_content="Memory 用于在多次调用之间保持状态。"),
            Document(page_content="Prompts 用于管理传递给 LLM 的输入。"),
            Document(page_content="LangChain 支持多种模型提供商，包括 OpenAI、Anthropic、Hugging Face。"),
            Document(page_content="RAG（检索增强生成）是结合外部知识库和 LLM 的技术。")
        ]
        
        print("正在创建向量存储...")
        
        # 创建向量存储
        vectorstore = Chroma.from_documents(
            documents=knowledge_base,
            embedding=embeddings,
            persist_directory="./chroma_qa_db"
        )
        
        # 创建检索器
        # k 参数指定检索到的文档数量
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 创建 RetrievalQA 链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" 将所有检索到的文档组合成一个提示词
            retriever=retriever,
            return_source_documents=True  # 返回检索到的文档
        )
        
        print("RetrievalQA 创建成功！")
        print()
        
        # 提问
        questions = [
            "LangChain 的核心组件有哪些？",
            "什么是 Chains？",
            "LangChain 支持哪些模型提供商？"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            print("-" * 60)
            
            # 执行问答
            result = qa_chain.invoke({"query": question})
            
            print(f"答案: {result['result']}")
            
            # 显示检索到的源文档
            print("\n参考文档:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"{i}. {doc.page_content}")
            print()
        
        # 清理向量数据库
        import shutil
        if os.path.exists("./chroma_qa_db"):
            shutil.rmtree("./chroma_qa_db")
            print("已清理向量数据库")
        print()
        
    except Exception as e:
        print(f"调用 RetrievalQA 时发生错误: {e}")


def example_rag_pipeline():
    """
    演示完整的 RAG 流程
    
    RAG 完整流程：
    1. 加载文档
    2. 分割文本
    3. 向量化
    4. 存储到向量数据库
    5. 检索相关文档
    6. 使用 LLM 生成回答
    """
    print("=" * 60)
    print("示例 8：完整的 RAG 流程")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 步骤 1: 准备文档
        print("\n步骤 1: 准备文档")
        print("-" * 40)
        
        # 创建示例文档
        sample_docs = """
        Python 数据科学生态系统
        ========================
        
        NumPy 是 Python 中用于科学计算的基础库。它提供了高性能的多维数组对象和相关的工具。
        
        Pandas 是用于数据分析和数据处理的库。它提供了 DataFrame 数据结构，类似于 Excel 表格。
        
        Matplotlib 是 Python 的基础绘图库，可以创建各种静态、动态和交互式可视化。
        
        Seaborn 是基于 Matplotlib 的高级可视化库，提供了更美观的默认样式和更高级的绘图接口。
        
        Scikit-learn 是机器学习库，提供了各种分类、回归、聚类算法和数据预处理工具。
        
        TensorFlow 是 Google 开发的深度学习框架。
        
        PyTorch 是 Facebook 开发的深度学习框架，在研究领域非常流行。
        """
        
        # 步骤 2: 分割文本
        print("\n步骤 2: 分割文本")
        print("-" * 40)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        
        chunks = text_splitter.split_text(sample_docs)
        print(f"文档被分割成 {len(chunks)} 个块")
        
        # 创建文档对象
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # 步骤 3: 向量化
        print("\n步骤 3: 向量化")
        print("-" * 40)
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # 步骤 4: 创建向量存储
        print("\n步骤 4: 创建向量存储")
        print("-" * 40)
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_pipeline_db"
        )
        print("向量存储创建成功")
        
        # 步骤 5: 创建问答链
        print("\n步骤 5: 创建问答链")
        print("-" * 40)
        
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0,
            model="gpt-3.5-turbo"
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
        )
        
        # 步骤 6: 提问
        print("\n步骤 6: 提问")
        print("-" * 40)
        
        question = "Python 中有哪些流行的深度学习框架？"
        print(f"问题: {question}")
        print()
        
        result = qa_chain.invoke({"query": question})
        print(f"答案: {result['result']}")
        
        # 清理
        import shutil
        if os.path.exists("./chroma_pipeline_db"):
            shutil.rmtree("./chroma_pipeline_db")
        print()
        
    except Exception as e:
        print(f"RAG 流程执行错误: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("*" * 60)
    print("LangChain RAG 模块学习")
    print("*" * 60)
    print("\n")
    
    # 运行 Document Loaders 示例
    example_text_loader()
    example_document_manipulation()
    
    # 运行 Text Splitters 示例
    example_character_splitter()
    example_recursive_splitter()
    
    # 运行 Embeddings 和 Vectorstores 示例
    # 注意：以下示例需要有效的 OpenAI API Key
    example_embeddings()
    example_vectorstore()
    
    # 运行 RetrievalQA 示例
    example_retrieval_qa()
    example_rag_pipeline()
    
    print("\n")
    print("*" * 60)
    print("所有示例运行完成！")
    print("*" * 60)


if __name__ == "__main__":
    main()
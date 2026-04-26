"""
LangChain Basics (refreshed for modern usage)
============================================

这一节不再以旧式 Chain 类为主，而是从今天更常见的 LCEL / Runnable 思路开始。

你会看到三件最重要的事：
1. Prompt 是怎么组织输入的
2. Model 是怎么被接到流水线里的
3. Parser 或结构化输出是怎么让结果更可用的

核心心智模型：
    prompt | model | parser
"""

# 导入注解特性，允许在类型注解中使用尚未定义的类型（前向引用）
from __future__ import annotations

# 导入 json 模块，用于 JSON 数据的编码和解码
import json
# 导入 os 模块，用于操作系统相关的功能，如读取环境变量
import os

# 导入 load_dotenv 函数，用于从 .env 文件中加载环境变量
from dotenv import load_dotenv
# 导入 Pydantic 的 BaseModel 和 Field 类，用于创建数据模型和字段配置
from pydantic import BaseModel, Field

# 导入 StrOutputParser，用于将模型输出的消息对象解析为纯字符串
from langchain_core.output_parsers import StrOutputParser
# 导入提示词模板相关类：ChatPromptTemplate（聊天模板）、MessagesPlaceholder（消息历史占位符）、PromptTemplate（基础模板）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
# 导入 OpenAI 的聊天模型接口
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量到 Python 的 os.environ 中
load_dotenv()


def print_title(title: str) -> None:
    """打印格式化的标题，用于在控制台输出中创建清晰的分隔"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """
    构建并返回一个配置好的 ChatOpenAI 实例

    参数:
        temperature: 控制模型生成随机性的参数，默认 0.2（较低，输出更确定性）

    返回:
        ChatOpenAI 实例
    """
    # 检查环境变量中是否存在 OPENAI_API_KEY，如果不存在则抛出运行时错误
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请先把 langchain-learning/.env.example 复制为 .env 并填写密钥。"
        )

    # 创建配置字典
    kwargs = {
        # 从环境变量获取模型名称，默认为 "gpt-4.1-mini"
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        # 设置温度参数，控制模型输出的随机性（0=完全确定性，1=高度随机）
        "temperature": temperature,
    }
    # 获取自定义 API 端点（可选）
    base_url = os.getenv("OPENAI_BASE_URL")
    # 如果存在自定义 URL，则添加到配置中
    if base_url:
        kwargs["base_url"] = base_url

    # 使用配置字典创建并返回 ChatOpenAI 实例
    return ChatOpenAI(**kwargs)


def example_prompt_templates() -> None:
    """
    演示 LangChain 中两种主要的提示词模板用法：
    1. PromptTemplate：基础字符串模板
    2. ChatPromptTemplate：支持多角色消息的聊天模板
    """
    print_title("示例 1：PromptTemplate 与 ChatPromptTemplate")

    # 创建基础提示词模板，使用 from_template 方法
    # {topic} 是可替换的变量占位符
    prompt = PromptTemplate.from_template(
        "请用不超过 80 个字解释 {topic}，并给一个贴近日常生活的类比。"
    )
    # 使用 format 方法将变量值填入模板
    rendered = prompt.format(topic="向量数据库")
    print("PromptTemplate 格式化结果：")
    print(rendered)

    # 创建聊天提示词模板，使用 from_messages 方法
    # 接受一个元组列表，每个元组格式为 (角色类型, 消息内容)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            # 定义系统角色消息，设置模型的行为和风格
            ("system", "你是一位善于类比的技术老师。"),
            # 创建历史消息占位符，用于注入对话历史
            MessagesPlaceholder("history"),
            # 定义用户消息，其中 {topic} 是动态变量
            ("human", "请解释：{topic}"),
        ]
    )
    # 使用 format_messages 方法生成完整的消息列表
    # 替换 {topic} 变量为 "RAG"，并提供历史对话消息
    messages = chat_prompt.format_messages(
        topic="RAG",
        history=[
            ("human", "我刚学会 prompt 和 model 的区别。"),
            ("ai", "很好，接下来可以继续理解检索增强。"),
        ],
    )

    print("\nChatPromptTemplate 生成的消息：")
    # 遍历消息列表
    for message in messages:
        # 打印每条消息的类型和内容
        print(f"- {message.type}: {message.content}")

    print(
        "\n要点：在现代 LangChain 里，prompt 不只是字符串模板，"
        "而是组织消息、历史上下文和变量注入的接口。"
    )


def example_lcel_chain() -> None:
    """
    演示 LCEL (LangChain Expression Language) 的核心用法
    LCEL 是 LangChain 的新一代工作流编排方式，使用管道操作符 | 串联组件
    """
    print_title("示例 2：最小可用 LCEL 流水线")

    # 创建聊天模型实例
    model = build_chat_model()
    # 创建聊天提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位面向初学者的 LangChain 教练。"),
            (
                "human",
                "请解释 {concept}。要求：先说一句定义，再列出 3 个关键点，最后说一个常见误区。",
            ),
        ]
    )
    # 创建 LCEL 链：这是现代 LangChain 的核心模式
    # 管道操作符 | 将左侧组件的输出传递给右侧组件的输入
    # 流程：prompt → model → StrOutputParser
    #   1. prompt：格式化提示词
    #   2. model：调用大语言模型
    #   3. StrOutputParser：解析模型输出为纯字符串
    chain = prompt | model | StrOutputParser()

    # invoke 方法触发整个流水线的执行，传入变量字典
    result = chain.invoke({"concept": "LCEL"})
    print(result)

    print(
        "\n要点：这就是今天最常见的 LangChain 句型。"
        "如果你只记住一个模式，先记住这个。"
    )


# 定义 LearningCard 数据模型，用于结构化输出
# 继承 BaseModel，创建 Pydantic 模型，提供数据验证和序列化功能
class LearningCard(BaseModel):
    """学习卡片数据模型"""
    # concept 字段：字符串类型，描述概念名称
    concept: str = Field(description="概念名称")
    # one_sentence_summary 字段：字符串类型，一句话概括
    one_sentence_summary: str = Field(description="一句话概括")
    # key_points 字段：字符串列表类型，3 到 5 条关键点
    key_points: list[str] = Field(description="3 到 5 条关键点")
    # when_to_use 字段：字符串类型，适用场景
    when_to_use: str = Field(description="适用场景")


def example_structured_output() -> None:
    """
    演示如何让模型输出结构化的数据而非自由文本
    使用 with_structured_output 方法自动将模型输出解析为指定的数据模型
    """
    print_title("示例 3：结构化输出")

    # 创建温度为 0 的模型（确定性输出）
    model = build_chat_model(temperature=0)
    # with_structured_output 方法将模型包装成输出指定格式的模型
    # 这会自动将模型的输出解析为 LearningCard 对象
    structured_model = model.with_structured_output(LearningCard)

    # 调用结构化模型，生成学习卡片
    card = structured_model.invoke(
        "请生成一张关于 'tool calling' 的学习卡片，面向刚开始学 LangChain 的 Python 开发者。"
    )

    # 打印结构化结果
    # card.model_dump()：将 Pydantic 模型转换为 Python 字典
    # json.dumps(..., ensure_ascii=False, indent=2)：格式化为 JSON 字符串，保留中文，缩进2空格
    print(json.dumps(card.model_dump(), ensure_ascii=False, indent=2))

    print(
        "\n要点：相比手写 JSON 提示词，`with_structured_output(...)` 更稳，"
        "也更符合今天模型能力的用法。"
    )


def example_batch_reasoning() -> None:
    """
    演示如何批量处理多个相似任务，提高效率
    使用 chain.batch() 方法一次性处理多个输入，相比循环调用通常更高效
    """
    print_title("示例 4：批量处理同一类任务")

    # 创建聊天模型实例
    model = build_chat_model()
    # 创建聊天提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位精炼的 AI 概念讲解员。"),
            ("human", "请用一句话说明 {topic} 在 LangChain 里的作用。"),
        ]
    )
    # 创建完整的 LCEL 链：prompt | model | StrOutputParser
    chain = prompt | model | StrOutputParser()

    # 创建批量输入列表，每个字典都有 topic 键
    topics = [
        {"topic": "PromptTemplate"},
        {"topic": "RunnableParallel"},
        {"topic": "Retriever"},
    ]
    # 批量调用链，一次性处理多个输入
    # 相比循环调用，这通常更高效，可能并发执行
    outputs = chain.batch(topics)

    # 打印所有结果
    # zip 函数将主题和输出配对
    for topic, output in zip(topics, outputs):
        print(f"- {topic['topic']}: {output}")

    print(
        "\n要点：当输入结构相同、任务彼此独立时，`batch(...)` 往往比手写循环更自然。"
    )


def main() -> None:
    """主函数，程序入口点"""
    # 打印主标题
    print_title("LangChain 基础模块（现代版）")
    # 调用提示词模板示例（不需要 API 密钥）
    example_prompt_templates()

    try:
        # 调用三个需要模型 API 的示例函数
        example_lcel_chain()
        example_structured_output()
        example_batch_reasoning()
    except RuntimeError as exc:
        # 捕获 RuntimeError（通常是 API 密钥缺失），并打印友好提示
        print(exc)
        print("已跳过需要在线模型的示例。")


# Python 惯用法：检查是否直接运行此脚本（而不是被导入）
if __name__ == "__main__":
    # 如果是，则调用 main 函数
    main()

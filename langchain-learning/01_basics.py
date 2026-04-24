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

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_chat_model(temperature: float = 0.2) -> ChatOpenAI:
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


def example_prompt_templates() -> None:
    print_title("示例 1：PromptTemplate 与 ChatPromptTemplate")

    prompt = PromptTemplate.from_template(
        "请用不超过 80 个字解释 {topic}，并给一个贴近日常生活的类比。"
    )
    rendered = prompt.format(topic="向量数据库")
    print("PromptTemplate 格式化结果：")
    print(rendered)

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位善于类比的技术老师。"),
            MessagesPlaceholder("history"),
            ("human", "请解释：{topic}"),
        ]
    )
    messages = chat_prompt.format_messages(
        topic="RAG",
        history=[
            ("human", "我刚学会 prompt 和 model 的区别。"),
            ("ai", "很好，接下来可以继续理解检索增强。"),
        ],
    )

    print("\nChatPromptTemplate 生成的消息：")
    for message in messages:
        print(f"- {message.type}: {message.content}")

    print(
        "\n要点：在现代 LangChain 里，prompt 不只是字符串模板，"
        "而是组织消息、历史上下文和变量注入的接口。"
    )


def example_lcel_chain() -> None:
    print_title("示例 2：最小可用 LCEL 流水线")

    model = build_chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位面向初学者的 LangChain 教练。"),
            (
                "human",
                "请解释 {concept}。要求：先说一句定义，再列出 3 个关键点，最后说一个常见误区。",
            ),
        ]
    )
    chain = prompt | model | StrOutputParser()

    result = chain.invoke({"concept": "LCEL"})
    print(result)

    print(
        "\n要点：这就是今天最常见的 LangChain 句型。"
        "如果你只记住一个模式，先记住这个。"
    )


class LearningCard(BaseModel):
    concept: str = Field(description="概念名称")
    one_sentence_summary: str = Field(description="一句话概括")
    key_points: list[str] = Field(description="3 到 5 条关键点")
    when_to_use: str = Field(description="适用场景")


def example_structured_output() -> None:
    print_title("示例 3：结构化输出")

    model = build_chat_model(temperature=0)
    structured_model = model.with_structured_output(LearningCard)

    card = structured_model.invoke(
        "请生成一张关于 'tool calling' 的学习卡片，面向刚开始学 LangChain 的 Python 开发者。"
    )

    print(json.dumps(card.model_dump(), ensure_ascii=False, indent=2))

    print(
        "\n要点：相比手写 JSON 提示词，`with_structured_output(...)` 更稳，"
        "也更符合今天模型能力的用法。"
    )


def example_batch_reasoning() -> None:
    print_title("示例 4：批量处理同一类任务")

    model = build_chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位精炼的 AI 概念讲解员。"),
            ("human", "请用一句话说明 {topic} 在 LangChain 里的作用。"),
        ]
    )
    chain = prompt | model | StrOutputParser()

    topics = [
        {"topic": "PromptTemplate"},
        {"topic": "RunnableParallel"},
        {"topic": "Retriever"},
    ]
    outputs = chain.batch(topics)

    for topic, output in zip(topics, outputs):
        print(f"- {topic['topic']}: {output}")

    print(
        "\n要点：当输入结构相同、任务彼此独立时，`batch(...)` 往往比手写循环更自然。"
    )


def main() -> None:
    print_title("LangChain 基础模块（现代版）")
    example_prompt_templates()

    try:
        example_lcel_chain()
        example_structured_output()
        example_batch_reasoning()
    except RuntimeError as exc:
        print(exc)
        print("已跳过需要在线模型的示例。")


if __name__ == "__main__":
    main()

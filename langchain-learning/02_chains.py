"""
LangChain Chains / Runnables (modern view)
=========================================

这一节不把重点放在老式 `SequentialChain` 上，而是展示今天更值得优先理解的编排方式：

1. 顺序组合
2. 并行组合
3. 轻量路由

它们都围绕 Runnable / LCEL 展开。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
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


def example_sequential_composition() -> None:
    print_title("示例 1：顺序组合")

    model = build_chat_model()
    parser = StrOutputParser()

    outline_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位擅长教学设计的讲师。"),
            ("human", "请围绕 {topic} 生成一个三点式学习提纲。"),
        ]
    )
    outline_chain = outline_prompt | model | parser

    lesson_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位把抽象概念讲得很具体的老师。"),
            (
                "human",
                "主题：{topic}\n"
                "提纲：{outline}\n\n"
                "请把这个提纲扩写成一段适合初学者阅读的讲解，并加入一个现实类比。",
            ),
        ]
    )
    lesson_chain = (
        {
            "topic": RunnablePassthrough(),
            "outline": outline_chain,
        }
        | lesson_prompt
        | model
        | parser
    )

    result = lesson_chain.invoke("RAG")
    print(result)

    print("\n要点：上一步的输出可以自然流向下一步，而不是必须套进旧式 Chain 类。")


def example_parallel_composition() -> None:
    print_title("示例 2：并行组合")

    model = build_chat_model()
    parser = StrOutputParser()

    summary_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "你是一位精炼的内容编辑。"),
                ("human", "请把下面内容概括成 2 句话：\n\n{content}"),
            ]
        )
        | model
        | parser
    )

    highlights_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "你是一位课程助教。"),
                ("human", "请从下面内容中提炼 3 条重点：\n\n{content}"),
            ]
        )
        | model
        | parser
    )

    parallel_chain = RunnableParallel(
        summary=summary_chain,
        highlights=highlights_chain,
    )

    content = (
        "LangChain 的价值不只是调用模型，而是把 prompt、模型、检索、工具、状态、"
        "结构化输出等能力组织成一个可维护的工作流。"
        "当项目复杂度上来时，真正重要的是编排和状态管理。"
    )
    result = parallel_chain.invoke({"content": content})

    print("摘要：")
    print(result["summary"])
    print("\n重点：")
    print(result["highlights"])

    print("\n要点：彼此独立的子任务适合并行表达，代码也更清楚。")


def example_routing() -> None:
    print_title("示例 3：轻量路由")

    model = build_chat_model(temperature=0)
    parser = StrOutputParser()

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个路由器。只输出 math、code、general 三者之一。"),
            ("human", "问题：{question}"),
        ]
    )
    route_chain = route_prompt | model | parser | RunnableLambda(lambda text: text.strip().lower())

    math_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "你是一位数学老师。"),
                ("human", "请分步骤解答：{question}"),
            ]
        )
        | model
        | parser
    )

    code_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "你是一位 Python 工程师。"),
                ("human", "请给出 Python 视角的回答：{question}"),
            ]
        )
        | model
        | parser
    )

    general_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "你是一位通识助教。"),
                ("human", "请清晰回答：{question}"),
            ]
        )
        | model
        | parser
    )

    answer_chain = (
        {
            "route": route_chain,
            "question": RunnableLambda(lambda data: data["question"]),
        }
        | RunnableBranch(
            (lambda data: data["route"] == "math", math_chain),
            (lambda data: data["route"] == "code", code_chain),
            general_chain,
        )
    )

    questions = [
        "如何计算圆的面积？",
        "Python 里 dataclass 有什么作用？",
        "为什么 RAG 对企业问答系统有帮助？",
    ]

    for question in questions:
        route = route_chain.invoke({"question": question})
        answer = answer_chain.invoke({"question": question})
        print(f"\n问题：{question}")
        print(f"路由结果：{route}")
        print(f"回答：{answer}")

    print(
        "\n要点：很多所谓“复杂 RouterChain”场景，今天用 RunnableBranch 就已经足够清爽。"
    )


def main() -> None:
    print_title("LangChain 编排模块（现代版）")

    try:
        example_sequential_composition()
        example_parallel_composition()
        example_routing()
    except RuntimeError as exc:
        print(exc)
        print("已跳过需要在线模型的示例。")


if __name__ == "__main__":
    main()

"""
Modern Agents with LangChain
============================

这一节聚焦今天更值得新手掌握的 agent 用法：

1. 用 `@tool` 定义工具
2. 用 `create_agent(...)` 创建 agent
3. 让 agent 做工具调用
4. 让 agent 输出结构化结果

相比很多旧教程，这里不把 `initialize_agent(...)` 当作主角。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

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


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""

    return a * b


@tool
def get_today_date() -> str:
    """获取当前日期。"""

    return datetime.now().strftime("%Y-%m-%d")


@tool
def search_langchain_notes(topic: str) -> str:
    """查询一个小型本地笔记库，适合回答 LangChain 基础概念问题。"""

    notes = {
        "lcel": "LCEL 是 LangChain Expression Language，用来把 runnable 组合成工作流。",
        "rag": "RAG 是先检索后生成，适合接外部知识库。",
        "agent": "Agent 用于需要动态决策、调用工具、步骤不固定的任务。",
        "memory": "现代记忆更应从消息历史和线程状态理解，而不只是旧式 Memory 类。",
        "tool calling": "Tool calling 让模型可以可靠地选择并调用函数，而不是只在文本里假装会用工具。",
    }
    return notes.get(topic.lower(), f"笔记库中暂时没有 {topic} 的说明。")


def example_tools() -> None:
    print_title("示例 1：工具本身先能独立工作")

    print(f"multiply(17, 19) -> {multiply.invoke({'a': 17, 'b': 19})}")
    print(f"get_today_date() -> {get_today_date.invoke({})}")
    print(
        "search_langchain_notes('tool calling') -> "
        f"{search_langchain_notes.invoke({'topic': 'tool calling'})}"
    )

    print(
        "\n要点：agent 只是更高一层的决策器。先确认工具本身语义清楚、输入输出稳定。"
    )


def example_tool_calling_agent() -> None:
    print_title("示例 2：工具调用型 Agent")

    agent = create_agent(
        model=build_chat_model(temperature=0),
        tools=[multiply, get_today_date, search_langchain_notes],
        system_prompt=(
            "你是一位谨慎、会调用工具的 LangChain 助教。"
            "需要计算就调用计算工具，需要日期就调用日期工具，"
            "需要概念说明就查询本地笔记库。"
        ),
    )

    user_question = (
        "今天是几号？顺便帮我算一下 17 * 19，"
        "然后再解释一下为什么 tool calling 是现代 agent 的重要能力。"
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_question}]}
    )

    print(f"用户问题：{user_question}")
    print("\nAgent 最后一条消息：")
    print(result["messages"][-1].content)

    print(
        "\n要点：现代 agent 的关键不在“看起来像在思考”，"
        "而在于它是否能稳定地决定什么时候该调用什么工具。"
    )


class StudyPlan(BaseModel):
    topic: str = Field(description="学习主题")
    difficulty: str = Field(description="难度级别")
    daily_steps: list[str] = Field(description="按天拆解的学习步骤")
    pitfalls: list[str] = Field(description="容易踩的坑")


def example_structured_output_agent() -> None:
    print_title("示例 3：结构化输出型 Agent")

    agent = create_agent(
        model=build_chat_model(temperature=0),
        tools=[search_langchain_notes],
        system_prompt=(
            "你是一位课程设计助教。必要时可查询本地笔记库，"
            "并把结果整理成结构化学习计划。"
        ),
        response_format=StudyPlan,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "给我一个 5 天的 LangChain 入门计划，重点覆盖 LCEL、RAG、Agent。",
                }
            ]
        }
    )

    structured = result["structured_response"]
    print(json.dumps(structured.model_dump(), ensure_ascii=False, indent=2))

    print(
        "\n要点：当你的下游还要继续消费结果时，结构化输出比长篇自然语言更适合工程落地。"
    )


def main() -> None:
    print_title("LangChain Agent 模块（现代版）")
    example_tools()

    try:
        example_tool_calling_agent()
        example_structured_output_agent()
    except RuntimeError as exc:
        print(exc)
        print("已跳过需要在线模型的示例。")


if __name__ == "__main__":
    main()

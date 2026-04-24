"""
LangChain Memory / State (modern view)
=====================================

这一节最重要的更新是：

1. “记忆”不只是旧式 Memory 类
2. 新项目更应理解消息历史与线程状态
3. Agent 的短期记忆常常通过 LangGraph checkpointer 提供

你仍然会在老代码里看到 ConversationBufferMemory 这些类，
但今天更值得优先掌握的是下面两条主线：

- RunnableWithMessageHistory
- create_agent(..., checkpointer=...)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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


def example_message_history() -> list[BaseMessage]:
    print_title("示例 1：RunnableWithMessageHistory")

    model = build_chat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位会记住上下文的学习伙伴。"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | model

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "demo-user"}}

    first = chain_with_history.invoke(
        {"question": "我叫阿青，我最近在学 LangChain。"},
        config=config,
    )
    second = chain_with_history.invoke(
        {"question": "你记得我叫什么吗？我最近在学什么？"},
        config=config,
    )

    print("第 1 次回复：")
    print(first.content)
    print("\n第 2 次回复：")
    print(second.content)

    history = get_session_history("demo-user").messages
    print(f"\n当前会话累计消息数：{len(history)}")
    for index, message in enumerate(history, start=1):
        print(f"{index}. {message.type}: {message.content}")

    print("\n要点：这里的“记忆”本质上是会话级消息历史。")
    return history


@tool
def concept_lookup(term: str) -> str:
    """查询一个极小型本地知识库，用于演示 agent 的工具调用。"""

    knowledge = {
        "rag": "RAG 是检索增强生成：先检索外部知识，再把结果交给模型生成回答。",
        "lcel": "LCEL 指的是 LangChain Expression Language，也就是基于 runnable 的组合表达方式。",
        "agent": "Agent 会根据目标自主决定是否调用工具、按什么顺序行动。",
    }
    return knowledge.get(term.lower(), f"本地知识库里暂时没有 {term} 的定义。")


def example_agent_short_term_memory() -> None:
    print_title("示例 2：Agent 的短期记忆（LangGraph checkpointer）")

    model = build_chat_model(temperature=0)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[concept_lookup],
        system_prompt=(
            "你是一位教学型 agent。你可以记住同一线程里的对话，"
            "必要时使用工具补充定义。"
        ),
        checkpointer=checkpointer,
    )

    shared_config = {"configurable": {"thread_id": "student-1"}}

    result_1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫阿青，我最近重点在学 RAG。",
                }
            ]
        },
        config=shared_config,
    )
    result_2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫什么？我最近在学什么？顺便帮我查一下 RAG 是什么。",
                }
            ]
        },
        config=shared_config,
    )
    result_3 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫什么？",
                }
            ]
        },
        config={"configurable": {"thread_id": "another-thread"}},
    )

    print("线程 student-1 第一次调用的最后一条消息：")
    print(result_1["messages"][-1].content)

    print("\n线程 student-1 第二次调用的最后一条消息：")
    print(result_2["messages"][-1].content)

    print("\n切换到 another-thread 后的最后一条消息：")
    print(result_3["messages"][-1].content)

    print(
        "\n要点：Agent 的记忆是和 thread_id 绑定的。"
        "这比把“记忆”理解成一个孤立组件更贴近现在的工程实践。"
    )


def explain_message_history_shape(messages: list[BaseMessage]) -> None:
    print_title("补充：你真正保存下来的是什么")
    for index, message in enumerate(messages, start=1):
        print(f"{index}. {message.type}: {message.content}")
    print(
        "\n无论是普通聊天链还是 agent，真正重要的往往都是消息列表、线程 id、"
        "以及这些状态什么时候被读取、什么时候被写回。"
    )


def main() -> None:
    print_title("LangChain 记忆与状态模块（现代版）")

    try:
        history = example_message_history()
        example_agent_short_term_memory()
    except RuntimeError as exc:
        print(exc)
        print("已跳过需要在线模型的示例。")
        return

    explain_message_history_shape(history)


if __name__ == "__main__":
    main()

"""
LangChain Memory / State (modern view)
=====================================

这一节最重要的更新是：

1. "记忆"不只是旧式 Memory 类
2. 新项目更应理解消息历史与线程状态
3. Agent 的短期记忆常常通过 LangGraph checkpointer 提供

你仍然会在老代码里看到 ConversationBufferMemory 这些类，
但今天更值得优先掌握的是下面两条主线：

- RunnableWithMessageHistory
- create_agent(..., checkpointer=...)
"""

# 导入注解特性，支持前向引用
from __future__ import annotations

# 导入 os 模块，用于环境变量操作
import os

# 导入 load_dotenv，用于加载 .env 文件中的环境变量
from dotenv import load_dotenv
# 导入 InMemorySaver，LangGraph 的内存检查点保存器，用于持久化 agent 的状态
from langgraph.checkpoint.memory import InMemorySaver

# 导入 create_agent，用于创建 agent
from langchain.agents import create_agent
# 导入 tool 装饰器，用于将函数转换为可被 agent 调用的工具
from langchain.tools import tool
# 导入 InMemoryChatMessageHistory，内存中的聊天消息历史存储
from langchain_core.chat_history import InMemoryChatMessageHistory
# 导入 BaseMessage，所有消息类型的基类
from langchain_core.messages import BaseMessage
# 导入提示词模板相关类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 导入 RunnableWithMessageHistory，为链添加消息历史功能的包装器
from langchain_core.runnables.history import RunnableWithMessageHistory
# 导入 OpenAI 聊天模型
from langchain_openai import ChatOpenAI

# 加载环境变量
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


def example_message_history() -> list[BaseMessage]:
    """
    演示 RunnableWithMessageHistory 的用法
    这里的"记忆"本质上是会话级消息历史，每个 session_id 对应独立的历史记录

    返回:
        list[BaseMessage]: 会话消息历史列表
    """
    print_title("示例 1：RunnableWithMessageHistory")

    # 创建聊天模型实例
    model = build_chat_model()

    # 创建聊天提示词模板，包含消息历史占位符
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位会记住上下文的学习伙伴。"),
            MessagesPlaceholder("history"),  # 消息历史占位符
            ("human", "{question}"),
        ]
    )
    # 创建基础链：提示词 → 模型
    chain = prompt | model

    # 创建内存存储，用于存储不同会话的消息历史
    # 字典的键是 session_id，值是对应的 InMemoryChatMessageHistory 实例
    store: dict[str, InMemoryChatMessageHistory] = {}

    # 定义获取会话历史的函数
    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        """
        根据会话 ID 获取或创建消息历史

        参数:
            session_id: 会话 ID，用于区分不同的用户会话

        返回:
            InMemoryChatMessageHistory: 该会话的消息历史对象
        """
        # 如果会话不存在，则创建新的消息历史
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        # 返回该会话的消息历史
        return store[session_id]

    # 创建带有消息历史的链
    # RunnableWithMessageHistory 会自动管理消息历史：
    #   - 调用前：从 get_session_history 获取历史消息
    #   - 调用后：将新消息添加到历史中
    chain_with_history = RunnableWithMessageHistory(
        chain,                                    # 要包装的基础链
        get_session_history,                      # 获取历史消息的函数
        input_messages_key="question",           # 输入消息在输入字典中的键名
        history_messages_key="history",          # 历史消息在提示词模板中的占位符名
    )

    # 创建配置字典，指定会话 ID
    config = {"configurable": {"session_id": "demo-user"}}

    # 第一次调用：介绍自己的名字和学习内容
    first = chain_with_history.invoke(
        {"question": "我叫阿青，我最近在学 LangChain。"},
        config=config,
    )
    # 第二次调用：询问是否记得之前的信息（测试记忆功能）
    second = chain_with_history.invoke(
        {"question": "你记得我叫什么吗？我最近在学什么？"},
        config=config,
    )

    print("第 1 次回复：")
    print(first.content)
    print("\n第 2 次回复：")
    print(second.content)

    # 获取并打印当前会话的所有消息
    history = get_session_history("demo-user").messages
    print(f"\n当前会话累计消息数：{len(history)}")
    for index, message in enumerate(history, start=1):
        print(f"{index}. {message.type}: {message.content}")

    print("\n要点：这里的"记忆"本质上是会话级消息历史。")
    return history


# 定义工具函数，使用 @tool 装饰器将函数转换为可被 agent 调用的工具
@tool
def concept_lookup(term: str) -> str:
    """
    查询一个极小型本地知识库，用于演示 agent 的工具调用

    参数:
        term: 要查询的概念术语

    返回:
        str: 概念的定义或说明
    """
    # 本地知识库字典
    knowledge = {
        "rag": "RAG 是检索增强生成：先检索外部知识，再把结果交给模型生成回答。",
        "lcel": "LCEL 指的是 LangChain Expression Language，也就是基于 runnable 的组合表达方式。",
        "agent": "Agent 会根据目标自主决定是否调用工具、按什么顺序行动。",
    }
    # 从知识库获取定义，如果不存在则返回提示信息
    return knowledge.get(term.lower(), f"本地知识库里暂时没有 {term} 的定义。")


def example_agent_short_term_memory() -> None:
    """
    演示 Agent 的短期记忆功能
    使用 LangGraph 的 checkpointer 来持久化 agent 的状态，记忆与 thread_id 绑定
    """
    print_title("示例 2：Agent 的短期记忆（LangGraph checkpointer）")

    # 创建温度为 0 的模型（确定性输出）
    model = build_chat_model(temperature=0)
    # 创建内存检查点保存器，用于持久化 agent 的状态
    # 这就是 agent 的"记忆"机制：每次调用后状态会被保存到 checkpointer
    checkpointer = InMemorySaver()

    # 创建 agent
    agent = create_agent(
        model=model,                        # 使用的模型
        tools=[concept_lookup],            # agent 可以调用的工具列表
        system_prompt=(                     # 系统提示词，定义 agent 的行为
            "你是一位教学型 agent。你可以记住同一线程里的对话，"
            "必要时使用工具补充定义。"
        ),
        checkpointer=checkpointer,         # 指定检查点保存器，启用记忆功能
    )

    # 创建共享配置，指定线程 ID
    # thread_id 是 agent 记忆的标识符，不同 thread_id 对应不同的记忆空间
    shared_config = {"configurable": {"thread_id": "student-1"}}

    # 第一次调用：介绍自己的名字和学习重点
    result_1 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫阿青，我最近重点在学 RAG。",
                }
            ]
        },
        config=shared_config,  # 使用相同的 config，确保使用同一个记忆空间
    )
    # 第二次调用：询问是否记得之前的信息，并要求查询 RAG 的定义
    # agent 应该能从记忆中获取之前的对话，并调用工具查询 RAG 的定义
    result_2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫什么？我最近在学什么？顺便帮我查一下 RAG 是什么。",
                }
            ]
        },
        config=shared_config,  # 使用相同的 config，访问同一个记忆空间
    )
    # 第三次调用：使用不同的 thread_id
    # 应该返回"不知道"，因为这是一个全新的记忆空间
    result_3 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫什么？",
                }
            ]
        },
        config={"configurable": {"thread_id": "another-thread"}},  # 不同的 thread_id
    )

    # 打印第一次调用的最后一条消息
    print("线程 student-1 第一次调用的最后一条消息：")
    print(result_1["messages"][-1].content)

    # 打印第二次调用的最后一条消息
    print("\n线程 student-1 第二次调用的最后一条消息：")
    print(result_2["messages"][-1].content)

    # 打印第三次调用（不同线程）的最后一条消息
    print("\n切换到 another-thread 后的最后一条消息：")
    print(result_3["messages"][-1].content)

    print(
        "\n要点：Agent 的记忆是和 thread_id 绑定的。"
        "这比把"记忆"理解成一个孤立组件更贴近现在的工程实践。"
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
    """
    补充说明：展示真正保存下来的消息历史是什么样子的

    参数:
        messages: 消息列表
    """
    print_title("补充：你真正保存下来的是什么")
    # 遍历并打印每条消息的类型和内容
    for index, message in enumerate(messages, start=1):
        print(f"{index}. {message.type}: {message.content}")
    print(
        "\n无论是普通聊天链还是 agent，真正重要的往往都是消息列表、线程 id、"
        "以及这些状态什么时候被读取、什么时候被写回。"
    )


def main() -> None:
    """主函数，程序入口点"""
    # 打印主标题
    print_title("LangChain 记忆与状态模块（现代版）")

    try:
        # 调用消息历史示例，获取历史消息列表
        history = example_message_history()
        # 调用 agent 短期记忆示例
        example_agent_short_term_memory()
    except RuntimeError as exc:
        # 捕获 RuntimeError（通常是 API 密钥缺失），并打印友好提示
        print(exc)
        print("已跳过需要在线模型的示例。")
        return

    # 打印消息历史的具体形态
    explain_message_history_shape(history)


# Python 惯用法：检查是否直接运行此脚本
if __name__ == "__main__":
    main()

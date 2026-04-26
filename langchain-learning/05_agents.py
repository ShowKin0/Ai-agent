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

# 导入注解特性，支持前向引用
from __future__ import annotations

# 导入 json 模块，用于 JSON 数据的编码和解码
import json
# 导入 os 模块，用于环境变量操作
import os
# 导入 datetime，用于获取当前日期时间
from datetime import datetime

# 导入 load_dotenv，用于加载 .env 文件中的环境变量
from dotenv import load_dotenv
# 导入 Pydantic 的 BaseModel 和 Field 类，用于创建数据模型和字段配置
from pydantic import BaseModel, Field

# 导入 create_agent，用于创建 agent
from langchain.agents import create_agent
# 导入 tool 装饰器，用于将函数转换为可被 agent 调用的工具
from langchain.tools import tool
# 导入 OpenAI 聊天模型
from langchain_openai import ChatOpenAI

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


# 定义工具函数，使用 @tool 装饰器将函数转换为可被 agent 调用的工具
@tool
def multiply(a: int, b: int) -> int:
    """
    计算两个整数的乘积

    参数:
        a: 第一个整数
        b: 第二个整数

    返回:
        int: 两个整数的乘积
    """
    return a * b


# 定义工具函数：获取当前日期
@tool
def get_today_date() -> str:
    """
    获取当前日期

    返回:
        str: 当前日期，格式为 YYYY-MM-DD
    """
    # 获取当前时间并格式化为字符串
    return datetime.now().strftime("%Y-%m-%d")


# 定义工具函数：查询 LangChain 笔记库
@tool
def search_langchain_notes(topic: str) -> str:
    """
    查询一个小型本地笔记库，适合回答 LangChain 基础概念问题

    参数:
        topic: 要查询的主题

    返回:
        str: 主题的说明或提示信息
    """
    # 本地笔记库字典
    notes = {
        "lcel": "LCEL 是 LangChain Expression Language，用来把 runnable 组合成工作流。",
        "rag": "RAG 是先检索后生成，适合接外部知识库。",
        "agent": "Agent 用于需要动态决策、调用工具、步骤不固定的任务。",
        "memory": "现代记忆更应从消息历史和线程状态理解，而不只是旧式 Memory 类。",
        "tool calling": "Tool calling 让模型可以可靠地选择并调用函数，而不是只在文本里假装会用工具。",
    }
    # 从笔记库获取说明，如果不存在则返回提示信息
    return notes.get(topic.lower(), f"笔记库中暂时没有 {topic} 的说明。")


def example_tools() -> None:
    """
    演示工具本身的独立调用
    agent 只是更高一层的决策器，先确认工具本身语义清楚、输入输出稳定
    """
    print_title("示例 1：工具本身先能独立工作")

    # 直接调用 multiply 工具
    print(f"multiply(17, 19) -> {multiply.invoke({'a': 17, 'b': 19})}")
    # 直接调用 get_today_date 工具
    print(f"get_today_date() -> {get_today_date.invoke({})}")
    # 直接调用 search_langchain_notes 工具
    print(
        "search_langchain_notes('tool calling') -> "
        f"{search_langchain_notes.invoke({'topic': 'tool calling'})}"
    )

    print(
        "\n要点：agent 只是更高一层的决策器。先确认工具本身语义清楚、输入输出稳定。"
    )


def example_tool_calling_agent() -> None:
    """
    演示工具调用型 agent 的用法
    agent 会根据用户需求自主决定是否调用工具、调用哪些工具
    """
    print_title("示例 2：工具调用型 Agent")

    # 创建 agent
    agent = create_agent(
        model=build_chat_model(temperature=0),  # 使用温度为 0 的模型
        tools=[multiply, get_today_date, search_langchain_notes],  # 提供的工具列表
        system_prompt=(                              # 系统提示词，定义 agent 的行为
            "你是一位谨慎、会调用工具的 LangChain 助教。"
            "需要计算就调用计算工具，需要日期就调用日期工具，"
            "需要概念说明就查询本地笔记库。"
        ),
    )

    # 定义用户问题，包含多个任务
    user_question = (
        "今天是几号？顺便帮我算一下 17 * 19，"
        "然后再解释一下为什么 tool calling 是现代 agent 的重要能力。"
    )
    # 调用 agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_question}]}
    )

    # 打印用户问题和 agent 的回复
    print(f"用户问题：{user_question}")
    print("\nAgent 最后一条消息：")
    print(result["messages"][-1].content)

    print(
        "\n要点：现代 agent 的关键不在看起来像在思考"
        "而在于它是否能稳定地决定什么时候该调用什么工具。"
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


# 定义学习计划数据模型，用于结构化输出
class StudyPlan(BaseModel):
    """学习计划数据模型"""
    topic: str = Field(description="学习主题")
    difficulty: str = Field(description="难度级别")
    daily_steps: list[str] = Field(description="按天拆解的学习步骤")
    pitfalls: list[str] = Field(description="容易踩的坑")


def example_structured_output_agent() -> None:
    """
    演示结构化输出型 agent 的用法
    agent 不仅调用工具，还会将结果整理成结构化的数据格式
    当下游还要继续消费结果时，结构化输出比长篇自然语言更适合工程落地
    """
    print_title("示例 3：结构化输出型 Agent")

    # 创建 agent，指定 response_format 为 StudyPlan
    agent = create_agent(
        model=build_chat_model(temperature=0),  # 使用温度为 0 的模型
        tools=[search_langchain_notes],         # 提供的工具列表
        system_prompt=(                         # 系统提示词
            "你是一位课程设计助教。必要时可查询本地笔记库，"
            "并把结果整理成结构化学习计划。"
        ),
        response_format=StudyPlan,  # 指定输出格式为 StudyPlan 数据模型
    )

    # 调用 agent
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

    # 提取结构化响应
    structured = result["structured_response"]
    # 打印格式化的 JSON
    print(json.dumps(structured.model_dump(), ensure_ascii=False, indent=2))

    print(
        "\n要点：当你的下游还要继续消费结果时，结构化输出比长篇自然语言更适合工程落地。"
    )


def main() -> None:
    """主函数，程序入口点"""
    # 打印主标题
    print_title("LangChain Agent 模块（现代版）")
    # 调用工具示例（不需要 API 密钥）
    example_tools()

    try:
        # 调用工具调用型 agent 示例
        example_tool_calling_agent()
        # 调用结构化输出型 agent 示例
        example_structured_output_agent()
    except RuntimeError as exc:
        # 捕获 RuntimeError（通常是 API 密钥缺失），并打印友好提示
        print(exc)
        print("已跳过需要在线模型的示例。")


# Python 惯用法：检查是否直接运行此脚本
if __name__ == "__main__":
    main()

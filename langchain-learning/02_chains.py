"""
LangChain Chains / Runnables (modern view)
=========================================

这一节不把重点放在老式 `SequentialChain` 上，而是展示今天更值得优先理解的编排方式：

1. 顺序组合
2. 并行组合
3. 轻量路由

它们都围绕 Runnable / LCEL 展开。
"""

# 导入注解特性，支持前向引用
from __future__ import annotations

# 导入 os 模块，用于环境变量操作
import os

# 导入 load_dotenv，用于加载 .env 文件中的环境变量
from dotenv import load_dotenv

# 导入 StrOutputParser，将模型输出解析为纯字符串
from langchain_core.output_parsers import StrOutputParser
# 导入 ChatPromptTemplate，用于创建聊天提示词模板
from langchain_core.prompts import ChatPromptTemplate
# 导入 LangChain 核心的 Runnable 组件
# RunnableBranch：分支组件，根据条件选择不同的执行路径（路由功能）
# RunnableLambda：将普通 Python 函数包装为 Runnable，使其可以加入 LCEL 链
# RunnableParallel：并行组件，同时执行多个独立的 Runnable 并合并结果
# RunnablePassthrough：透传组件，将输入原样传递到输出中，常用于在并行时保留原始输入
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
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


def example_sequential_composition() -> None:
    """
    演示如何使用 LCEL 管道操作符串联多个处理步骤
    上一步的输出可以自然流向下一步，而不是必须套进旧式 Chain 类
    """
    print_title("示例 1：顺序组合")

    # 创建聊天模型实例
    model = build_chat_model()
    # 创建字符串输出解析器
    parser = StrOutputParser()

    # 创建提纲生成提示词模板
    outline_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位擅长教学设计的讲师。"),
            ("human", "请围绕 {topic} 生成一个三点式学习提纲。"),
        ]
    )
    # 创建提纲生成链：提示词 → 模型 → 解析器
    outline_chain = outline_prompt | model | parser

    # 创建详细讲解提示词模板
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
    # 创建详细讲解链
    # 使用字典和 RunnablePassthrough 组合输入：
    #   - topic: 使用 RunnablePassthrough() 透传原始输入
    #   - outline: 通过 outline_chain 生成提纲
    # 然后将这个字典传递给 lesson_prompt，再经过 model 和 parser
    lesson_chain = (
        {
            "topic": RunnablePassthrough(),  # 透传原始主题到下一步
            "outline": outline_chain,         # 使用 outline_chain 生成提纲
        }
        | lesson_prompt  # 格式化提示词
        | model          # 调用模型
        | parser         # 解析输出
    )

    # 调用链，传入主题 "RAG"
    result = lesson_chain.invoke("RAG")
    print(result)

    print("\n要点：上一步的输出可以自然流向下一步，而不是必须套进旧式 Chain 类。")


def example_parallel_composition() -> None:
    """
    演示如何使用 RunnableParallel 并行执行多个独立的子任务
    彼此独立的子任务适合并行表达，代码也更清楚
    """
    print_title("示例 2：并行组合")

    # 创建聊天模型实例
    model = build_chat_model()
    # 创建字符串输出解析器
    parser = StrOutputParser()

    # 创建摘要生成链：将内容概括成 2 句话
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

    # 创建重点提取链：从内容中提炼 3 条重点
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

    # 创建并行链，同时执行 summary_chain 和 highlights_chain
    # RunnableParallel 会并行执行多个链，并将结果合并成一个字典
    # 字典的键是 "summary" 和 "highlights"，对应的值是各链的输出
    parallel_chain = RunnableParallel(
        summary=summary_chain,
        highlights=highlights_chain,
    )

    # 定义要处理的内容
    content = (
        "LangChain 的价值不只是调用模型，而是把 prompt、模型、检索、工具、状态、"
        "结构化输出等能力组织成一个可维护的工作流。"
        "当项目复杂度上来时，真正重要的是编排和状态管理。"
    )
    # 调用并行链，传入包含 content 的字典
    result = parallel_chain.invoke({"content": content})

    # 打印摘要
    print("摘要：")
    print(result["summary"])
    # 打印重点
    print("\n重点：")
    print(result["highlights"])

    print("\n要点：彼此独立的子任务适合并行表达，代码也更清楚。")


def example_routing() -> None:
    """
    演示如何使用 RunnableBranch 实现轻量级路由
    根据问题类型（数学、编程、通用）选择不同的处理链
    """
    print_title("示例 3：轻量路由")

    # 创建温度为 0 的模型（确定性输出）
    model = build_chat_model(temperature=0)
    # 创建字符串输出解析器
    parser = StrOutputParser()

    # 创建路由提示词模板，用于判断问题类型
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个路由器。只输出 math、code、general 三者之一。"),
            ("human", "问题：{question}"),
        ]
    )
    # 创建路由链：提示词 → 模型 → 解析器 → 清理输出（去除空格并转小写）
    # RunnableLambda 将普通函数包装成 Runnable
    route_chain = route_prompt | model | parser | RunnableLambda(lambda text: text.strip().lower())

    # 创建数学问题处理链
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

    # 创建编程问题处理链
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

    # 创建通用问题处理链（默认路由）
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

    # 创建完整的问题回答链
    answer_chain = (
        {
            # 使用 route_chain 判断问题类型
            "route": route_chain,
            # 使用 RunnableLambda 提取原始问题
            "question": RunnableLambda(lambda data: data["question"]),
        }
        | RunnableBranch(  # 使用 RunnableBranch 实现路由逻辑
            # 第一个条件：如果 route == "math"，使用 math_chain
            (lambda data: data["route"] == "math", math_chain),
            # 第二个条件：如果 route == "code"，使用 code_chain
            (lambda data: data["route"] == "code", code_chain),
            # 默认分支：使用 general_chain
            general_chain,
        )
    )

    # 定义测试问题列表
    questions = [
        "如何计算圆的面积？",           # 数学问题
        "Python 里 dataclass 有什么作用？",  # 编程问题
        "为什么 RAG 对企业问答系统有帮助？",  # 通用问题
    ]

    # 遍历并处理每个问题
    for question in questions:
        # 先获取路由结果
        route = route_chain.invoke({"question": question})
        # 再获取回答
        answer = answer_chain.invoke({"question": question})
        print(f"\n问题：{question}")
        print(f"路由结果：{route}")
        print(f"回答：{answer}")

    print(
        "\n要点：很多所谓"复杂 RouterChain"场景，今天用 RunnableBranch 就已经足够清爽。"
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
    """主函数，程序入口点"""
    # 打印主标题
    print_title("LangChain 编排模块（现代版）")

    try:
        # 调用所有示例函数
        example_sequential_composition()
        example_parallel_composition()
        example_routing()
    except RuntimeError as exc:
        # 捕获 RuntimeError（通常是 API 密钥缺失），并打印友好提示
        print(exc)
        print("已跳过需要在线模型的示例。")


# Python 惯用法：检查是否直接运行此脚本
if __name__ == "__main__":
    main()

"""
LangChain 代理与工具模块学习示例
===============================

本文件演示 LangChain 的 Agent 和 Tool 使用方法：
1. 自定义 Tool - 创建和使用自定义工具
2. ReAct Agent - 使用 ReAct 模式的代理
3. 工具调用机制和步骤追踪

核心知识点：
- Agent（代理）：使用 LLM 决定采取什么行动的组件
- Tool（工具）：Agent 可以调用的功能，可以是 Python 函数
- ReAct 模式：Reasoning + Acting，通过思考-行动循环完成任务
- 步骤追踪：Agent 可以追踪完成复杂任务的多步操作

运行前请确保：
1. 已安装依赖：pip install -r requirements.txt
2. 已配置 .env 文件（复制 .env.example 并填入 OPENAI_API_KEY）
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parser import BaseOutputParser
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    Tool,
    initialize_agent,
    AgentType
)


# ============================================================================
# 1. 自定义 Tool 示例
# ============================================================================

def example_custom_tools():
    """
    演示如何创建和使用自定义工具
    
    Tool 是 Agent 可以调用的功能，可以是任何 Python 函数。
    每个 Tool 必须定义：
    - name: 工具名称，Agent 会使用这个名字来引用
    - func: 工具函数
    - description: 工具的描述，LLM 会根据描述决定何时使用该工具
    """
    print("=" * 60)
    print("示例 1：自定义工具基本用法")
    print("=" * 60)
    
    # 自定义工具函数 1：简单计算器
    def calculator(expression):
        """
        执行简单的数学计算
        
        Args:
            expression: 数学表达式，如 "2 + 3"
            
        Returns:
            计算结果
        """
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    # 自定义工具函数 2：获取当前日期
    def get_current_date(query):
        """
        获取当前日期时间查询工具
        """
        from datetime import datetime
        now = datetime.now()
        return f"当前日期时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # 自定义工具函数 3：字符串处理
    def string_processor(input_text):
        """
        字符串处理工具：统计字符数和词数
        """
        char_count = len(input_text)
        word_count = len(input_text.split())
        return f"输入字符串包含 {char_count} 个字符和 {word_count} 个词"
    
    # 包装为 Tool 对象
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="用于执行基本的数学计算，接受数学表达式字符串，如 '2 + 3' 或 '10 * 5'"
        ),
        Tool(
            name="CurrentDate",
            func=get_current_date,
            description="获取当前日期时间，需要输入任意字符串"
        ),
        Tool(
            name="StringProcessor",
            func=string_processor,
            description="处理字符串，统计字符数和词数，输入任意需要处理的字符串"
        )
    ]
    
    # 测试工具，不使用 Agent
    print("\n直接测试工具:")
    print("-" * 40)
    
    print("\n1. 计算器工具:")
    result1 = tools[0].func("2 * 3 + 4")
    print(f"  输入: '2 * 3 + 4'")
    print(f"  输出: {result1}")
    
    print("\n2. 日期工具:")
    result2 = tools[1].func("test")
    print(f"  输出: {result2}")
    
    print("\n3. 字符串处理工具:")
    result3 = tools[2].func("Hello World")
    print(f"  输入: 'Hello World'")
    print(f"  输出: {result3}")
    print()
    
    return tools


def example_complex_custom_tools():
    """
    演示更复杂的自定义工具
    
    这些工具模拟了真实的业务场景，如天气查询、数据库查询等。
    """
    print("=" * 60)
    print("示例 2：复杂的自定义工具")
    print("=" * 60)
    
    # 工具 1：天气查询工具
    def weather_query(city):
        """
        模拟天气查询工具
        """
        weather_data = {
            "北京": "晴朗，25°C",
            "上海": "多云，28°C",
            "广州": "阴天，30°C",
            "深圳": "晴朗，32°C"
        }
        
        weather = weather_data.get(city, "未找到该城市")
        return f"{city} 的天气: {weather}"
    
    # 工具 2：数据库查询工具
    def database_query(query_type):
        """
        模拟数据库查询工具
        """
        data = {
            "users": "用户数量: 1000",
            "products": "产品数量: 500",
            "orders": "订单数量: 2000"
        }
        return data.get(query_type, "未知查询类型")
    
    # 工具 3：文本搜索
    def text_search(keyword, documents):
        """
        在文档集合中搜索关键词
        """
        docs = [
            "Python 是一种高级编程语言",
            "Java 是一种面向对象的编程语言",
            "JavaScript 主要用于网页开发"
        ]
        
        results = [doc for doc in docs if keyword.lower() in doc.lower()]
        return f"搜索关键词 '{keyword}' 找到 {len(results)} 个结果: {results}"
    
    tools = [
        Tool(
            name="WeatherQuery",
            func=weather_query,
            description="查询指定城市的天气，输入城市名称，如：北京、上海、广州、深圳"
        ),
        Tool(
            name="DatabaseQuery",
            func=database_query,
            description="查询数据库统计信息，输入查询类型，如：users/products/orders"
        ),
        Tool(
            name="TextSearch",
            func=text_search,
            description="在文档集合中搜索关键词，输入需要搜索的关键词"
        )
    ]
    
    # 测试工具
    print("\n测试复杂工具:")
    print("-" * 40)
    
    print(f"\n天气查询: {tools[0].func('北京')}")
    print(f"数据库查询: {tools[1].func('users')}")
    print(f"文本搜索: {tools[2].func('Python')}")
    print()
    
    return tools


# ============================================================================
# 2. ReAct Agent 示例
# ============================================================================

def example_react_agent():
    """
    演示 ReAct（Reasoning + Acting）Agent 的用法
    
    ReAct Agent 的工作流程：
    1. 观察（Observation）：接收用户输入
    2. 思考（Thought）：思考应该使用哪个工具完成目标
    3. 行动（Action）：选择并执行相应的工具
    4. 观察（Observation）：获得工具执行结果
    5. 重复 2-4 步，直到完成任务
    
    这种模式使得 Agent 能够自主规划并完成多步骤的复杂任务。
    """
    print("=" * 60)
    print("示例 3：ReAct Agent 基本用法")
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
        
        # 自定义工具
        def calculator(expression):
            """计算器工具"""
            try:
                return str(eval(expression))
            except:
                return "计算错误"
        
        def get_current_date(query):
            """获取当前日期"""
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d")
        
        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="计算数学表达式，输入如 '2 + 3'"
            ),
            Tool(
                name="CurrentDate",
                func=get_current_date,
                description="获取当前日期，不需要参数"
            )
        ]
        
        # 创建 ReAct Agent
        # 使用 initialize_agent 方法创建 Agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True  # 打印详细过程
        )
        
        print("\nAgent 创建完成，开始测试:\n")
        
        # 测试 Agent
        print("测试任务 1: 今天是星期几？")
        print("-" * 40)
        response1 = agent.run("今天是星期几？")
        print(f"最终答案: {response1}\n")
        
        print("测试任务 2: 2 + 3 等于多少？")
        print("-" * 40)
        response2 = agent.run("2 + 3 等于多少？")
        print(f"最终答案: {response2}\n")
        
        print("测试任务 3: 计算 (10 + 5) * 2 的结果")
        print("-" * 40)
        response3 = agent.run("计算 (10 + 5) * 2 的结果")
        print(f"最终答案: {response3}\n")
        
    except Exception as e:
        print(f"创建 ReAct Agent 时发生错误: {e}")
        print("提示：请确保 API Key 有效，网络连接正常，或者使用本地模型")
        print("注意：由于 LangChain API 更新，以上代码可能需要调整")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("*" * 60)
    print("LangChain 代理与工具模块学习")
    print("*" * 60)
    print("\n")
    
    # 运行自定义工具示例
    example_custom_tools()
    example_complex_custom_tools()
    
    # 运行 ReAct Agent 示例
    # 注意：以下示例需要有效的 OpenAI API Key
    print("\n" + "=" * 60)
    print("注意：以下示例需要有效的 OpenAI API Key")
    print("如果未配置，跳过相关测试\n")
    print("=" * 60)
    
    example_react_agent()
    
    print("\n")
    print("*" * 60)
    print("所有示例运行完成！")
    print("*" * 60)


if __name__ == "__main__":
    main()
"""
LangChain 基础模块学习示例
==========================

本文件演示 LangChain 的基础组件使用方法：
1. Prompt Templates - 提示词模板的构建与参数化
2. LLM 和 ChatModel - 大语言模型的直接调用
3. Output Parsers - 输出解析器的使用

核心知识点：
- PromptTemplate：用于创建可重用的提示词模板，支持变量插值
- ChatPromptTemplate：专门为对话模型设计的模板，区分不同角色的消息
- SystemMessage vs HumanMessage：
  * SystemMessage：设置 AI 的角色和行为准则
  * HumanMessage：代表用户输入
  * AIMessage：代表 AI 的响应

运行前请确保：
1. 已安装依赖：pip install -r requirements.txt
2. 已配置 .env 文件（复制 .env.example 并填入 OPENAI_API_KEY）
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


# ============================================================================
# 1. Prompt Templates 示例
# ============================================================================

def example_prompt_template():
    """
    演示 PromptTemplate 的基本用法
    
    PromptTemplate 允许我们创建可重用的提示词模板，
    其中可以包含占位符，运行时通过参数替换。
    """
    print("=" * 60)
    print("示例 1：PromptTemplate 基础用法")
    print("=" * 60)
    
    # 创建一个简单的提示词模板
    # {topic} 是占位符，后续会被替换
    template = """
    请给我介绍关于 {topic} 的知识。
    要求：用通俗易懂的语言，不超过200字。
    """
    
    # 使用 from_template 方法创建模板对象
    prompt = PromptTemplate.from_template(template)
    
    # 使用 format 方法填充占位符
    formatted_prompt = prompt.format(topic="人工智能")
    print(f"格式化后的提示词:\n{formatted_prompt}")
    print()
    
    # 另一种创建方式：直接实例化 PromptTemplate
    # input_variables 参数指定占位符名称
    # template 参数指定模板内容
    prompt2 = PromptTemplate(
        input_variables=["name", "age"],
        template="我的名字是 {name}，今年 {age} 岁。"
    )
    
    formatted_prompt2 = prompt2.format(name="小明", age=25)
    print(f"格式化后的提示词 2:\n{formatted_prompt2}")
    print()


def example_chat_prompt_template():
    """
    演示 ChatPromptTemplate 的用法
    
    ChatPromptTemplate 专门为对话模型设计，
    可以区分 SystemMessage（系统指令）、HumanMessage（用户输入）等角色。
    """
    print("=" * 60)
    print("示例 2：ChatPromptTemplate 用法")
    print("=" * 60)
    
    # 创建对话提示词模板
    # from_messages 方法允许我们按角色组织消息
    chat_template = ChatPromptTemplate.from_messages([
        # SystemMessage：设置 AI 的角色和行为
        ("system", "你是一个专业的 {role}，用简洁的语言回答问题。"),
        # HumanMessage：代表用户的输入
        ("human", "请解释 {topic} 的概念。")
    ])
    
    # 格式化模板
    formatted_chat = chat_template.format(role="数学老师", topic="微积分")
    print(f"格式化后的对话提示词:")
    for message in formatted_chat:
        print(f"- {message.type}: {message.content}")
    print()
    
    # 另一种方式：使用 MessagesPlaceholder 处理多轮对话
    from langchain.prompts import MessagesPlaceholder
    
    chat_template2 = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个友好的助手。"),
        MessagesPlaceholder(variable_name="history"),  # 历史对话占位符
        HumanMessage(content="{input}")
    ])
    
    print("包含历史对话的模板创建成功")
    print()


# ============================================================================
# 2. LLM 和 ChatModel 示例
# ============================================================================

def example_llm():
    """
    演示 LLM（大语言模型）的基本调用
    
    OpenAI 类封装了对 OpenAI GPT 模型的调用。
    """
    print("=" * 60)
    print("示例 3：LLM 基础调用")
    print("=" * 60)
    
    # 获取 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        print("请确保已创建 .env 文件并配置 API Key")
        return
    
    try:
        # 创建 LLM 实例
        # temperature 参数控制输出的随机性（0-1），值越小输出越确定
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo-instruct"  # 使用较便宜的模型进行演示
        )
        
        # 直接调用模型进行文本生成
        prompt = "用一句话解释什么是机器学习。"
        response = llm.invoke(prompt)
        
        print(f"提示词: {prompt}")
        print(f"模型响应: {response}")
        print()
        
    except Exception as e:
        print(f"调用 LLM 时发生错误: {e}")
        print("提示：请检查 API Key 是否正确，网络连接是否正常")


def example_chat_model():
    """
    演示 ChatModel（对话模型）的用法
    
    ChatOpenAI 专门用于对话场景，接受消息列表作为输入。
    """
    print("=" * 60)
    print("示例 4：ChatModel 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建聊天模型实例
        chat = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
        # 使用消息列表调用模型
        # 消息列表包含不同角色的消息，模拟真实对话场景
        messages = [
            SystemMessage(content="你是一个 Python 编程专家。"),
            HumanMessage(content="如何创建一个简单的 Python 函数？")
        ]
        
        response = chat.invoke(messages)
        
        print(f"用户提问: {messages[1].content}")
        print(f"AI 回答: {response.content}")
        print()
        
        # 演示多轮对话
        messages2 = [
            SystemMessage(content="你是一个有礼貌的助手。"),
            HumanMessage(content="你好！"),
            AIMessage(content="你好！有什么我可以帮助你的吗？"),
            HumanMessage(content="请告诉我今天的日期。")
        ]
        
        response2 = chat.invoke(messages2)
        print(f"多轮对话中的最后回答: {response2.content}")
        print()
        
    except Exception as e:
        print(f"调用 ChatModel 时发生错误: {e}")


# ============================================================================
# 3. Output Parsers 示例
# ============================================================================

def example_output_parser():
    """
    演示 OutputParser（输出解析器）的用法
    
    OutputParser 用于将模型输出的文本解析为结构化数据（如字典、JSON）。
    """
    print("=" * 60)
    print("示例 5：OutputParser 用法")
    print("=" * 60)
    
    # 定义输出模式（Schema）
    # ResponseSchema 描述我们期望的数据结构
    response_schemas = [
        ResponseSchema(
            name="answer", 
            description="用户问题的答案，不超过50字",
            type="string"
        ),
        ResponseSchema(
            name="confidence",
            description="答案的置信度（高/中/低）",
            type="string"
        )
    ]
    
    # 创建输出解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # 获取格式化指令
    # 这段指令会被插入到提示词中，告诉模型如何格式化输出
    format_instructions = output_parser.get_format_instructions()
    print("格式化指令:")
    print(format_instructions)
    print()
    
    # 创建提示词模板，包含格式化指令
    template = """
    请回答以下问题，并按照指定的格式输出：
    
    问题：{question}
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # 格式化提示词
    formatted_prompt = prompt.format(question="地球绕太阳转一圈需要多久？")
    print(f"格式化后的提示词:")
    print(formatted_prompt)
    print()
    
    # （实际使用时，将这个提示词传给 LLM，然后用 parser 解析输出）
    # response = llm.invoke(formatted_prompt)
    # parsed_output = output_parser.parse(response)
    # print(f"解析后的输出: {parsed_output}")
    
    print("提示：在实际使用时，将格式化后的提示词传给 LLM，")
    print("然后使用 output_parser.parse() 方法解析输出")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("*" * 60)
    print("LangChain 基础模块学习")
    print("*" * 60)
    print("\n")
    
    # 运行 Prompt Templates 示例
    example_prompt_template()
    example_chat_prompt_template()
    
    # 运行 LLM 和 ChatModel 示例
    # 注意：以下示例需要有效的 OpenAI API Key
    example_llm()
    example_chat_model()
    
    # 运行 Output Parser 示例
    example_output_parser()
    
    print("\n")
    print("*" * 60)
    print("所有示例运行完成！")
    print("*" * 60)


if __name__ == "__main__":
    main()
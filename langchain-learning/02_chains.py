"""
LangChain 链模块学习示例
==========================

本文件演示 LangChain 的链（Chain）使用方法：
1. LLMChain - 基础链，将提示词和模型组合
2. SimpleSequentialChain - 顺序链，将多个链按顺序执行
3. SequentialChain - 更复杂的顺序链，支持多输入输出
4. RouterChain - 路由链，根据输入动态选择执行路径

核心知识点：
- Chain（链）：将多个组件（Prompt、LLM、Parser 等）组合起来形成工作流
- 输入/输出传递机制：前一个链的输出作为后一个链的输入
- 路由机制：根据输入内容的特征选择不同的处理逻辑

运行前请确保：
1. 已安装依赖：pip install -r requirements.txt
2. 已配置 .env 文件（复制 .env.example 并填入 OPENAI_API_KEY）
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_classic.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_classic.chains import MultiPromptChain
from langchain_classic.chains import LLMRouterChain, RouterOutputParser
from langchain_classic.chains import MULTI_PROMPT_ROUTER_TEMPLATE


# ============================================================================
# 1. LLMChain 示例
# ============================================================================

def example_llm_chain():
    """
    演示 LLMChain 的基本用法
    
    LLMChain 是最基础的链，将 PromptTemplate 和 LLM 组合在一起。
    """
    print("=" * 60)
    print("示例 1：LLMChain 基础用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建 LLM 实例
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo-instruct"
        )
        
        # 创建提示词模板
        template = """
        请回答以下问题：
        问题：{question}
        答案：
        """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        
        # 创建 LLMChain
        # LLMChain 将 Prompt 和 LLM 组合，可以自动处理提示词格式化
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # 调用链
        question = "什么是 Python？"
        response = chain.invoke({"question": question})
        
        print(f"问题: {question}")
        print(f"答案: {response['text']}")
        print()
        
        # 演示链的输入输出流程
        print("链的输入输出流程：")
        print(f"输入: {{'question': '{question}'}}")
        print(f"输出: {response}")
        print()
        
    except Exception as e:
        print(f"调用 LLMChain 时发生错误: {e}")


def example_chat_llm_chain():
    """
    演示使用 ChatPromptTemplate 创建 LLMChain
    
    ChatPromptTemplate 更适合对话场景，可以区分不同角色的消息。
    """
    print("=" * 60)
    print("示例 2：使用 ChatPromptTemplate 的 LLMChain")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建聊天模型
        chat = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
        # 创建对话提示词模板
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个专业的翻译助手。"),
            HumanMessage(content="请将以下文本翻译成英文：{text}")
        ])
        
        # 创建 LLMChain
        chain = LLMChain(llm=chat, prompt=chat_template)
        
        # 调用链
        text = "你好，世界！"
        response = chain.invoke({"text": text})
        
        print(f"原文: {text}")
        print(f"翻译: {response['text']}")
        print()
        
    except Exception as e:
        print(f"调用 Chat LLMChain 时发生错误: {e}")


# ============================================================================
# 2. SimpleSequentialChain 示例
# ============================================================================

def example_simple_sequential_chain():
    """
    演示 SimpleSequentialChain 的用法
    
    SimpleSequentialChain 将多个链按顺序连接，
    前一个链的输出作为后一个链的输入。
    
    适用场景：
    - 需要多步处理的任务（如：起草 -> 润色 -> 总结）
    - 每个步骤只有一个输入和一个输出
    """
    print("=" * 60)
    print("示例 3：SimpleSequentialChain 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建 LLM 实例
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo-instruct"
        )
        
        # 第一个链：生成故事大纲
        outline_template = """
        请根据以下主题生成一个简短的故事大纲（不超过100字）：
        主题：{theme}
        """
        outline_prompt = PromptTemplate(
            template=outline_template,
            input_variables=["theme"]
        )
        outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
        
        # 第二个链：将大纲扩展成完整故事
        story_template = """
        请根据以下大纲，扩展成一个完整的故事（200-300字）：
        {outline}
        """
        story_prompt = PromptTemplate(
            template=story_template,
            input_variables=["outline"]
        )
        story_chain = LLMChain(llm=llm, prompt=story_prompt)
        
        # 创建顺序链
        # chains 参数指定要按顺序执行的链
        simple_chain = SimpleSequentialChain(
            chains=[outline_chain, story_chain],
            verbose=True  # 打印详细执行过程
        )
        
        # 调用顺序链
        theme = "勇敢的小猫"
        print(f"主题: {theme}")
        print("\n执行过程：")
        print("-" * 40)
        
        response = simple_chain.invoke(theme)
        
        print("-" * 40)
        print(f"\n最终结果:\n{response['output']}")
        print()
        
    except Exception as e:
        print(f"调用 SimpleSequentialChain 时发生错误: {e}")


# ============================================================================
# 3. SequentialChain 示例
# ============================================================================

def example_sequential_chain():
    """
    演示 SequentialChain 的用法
    
    SequentialChain 是 SimpleSequentialChain 的增强版，支持：
    - 多个输入变量
    - 多个输出变量
    - 更灵活的数据流控制
    
    关键参数：
    - chains: 要执行的链列表
    - input_variables: 最终的输入变量名
    - output_variables: 最终的输出变量名
    """
    print("=" * 60)
    print("示例 4：SequentialChain 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo-instruct"
        )
        
        # 链 1：生成标题
        title_template = """
        请为以下内容生成一个简短的标题（不超过10个字）：
        内容：{content}
        """
        title_prompt = PromptTemplate(
            template=title_template,
            input_variables=["content"]
        )
        title_chain = LLMChain(
            llm=llm,
            prompt=title_prompt,
            output_key="title"  # 指定输出键名
        )
        
        # 链 2：生成摘要
        summary_template = """
        请为以下内容生成一个摘要（不超过50字）：
        内容：{content}
        """
        summary_prompt = PromptTemplate(
            template=summary_template,
            input_variables=["content"]
        )
        summary_chain = LLMChain(
            llm=llm,
            prompt=summary_prompt,
            output_key="summary"
        )
        
        # 链 3：生成评价（使用标题和摘要）
        review_template = """
        请根据以下标题和摘要，给出一个简短的评价：
        标题：{title}
        摘要：{summary}
        评价：
        """
        review_prompt = PromptTemplate(
            template=review_template,
            input_variables=["title", "summary"]
        )
        review_chain = LLMChain(
            llm=llm,
            prompt=review_prompt,
            output_key="review"
        )
        
        # 创建 SequentialChain
        sequential_chain = SequentialChain(
            chains=[title_chain, summary_chain, review_chain],
            input_variables=["content"],  # 初始输入
            output_variables=["title", "summary", "review"],  # 最终输出
            verbose=True
        )
        
        # 调用链
        content = """
        今天天气很好，阳光明媚。我和朋友们一起去公园散步，
        看到了很多美丽的花朵和可爱的小鸟。我们还一起吃了野餐，
        度过了一个愉快的下午。
        """
        
        print(f"输入内容:\n{content}")
        print("\n执行过程：")
        print("-" * 40)
        
        response = sequential_chain.invoke({"content": content})
        
        print("-" * 40)
        print(f"\n标题: {response['title']}")
        print(f"摘要: {response['summary']}")
        print(f"评价: {response['review']}")
        print()
        
    except Exception as e:
        print(f"调用 SequentialChain 时发生错误: {e}")


# ============================================================================
# 4. RouterChain 示例
# ============================================================================

def example_router_chain():
    """
    演示 RouterChain 的用法
    
    RouterChain 根据输入内容，将请求路由到不同的处理链。
    
    适用场景：
    - 根据用户问题类型选择不同的处理逻辑
    - 多领域问答系统（如：数学、编程、历史等）
    - 需要根据输入动态选择处理流程的场景
    
    工作流程：
    1. 用户输入 -> RouterChain（路由链）
    2. RouterChain 分析输入，选择目标链
    3. 被选中的链处理输入并返回结果
    """
    print("=" * 60)
    print("示例 5：RouterChain 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo-instruct"
        )
        
        # 定义不同领域的提示词模板
        
        # 物理领域模板
        physics_template = """
        你是一位物理学专家。
        请回答以下物理问题：
        {input}
        """
        
        # 数学领域模板
        math_template = """
        你是一位数学专家。
        请回答以下数学问题：
        {input}
        """
        
        # 计算机科学领域模板
        cs_template = """
        你是一位计算机科学专家。
        请回答以下编程问题：
        {input}
        """
        
        # 创建提示词模板
        prompt_infos = [
            {
                "name": "physics",
                "description": "适合回答物理相关的问题",
                "prompt_template": physics_template
            },
            {
                "name": "math",
                "description": "适合回答数学相关的问题",
                "prompt_template": math_template
            },
            {
                "name": "cs",
                "description": "适合回答编程相关的问题",
                "prompt_template": cs_template
            }
        ]
        
        # 为每个领域创建目标链
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["input"]
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            destination_chains[name] = chain
        
        # 创建默认链（当无法确定领域时使用）
        default_prompt = PromptTemplate.from_template(
            "请回答以下问题：\n{input}"
        )
        default_chain = LLMChain(llm=llm, prompt=default_prompt)
        
        # 创建路由模板
        # 路由模板帮助 LLM 理解如何选择正确的目标链
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        
        # 创建路由提示词
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser()
        )
        
        # 创建路由链
        router_chain = LLMRouterChain(
            llm=llm,
            prompt=router_prompt
        )
        
        # 创建多提示词链
        # 这个链会自动将输入路由到合适的子链
        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True
        )
        
        # 测试不同领域的问题
        questions = [
            "什么是牛顿第二定律？",  # 物理
            "如何计算圆的面积？",    # 数学
            "什么是 Python 的装饰器？"  # 计算机科学
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            print("-" * 40)
            response = chain.invoke(question)
            print(f"回答: {response['output']}")
            print()
        
    except Exception as e:
        print(f"调用 RouterChain 时发生错误: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("*" * 60)
    print("LangChain 链模块学习")
    print("*" * 60)
    print("\n")
    
    # 运行 LLMChain 示例
    example_llm_chain()
    example_chat_llm_chain()
    
    # 运行 SequentialChain 示例
    example_simple_sequential_chain()
    example_sequential_chain()
    
    # 运行 RouterChain 示例
    example_router_chain()
    
    print("\n")
    print("*" * 60)
    print("所有示例运行完成！")
    print("*" * 60)


if __name__ == "__main__":
    main()
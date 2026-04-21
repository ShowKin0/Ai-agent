"""
LangChain 记忆模块学习示例
==========================

本文件演示 LangChain 的记忆（Memory）组件使用方法：
1. ConversationBufferMemory - 保存完整的对话历史
2. ConversationBufferWindowMemory - 只保留最近几轮对话
3. ConversationSummaryMemory - 对话历史的摘要记忆

核心知识点：
- Memory（记忆）：用于保存和传递对话状态
- 状态传递机制：Memory 自动管理对话历史，将其插入到 Prompt 中
- 不同 Memory 类型的适用场景：
  * BufferMemory：适合需要完整历史但对话较短的场景
  * WindowMemory：适合长对话，只关注最近内容的场景
  * SummaryMemory：适合长对话，需要保存关键信息的场景

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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationBufferMemory
)


# ============================================================================
# 1. ConversationBufferMemory 示例
# ============================================================================

def example_buffer_memory():
    """
    演示 ConversationBufferMemory 的用法
    
    ConversationBufferMemory 保存完整的对话历史，
    每次调用时将所有历史消息插入到 Prompt 中。
    
    适用场景：
    - 对话轮数较少（通常 < 10 轮）
    - 需要保留完整上下文信息
    
    注意事项：
    - 对话很长时可能超出模型的 Token 限制
    - 每次调用都会传递完整历史，增加 API 调用成本
    """
    print("=" * 60)
    print("示例 1：ConversationBufferMemory 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建 LLM 实例
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
        # 创建 BufferMemory
        # return_messages=True 表示以消息对象形式返回历史
        memory = ConversationBufferMemory(
            return_messages=True
        )
        
        # 创建对话链
        # Memory 会自动处理对话历史的保存和加载
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True  # 打印详细执行过程
        )
        
        # 模拟多轮对话
        print("\n=== 对话开始 ===\n")
        
        # 第一轮对话
        print("用户: 我的名字叫小明")
        response1 = conversation.invoke("我的名字叫小明")
        print(f"AI: {response1['response']}")
        print()
        
        # 第二轮对话 - AI 记住了用户的名字
        print("用户: 你叫什么名字？")
        response2 = conversation.invoke("你叫什么名字？")
        print(f"AI: {response2['response']}")
        print()
        
        # 第三轮对话 - AI 知道用户的名字
        print("用户: 我叫什么名字？")
        response3 = conversation.invoke("我叫什么名字？")
        print(f"AI: {response3['response']}")
        print()
        
        # 查看保存的对话历史
        print("\n=== 对话历史 ===")
        history = memory.load_memory_variables({})
        for i, message in enumerate(history['history'], 1):
            if isinstance(message, HumanMessage):
                print(f"{i}. 用户: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"{i}. AI: {message.content}")
        print()
        
    except Exception as e:
        print(f"调用 ConversationBufferMemory 时发生错误: {e}")


def example_buffer_memory_manual():
    """
    演示手动使用 ConversationBufferMemory
    
    有时我们需要更灵活地控制记忆的保存和加载。
    """
    print("=" * 60)
    print("示例 2：手动使用 ConversationBufferMemory")
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
        
        # 创建 BufferMemory（不使用 return_messages）
        memory = ConversationBufferMemory()
        
        # 手动保存对话历史
        memory.save_context(
            {"input": "今天天气怎么样？"},
            {"output": "今天天气晴朗，温度适宜。"}
        )
        
        memory.save_context(
            {"input": "明天呢？"},
            {"output": "明天可能会下雨，建议带伞。"}
        )
        
        # 加载历史记录
        history = memory.load_memory_variables({})
        print("保存的对话历史:")
        print(history['history'])
        print()
        
        # 将历史记录作为上下文使用
        template = """
        以下是对话历史：
        {history}
        
        当前问题：{question}
        
        请根据对话历史回答问题：
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["history", "question"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        response = chain.invoke({
            "history": history['history'],
            "question": "今天的天气如何？"
        })
        
        print(f"回答: {response['text']}")
        print()
        
    except Exception as e:
        print(f"调用 BufferMemory 手动模式时发生错误: {e}")


# ============================================================================
# 2. ConversationBufferWindowMemory 示例
# ============================================================================

def example_window_memory():
    """
    演示 ConversationBufferWindowMemory 的用法
    
    ConversationBufferWindowMemory 只保留最近的 k 轮对话，
    超过限制的旧对话会被丢弃。
    
    适用场景：
    - 长对话场景，避免 Token 超限
    - 只需要关注最近内容的场景
    - 减少历史对话对当前回答的影响
    
    参数：
    - k: 保留的对话轮数（包括用户和 AI 的各一轮算作一轮）
    """
    print("=" * 60)
    print("示例 3：ConversationBufferWindowMemory 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
        # 创建 WindowMemory，只保留最近 2 轮对话
        memory = ConversationBufferWindowMemory(
            k=2,  # 只保留最近 2 轮对话
            return_messages=True
        )
        
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False  # 关闭详细输出
        )
        
        print("\n=== 对话开始（WindowMemory，k=2）===\n")
        
        # 进行多轮对话
        conversations = [
            "我喜欢苹果",
            "我还喜欢香蕉",
            "你记得我喜欢什么水果吗？",  # 应该记得香蕉，苹果超出窗口
            "我还喜欢橙子",
            "现在我喜欢哪些水果？"  # 应该记得橙子和香蕉
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"第 {i} 轮")
            print(f"用户: {user_input}")
            response = conversation.invoke(user_input)
            print(f"AI: {response['response']}")
            
            # 显示当前窗口内的对话
            history = memory.load_memory_variables({})
            print(f"（当前窗口包含 {len(history['history'])} 条消息）")
            print()
        
    except Exception as e:
        print(f"调用 ConversationBufferWindowMemory 时发生错误: {e}")


# ============================================================================
# 3. ConversationSummaryMemory 示例
# ============================================================================

def example_summary_memory():
    """
    演示 ConversationSummaryMemory 的用法
    
    ConversationSummaryMemory 使用 LLM 自动生成对话历史的摘要，
    而不是保存完整的对话内容。
    
    适用场景：
    - 长对话场景，避免保存大量文本
    - 需要保留关键信息但不关心具体对话内容
    - 对话 Token 成本敏感的场景
    
    工作原理：
    1. 保存新的对话时，将新对话与之前的摘要合并
    2. 使用 LLM 生成新的摘要
    3. 摘要比完整对话更简洁，节省 Token
    """
    print("=" * 60)
    print("示例 4：ConversationSummaryMemory 用法")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
        # 创建 SummaryMemory
        memory = ConversationSummaryMemory(
            llm=llm,  # 需要传入 LLM 用于生成摘要
            return_messages=False
        )
        
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        print("\n=== 对话开始（SummaryMemory）===\n")
        
        # 进行多轮对话
        conversations = [
            "我要制定一个学习计划，学习 Python 编程",
            "我每天有 2 小时学习时间",
            "我的目标是能在 3 个月内找到 Python 开发工作",
            "我应该从哪里开始学起？",
            "有哪些学习资源推荐？"
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"第 {i} 轮")
            print(f"用户: {user_input}")
            response = conversation.invoke(user_input)
            print(f"AI: {response['response']}")
            
            # 显示当前摘要
            summary = memory.load_memory_variables({})
            print(f"\n当前摘要:")
            print(summary['history'])
            print("-" * 60)
            print()
        
    except Exception as e:
        print(f"调用 ConversationSummaryMemory 时发生错误: {e}")


# ============================================================================
# 4. Memory 在自定义链中的应用
# ============================================================================

def example_custom_chain_with_memory():
    """
    演示如何在自定义的 LLMChain 中使用 Memory
    
    除了 ConversationChain，我们也可以在自定义的链中使用 Memory。
    """
    print("=" * 60)
    print("示例 5：自定义链使用 Memory")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到 OPENAI_API_KEY 环境变量")
        return
    
    try:
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        
        # 创建 BufferMemory
        memory = ConversationBufferMemory(
            memory_key="chat_history",  # 指定历史记录在 prompt 中的变量名
            return_messages=True
        )
        
        # 创建提示词模板
        # 注意：需要包含 {chat_history} 占位符来插入历史记录
        template = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是一个友好的助手，记得之前的对话内容。"),
            MessagesPlaceholder(variable_name="chat_history"),  # 历史消息占位符
            HumanMessage(content="{input}")
        ])
        
        # 创建自定义链
        chain = LLMChain(
            llm=llm,
            prompt=template,
            memory=memory,
            verbose=False
        )
        
        print("\n=== 自定义链对话 ===\n")
        
        # 进行对话
        inputs = [
            "我正在学习 LangChain",
            "LangChain 是什么？",
            "它有哪些主要组件？"
        ]
        
        for user_input in inputs:
            print(f"用户: {user_input}")
            response = chain.invoke({"input": user_input})
            print(f"AI: {response['text']}")
            print()
        
        # 查看历史记录
        print("=== 对话历史 ===")
        history = memory.load_memory_variables({})
        for msg in history['chat_history']:
            if isinstance(msg, HumanMessage):
                print(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"AI: {msg.content}")
        print()
        
    except Exception as e:
        print(f"调用自定义链时发生错误: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("*" * 60)
    print("LangChain 记忆模块学习")
    print("*" * 60)
    print("\n")
    
    # 运行 BufferMemory 示例
    example_buffer_memory()
    example_buffer_memory_manual()
    
    # 运行 WindowMemory 示例
    example_window_memory()
    
    # 运行 SummaryMemory 示例
    example_summary_memory()
    
    # 运行自定义链示例
    example_custom_chain_with_memory()
    
    print("\n")
    print("*" * 60)
    print("所有示例运行完成！")
    print("*" * 60)


if __name__ == "__main__":
    # 导入 MessagesPlaceholder（放在这里避免循环导入）
    from langchain.prompts import MessagesPlaceholder
    main()
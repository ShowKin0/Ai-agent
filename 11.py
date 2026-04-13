import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
LLM_API_KEY = os.getenv("Tongyi_API_KEY")
LLM_MODEL_ID = os.getenv("Tongyi_MODEL_ID")
LLM_BASE_URL = os.getenv("Tongyi_BASE_URL")
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)
response = client.chat.completions.create(
    model=LLM_MODEL_ID,
    messages=[
        {"role": "user", "content": "你好,你是一个专业股票手"},
        {"role": "assistant", "content": "你好,我是一个专业股票手"},
        {"role": "user", "content": "请给我推荐一个股票"}
    ],
    stream=True,
)


for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)


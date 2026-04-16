from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
model_key=os.getenv("Tongyi_API_KEY")
model_id=os.getenv("Tongyi_MODEL_ID")
Model=Tongyi(api_key=model_key, model_id=model_id)
promptz_template=PromptTemplate.from_template(
    "你好,你是一个专业股票手,帮我分析一下这只股票{stock}"
)
promptz=promptz_template.format(stock="513180")
response=Model.stream(promptz)
for chunk in response:
    print(chunk, end="", flush=True)

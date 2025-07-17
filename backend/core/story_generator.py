from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

def call_llm(input):
    my_prompt = PromptTemplate.from_template("请回答这个问题: {input}")
    llm = ChatOpenAI(
        model = "gpt-4o",
        api_key = API_KEY,
        base_url = BASE_URL
    )
    chain = my_prompt | llm | StrOutputParser()
    output = chain.invoke({"input": input})
    return output

if __name__ == "__main__":
    input = "霹雳布袋戏最新剧集出到哪里了？"
    response = call_llm(input)
    print(response)
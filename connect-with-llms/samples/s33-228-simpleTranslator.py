import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import logger

## Logging ##
# clear_terminal()

## Foundation Model ##
llm: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

logger.info("")

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


parser = StrOutputParser()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

chain = prompt_template | llm | parser

app = FastAPI(
    title="simpleTranslator",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

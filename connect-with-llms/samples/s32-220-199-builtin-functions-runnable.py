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
model: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

logger.info("")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "tell me a curious fact about {soccer_player}"
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser
# response = chain.invoke({"soccer_player": "Ronaldo"})
# logger.info(response)

## Use of .bind() to add arguments to a Runnable in a LCEL Chain
# * For example, we can add an argument to stop the model response when it reaches the word "Ronaldo":

chain = prompt | model.bind(stop=["Ronaldo"]) | output_parser

# response = chain.batch(
#     [
#         {"soccer_player": "Messi"},
#         {"soccer_player": "Ronaldo"},
#         {"soccer_player": "Mardona"},
#     ]
# )
# logger.info(response)

## Use of .bind() to call an OpenAI Function in a LCEL Chain

functions = [
    {
        "name": "soccerfacts",
        "description": "Curious facts about a soccer player",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for the curious facts about a soccer player",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to the question",
                },
            },
            "required": ["question", "answer"],
        },
    }
]

from langchain_core.output_parsers import JsonOutputParser

chain = prompt | model.bind(functions=functions) | JsonOutputParser()

# response = chain.invoke(input={"soccer_player": "Messi"})
# logger.info(response)

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

chain = RunnableParallel({"original_input": RunnablePassthrough()})
response = chain.invoke("whatever")
logger.info(response)


def make_uppercase(arg):
    return arg["original_input"].upper()


chain = RunnableParallel({"original_input": RunnablePassthrough()}).assign(
    uppercase=RunnableLambda(make_uppercase)
)
response = chain.invoke("whatever")
logger.info(response)

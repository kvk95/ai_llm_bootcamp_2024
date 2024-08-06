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
model: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

logger.info("")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#### Legacy Chain

from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_template(
    "tell me a curious fact about {soccer_player}"
)

output_parser = StrOutputParser()

traditional_chain = LLMChain(llm=model, prompt=prompt)

response = traditional_chain.predict(soccer_player="Maradona")
logger.info(response)

#### New LCEL Chain
# * The "pipe" operator `|` is the main element of the LCEL chains.
# * The order (left to right) of the elements in a LCEL chain matters.
# * An LCEL Chain is a Sequence of Runnables.

chain = prompt | model | output_parser

response = chain.invoke({"soccer_player": "Ronaldo"})
logger.info(response)

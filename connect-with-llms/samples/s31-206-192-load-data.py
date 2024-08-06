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
chatModel: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

logger.info("")

# Load Data
## Intro
# * Connect with data sources and load private documents.

## Table of contents
# * Data loading integrations.
# * Formatting imported documents.
# * Transforming documents into embeddings.
# * Vector stores (aka vector databases).
# * QA from loaded document: Retrieving.

## LangChain built-in data loaders.
# * Labeled as "integrations".
# * Most of them require to install the corresponding libraries.

## LangChain documentation on Document Loaders
# * See the documentation page [here](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/).
# * See the list of built-in document loaders [here](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/).

## Setup
#### Recommended: create new virtualenv
# * mkdir your_project_name
# * cd your_project_name
# * pyenv virtualenv 3.11.4 your_venv_name
# * pyenv activate your_venv_name
# * pip install jupyterlab
# * jupyter lab

# * NOTE: Since right now is the best LLM in the market, we will use OpenAI by default. You will see how to connect with other Open Source LLMs like Llama3 or Mistral in a next lesson.
## Simple data loading
#### Loading a .txt file
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")

loaded_data = loader.load()
# * If you uncomment and execute the next cell you will see the contents of the loaded document.
# logger.info(loaded_data)

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("./data/Street_Tree_List.csv")
loaded_data = loader.load()
# logger.info(loaded_data[0:50])

#### Loading an .html file

from langchain_community.document_loaders import BSHTMLLoader

# pip install bs4
# pip install html5lib
# pip install lxml

loader = BSHTMLLoader(
    "./data/_100 AI Startups__ 100 LLM Apps that have earned $500,000 before their first year of existence.html",
    open_encoding="utf8",
)
loaded_data = loader.load()
# logger.info(loaded_data[0:50])

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./data/5pages.pdf")
loaded_data = loader.load_and_split()
# logger.info(loaded_data[0:50])

from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader("query=name, load_max_docs=1")
# loaded_data = loader.load()[0].page_content
# logger.info(loaded_data[0:50])

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", "Answer this {question}, here is some extra {context}"),
    ]
)
messages = chat_template.format_messages(
    name="JFK", question="Where was JFK born?", context=loaded_data
)
response = chatModel.invoke(messages)
logger.info(response)

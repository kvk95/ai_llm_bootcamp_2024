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

from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()
response = chain.invoke("Abram")
logger.info(response)


def russian_lastname(name: str) -> str:
    return f"{name}ovich"


from langchain_core.runnables import RunnableLambda

chain = RunnablePassthrough() | RunnableLambda(russian_lastname)
response = chain.invoke("Abram")
logger.info(response)

from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "operation_b": RunnableLambda(russian_lastname),
    }
)
response = chain.invoke("Abram")
logger.info(response)

chain = RunnableParallel(
    {
        "operation_a": RunnablePassthrough(),
        "soccer_player": lambda x: x["name"] + "ovich",
    }
)
response = chain.invoke({"name1": "Jordam", "name": "Abram"})
logger.info(response)


chain = RunnableParallel(
    {
        "addition": lambda x: x["d1"] + x["d2"],
        "multiply": lambda x: x["d1"] * x.get("d3", 0),
    }
)
response = chain.invoke({"d1": 5, "d2": 10, "d3": 20})
logger.info(response)

# from utils.MyVectorStore import chroma_from_texts
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from utils.MyEmbeddingFunction import SentenceEmbeddingFunction


# vectorstore = chroma_from_texts(
#     ["AI Accelera has trained more than 3,000 Enterprise Alumni."],
#     embedding=SentenceEmbeddingFunction(),
#     persist_directory="s32-218-198-builtin-runnables",
#     collection_name="s32-218-198-builtin-runnables",
# )
# retriever = vectorstore.as_retriever()

# template = """Answer the question based only on the following context:
# {context}

# Question: {question}

# Answer in the following language: {language}
# """

# prompt = ChatPromptTemplate.from_template(template)

# chain = (
#     {
#         "context": itemgetter("question") | retriever,
#         "question": itemgetter("question"),
#         "language": itemgetter("language"),
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )

# response = chain.invoke(
#     {
#         "question": "How many Enterprise Alumni has trained AI Accelera?",
#         "language": "Pirate English",
#     }
# )

# logger.info(response)

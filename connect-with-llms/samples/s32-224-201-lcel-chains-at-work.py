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

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyVectorStore import chroma_from_documents

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)
chroma_name = "s32-224-201-lcel-chains-at-work"
vectorstore = chroma_from_documents(
    documents=splits,
    embedding=SentenceEmbeddingFunction(),
    collection_name=chroma_name,
    persist_directory=chroma_name,
)

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
# logger.info(prompt)

# response = rag_chain.invoke("What is Task Decomposition?")
# logger.info(response)

#### Let's take a detailed look at the LCEL chain:
# * As you can see, the first part of the chain is a RunnableParallel (remember that RunnableParallel
# can have more than one syntax):
rag_chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")
logger.info(response)
"""
#### Note: what does the previos formatter function do?
The `format_docs` function takes a list of objects named `docs`. Each object in this list is expected to have an attribute named `page_content`, which stores textual content for each document.

The purpose of the function is to extract the `page_content` from each document in the `docs` list and then combine these contents into a single string. The contents of different documents are separated by two newline characters (`\n\n`), which means there will be an empty line between the content of each document in the final string. This formatting choice makes the combined content easier to read by clearly separating the content of different documents.

Here's a breakdown of how the function works:
1. The `for doc in docs` part iterates over each object in the `docs` list.
2. For each iteration, `doc.page_content` accesses the `page_content` attribute of the current document, which contains its textual content.
3. The `join` method then takes these pieces of text and concatenates them into a single string, inserting `\n\n` between each piece to ensure they are separated by a blank line in the final result.

The function ultimately returns this newly formatted single string containing all the document contents, neatly separated by blank lines.
"""

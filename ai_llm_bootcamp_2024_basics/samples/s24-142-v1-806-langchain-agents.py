import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##
from utils.MyUtils import clear_terminal, logger 
clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# Basic App: Question & Answering from a Document
from langchain.document_loaders import TextLoader
loader = TextLoader("data/be-good.txt")
document = loader.load()
#**The document is loaded as a Python list with metadata**
logger.info(type(document))
logger.info(len(document))
logger.info(document[0].metadata)
logger.info(f"You have {len(document)} document.")
logger.info(f"Your document has {len(document[0].page_content)} characters")

#**Split the document in small chunks**
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)
document_chunks = text_splitter.split_documents(document)
logger.info(f"Now you have {len(document_chunks)} chunks.")

## Embeddings ##
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction 
embeddings = SentenceEmbeddingFunction()

## Create vector store ChromaDB from documents ##
from utils.MyVectorStore import chroma_from_documents
 
stored_embeddings = chroma_from_documents(
    documents=document_chunks, embedding=embeddings, collection_name="chroma_db_s24_142"
)

from utils.MyVectorStore import chroma_from_documents

#**Create a Retrieval Question & Answering Chain**
from langchain.chains import RetrievalQA
QA_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=stored_embeddings.as_retriever()
)

#**Now we have a Question & Answering APP**
question = """
What is this article about? 
Describe it in less than 100 words.
"""
resp = QA_chain.run(question)
logger.info(type(resp))

question2 = """
And how does it explain how to create somethin people want?
"""
resp = QA_chain.run(question2)
logger.info(type(resp))

## Simple Agent
# from langchain.agents import load_tools
# from langchain.agents import AgentType

# tool_names = ["llm-math"]
# tools = load_tools(tool_names, llm=llm)
# logger.info(tools)
# from langchain.agents import initialize_agent

# agent = initialize_agent(tools,
#                          llm,
#                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                          verbose=True,
#                          max_iterations=3)
# resp = agent.run("What is 133 by 142?")
# logger.info(type(resp))

# #**Let's make the agent fail**
# resp = agent.run("Who was the wife of Napoleon Bonaparte?")
# logger.info(type(resp))

### Custom Agent
from langchain.agents import initialize_agent
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun
from langchain.callbacks.manager import CallbackManagerForToolRun

class CustomSearchTool(BaseTool):
    name = "article search"
    description = "useful for when you need to answer questions about our article"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        store = stored_embeddings.as_retriever()
        docs = store.get_relevant_documents(query)
        text_list = [doc.page_content for doc in docs]
        return "\n".join(text_list)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
from langchain.agents import AgentType

tools = [CustomSearchTool()]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    max_iterations=3
)

resp = agent.run("What is this article about? Describe it in less than 100 words.")
logger.info(type(resp))
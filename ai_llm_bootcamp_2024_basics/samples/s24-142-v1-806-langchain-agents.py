from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

# Basic App: Question & Answering from a Document
from langchain.document_loaders import TextLoader
loader = TextLoader("data/be-good.txt")
document = loader.load()
#**The document is loaded as a Python list with metadata**
print_to_console(type(document))
print_to_console(len(document))
print_to_console(document[0].metadata)
print_to_console(f"You have {len(document)} document.")
print_to_console(f"Your document has {len(document[0].page_content)} characters")

#**Split the document in small chunks**
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)
document_chunks = text_splitter.split_documents(document)
print_to_console(f"Now you have {len(document_chunks)} chunks.")
#**Convert text chunks in numeric vectors (called "embeddings")**

from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
gemini_api_key = os.environ["gemini_api_key"]
#embeddings = OllamaEmbeddings(model=LlmModel.MISTRAL.value, show_progress=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)

#**Load the embeddings to a vector database**
from langchain_chroma import Chroma
from chromadb.config import Settings
stored_embeddings = Chroma.from_documents(document_chunks, embeddings, persist_directory="./chroma_db",client_settings= Settings( anonymized_telemetry=False, is_persistent=True, ))

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
print_to_console(type(resp))

question2 = """
And how does it explain how to create somethin people want?
"""
resp = QA_chain.run(question2)
print_to_console(type(resp))

## Simple Agent
# from langchain.agents import load_tools
# from langchain.agents import AgentType

# tool_names = ["llm-math"]
# tools = load_tools(tool_names, llm=llm)
# print_to_console(tools)
# from langchain.agents import initialize_agent

# agent = initialize_agent(tools,
#                          llm,
#                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                          verbose=True,
#                          max_iterations=3)
# resp = agent.run("What is 133 by 142?")
# print_to_console(type(resp))

# #**Let's make the agent fail**
# resp = agent.run("Who was the wife of Napoleon Bonaparte?")
# print_to_console(type(resp))

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
print_to_console(type(resp))
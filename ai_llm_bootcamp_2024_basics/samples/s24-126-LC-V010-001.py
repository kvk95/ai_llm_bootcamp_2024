#!/usr/bin/env python
# coding: utf-8

# ## LangChain v0.1.0

# #### LangChain's Blog Post and Video about the new release
# * Released Jan 8, 2024.
# * [LangChain v0.1.0](https://blog.langchain.dev/langchain-v0-1-0/)
# * [YouTube Walkthroug](https://www.youtube.com/watch?v=Fk_zZr2DbQY&list=PLfaIDFEXuae0gBSJ9T0w7cu7iJZbH3T31)

# #### Summary
# * First stable version.
# * Fully backwards compatible.
# * Both in Python and Javascript.
# * Improved functionality.
# * Improved documentation.

# #### Main change: the old LangChain package is splitted
# * lanchain-core: core functionality.
# * langchain-community: third-party integrations
# * standalone partner packages
#     * example: langchain-openai

# #### In theory, all previous LangChain projects should work
# * In practice, this does not seem credible.

# #### Example using langchain-core

# #### Example using langchain-community

# #### Example using langchain-openai

# ## New Quickstart
# * Setup LangChain, LangSmith and LangServe.
# * Use basic componets:
#     * Prompt templates.
#     * Models.
#     * Output parsers.
# * Use LCEL.
# * Build a simple app.
# * Trace your app with LangSmith.
# * Serve your app with LangServe. 

# #### Create a new virtual environment

# #### Create a .env file with the OpenAI credentials

# #### LangChain Installation
# pip install langchain

# #### If you set the LangSmith credentials in the .env file, LangChain will start logging traces.

# If you do want to use LangSmith, after you sign up at LangSmith, make sure to set your environment variables in your .env file to start logging traces:

# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=...

# #### What we will cover
# * Simple LLM chain.
# * Retrieval chain.
# * Conversation Retrieval chain.
# * Agent.

# #### Use the new langchain_google_genai
# pip install langchain_google_genai

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

#TODO: uncomment
#logger.info(llm.invoke("What was the name of Napoleon's wife?"))

from langchain_core.prompts import ChatPromptTemplate
my_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are friendly assisstant."),
    ("user", "{input}")
])

#Create a chain with LCEL
my_chain = my_prompt_template | llm 
#TODO: uncomment
#logger.info(my_chain.invoke({"input": "Where was Napoleon defeated?"}))

# Create an Output Parser to convert the chat message to a string
from langchain_core.output_parsers import StrOutputParser
to_string_output_parser = StrOutputParser()
my_chain = my_prompt_template | llm | to_string_output_parser
#TODO: uncomment
#my_chain.invoke({"input": "Where was the main victory of Napoleon?"})

# Simple RAG: Private Document, Splitter, Vector Database and Retrieval Chain.
#We can load our private document from different sources (from a file, from the web, etc). 
# In this example we will load our private data from the web using WebBaseLoader. 
# In order to use WebBaseLoader we will need to install BeautifulSoup:

# pip install beautifulsoup4
# To import WebBaseLoader, we will **use the new langchain_community**:

from langchain_community.document_loaders import WebBaseLoader
my_loader = WebBaseLoader("https://aiaccelera.com/ai-consulting-for-businesses/")
my_private_docs = my_loader.load()
logger.info('----------------------------------------------------------------')
logger.info(my_private_docs)
logger.info('----------------------------------------------------------------')

#https://www.datacamp.com/tutorial/run-llama-3-locally
# We will use Ollama embeddings to convert our private docs to numbers:

#**Split the document in small chunks**
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)
document_chunks = text_splitter.split_documents(my_private_docs)
logger.info(f"Now you have {len(document_chunks)} chunks.")

## Embeddings ##
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction 
my_embeddings = SentenceEmbeddingFunction()

## Create vector store ChromaDB from documents ##
from utils.MyVectorStore import chroma_from_documents 
my_vector_database = chroma_from_documents(
    documents=document_chunks, embedding=my_embeddings, collection_name="chroma_s24-126"
)

#Now we will create a chain that takes the question and the retrieved documents and generates an answer:
from langchain.chains.combine_documents import create_stuff_documents_chain
my_prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based only on the 
    provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
)
my_document_answering_chain = create_stuff_documents_chain(llm, my_prompt_template)

#Next we will create the retrieval chain:
from langchain.chains import create_retrieval_chain
my_retriever = my_vector_database.as_retriever()
my_retrieval_chain = create_retrieval_chain(my_retriever, my_document_answering_chain)
#We can now start using the retrieval chain:
response = my_retrieval_chain.invoke({
    "input": "Summarize the provided context in less than 100 words"
})
logger.info('----------------------------------------------------------------')
logger.info(response["answer"])
logger.info('----------------------------------------------------------------')
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

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# region gemini
gemini_api_key = os.environ["gemini_api_key_vijay"]
#gemini_api_key = os.environ["gemini_api_key"]
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", 
                             google_api_key=gemini_api_key)

# endregion gemini

# region llama3.1
# pip install -qU langchain-ollama  
# from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0,
#     # other params...
# )
# endregion llama3.1

#TODO: uncomment
#print(llm.invoke("What was the name of Napoleon's wife?"))

from langchain_core.prompts import ChatPromptTemplate
my_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are friendly assisstant."),
    ("user", "{input}")
])

#Create a chain with LCEL
my_chain = my_prompt_template | llm 
#TODO: uncomment
#print(my_chain.invoke({"input": "Where was Napoleon defeated?"}))

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
print('----------------------------------------------------------------')
print(my_private_docs)
print('----------------------------------------------------------------')

#https://www.datacamp.com/tutorial/run-llama-3-locally
# We will use Ollama embeddings to convert our private docs to numbers:

#from langchain_community.embeddings import OllamaEmbeddings
#my_embeddings = OllamaEmbeddings(model="llama3.1", show_progress=True)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
my_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)

#We will use Chroma as vector database:
# pip install langchain_chroma
from langchain_chroma import Chroma

#We will use RecursiveCharacterTextSplitter to divide the private docs into smaller text chunks:
from langchain.text_splitter import RecursiveCharacterTextSplitter
my_text_splitter = RecursiveCharacterTextSplitter()
my_text_chunks = my_text_splitter.split_documents(my_private_docs)
my_vector_database = Chroma.from_documents(my_text_chunks, my_embeddings)

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
print('----------------------------------------------------------------')
print(response["answer"])
print('----------------------------------------------------------------')
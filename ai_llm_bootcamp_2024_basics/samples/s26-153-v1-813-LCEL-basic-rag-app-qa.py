from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic App: Question & Answering from a Document
#**Load the text file**
from langchain.document_loaders import TextLoader
loader = TextLoader("data/be-good-and-how-not-to-die.txt")
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
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_api_key = os.environ["gemini_api_key"]
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)

from langchain_chroma import Chroma
from chromadb.config import Settings
stored_embeddings = Chroma.from_documents(document_chunks, 
                                          embeddings, 
                                          persist_directory="./chroma_db",
                                          client_settings= Settings( anonymized_telemetry=False, is_persistent=True, ))

# **Create a Retrieval Question & Answering Chain**
from langchain.chains import RetrievalQA
QA_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=stored_embeddings.as_retriever()
)

question = """
What is this article about? 
Describe it in less than 100 words.
"""

# resp = QA_chain.run(question)
# print_to_console(resp)

## New way: with LCEL
from langchain.prompts import PromptTemplate

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
retriever = stored_embeddings.as_retriever()

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("What is this article about? Describe it in less than 100 words.")
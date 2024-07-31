from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

## Basic RAG app with the vector database DeepLake
#**Name the new database you will create**
my_activeloop_dataset_name = "basic-rag-with-deeplake"

#**Load dependencies**
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

#**Create the external knowledge document**
usa_curious_facts = [
    """
    The US celebrates Independence Day from the British Empire 
    on July 4. However, the country’s Declaration of Independence 
    was passed on July 2. It was only officially ratified on July 4.
    """,
    """
    The very first documented European to arrive in North America was 
    the Spaniard Juan Ponce de León, who landed in Florida in 1513.
    """
]

#**Divide the document in smaller chunks of text**
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 0
)
doc_chunks = text_splitter.create_documents(usa_curious_facts)
print(f"Now you have {len(doc_chunks)} chunks.")

#**Create the Chroma vector database**
gemini_api_key = os.environ["gemini_api_key"]
#embeddings = OllamaEmbeddings(model=LlmModel.MISTRAL.value, show_progress=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)

db = Chroma.from_documents(
        doc_chunks, embeddings, persist_directory="./chroma_db",
                                          client_settings= Settings( anonymized_telemetry=False, is_persistent=True, ))
# **Create the QA Chain**
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

#**Ask the App about the document**
# resp = qa_chain.invoke("When was actually passed the U.S. Declaration of Independence?")
# print_to_console(resp)

#**Add new data to the vector database**
additional_usa_curious_facts = [
    """
    Alaska is the largest state in the US, and used to belong 
    to the Russian Empire before the US purchased it.
    """,
    """
    Big cities and regions have their own style of pizza: Chicago 
    Deep-Dish, New York Style, Detroit Pizza, St Louis-Style, and 
    New England Beach Pizza are just a few different varieties.
    """
]

additional_doc_chunks = text_splitter.create_documents(additional_usa_curious_facts)
db.add_documents(additional_doc_chunks)

#**Ask the app about the new data**
# resp = qa_chain.invoke("What is the largest state in the US?")
# print_to_console(resp)

# resp = qa_chain.invoke("Tell me 3 states with their own style of pizza")
# print_to_console(resp)

## Similarity Search

resp = db.similarity_search_with_score("What is the largest state in the US?", k=1)
print_to_console(resp)

## Retriever
retriever = db.as_retriever(search_kwargs={"k": 1})
resp = retriever.get_relevant_documents(query="What is the largest state in the US?")
print_to_console(resp)

## Indexing API
# In case you need to update the contents of your vector database.


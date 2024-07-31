from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic RAG app with the vector database Chroma
# **Name the new database you will create**

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQA
import os
gemini_api_key = os.environ["gemini_api_key"]

# **Create the external knowledge document**
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

# **Divide the document in smaller chunks of text**
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 0
)
doc_chunks = text_splitter.create_documents(usa_curious_facts)
print_to_console(f"Now you have {len(doc_chunks)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)


db = Chroma.from_documents(doc_chunks, embeddings, 
                                          persist_directory="./chroma_s28_172_db",
                                          client_settings= Settings( anonymized_telemetry=False, 
                                                                    is_persistent=True, ))
print_to_console(db)

# **Create the QA Chain**
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)
resp = qa_chain.run("When was actually passed the U.S. Declaration of Independence?")
print_to_console(resp)

# **Add new data to the vector database**
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
print_to_console(db)
# **Ask the app about the new data**
resp = qa_chain.run("What is the largest state in the US?")
print_to_console(resp)

resp = qa_chain.run("Tell me 3 states with their own style of pizza")
print_to_console(resp)
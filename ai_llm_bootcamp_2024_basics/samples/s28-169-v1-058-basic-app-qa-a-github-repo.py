from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic app for QA a a code library or a github repository

# **Load the github repo**
# <br>
# We will load the code of the github repo The Fuzz, a small python module for string matching.

root_dir = "data/thefuzz-master"

document_chunks = []
from langchain.document_loaders import TextLoader
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(
                os.path.join(dirpath, file),
                encoding="utf-8"
            )
            document_chunks.extend(loader.load_and_split())
        except Exception as e:
            pass

print_to_console(f"We have {len(document_chunks)} chunks.")
print_to_console(document_chunks[0].page_content[:300])

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
gemini_api_key = os.environ["gemini_api_key"]
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)

#**Load the embeddings to a vector database**
from langchain_chroma import Chroma
from chromadb.config import Settings
stored_embeddings = Chroma.from_documents(document_chunks, embeddings, 
                                          persist_directory="./chroma_db_169",
                                          client_settings= Settings( anonymized_telemetry=False, 
                                                                    is_persistent=True, ))

# **Create the RetrievalQA chain**
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=stored_embeddings.as_retriever()
)

# **Now we can make questions about the github library**
question = """
What function do I use if I want to find 
the most similar item in a list of items?
"""
answer = qa_chain.run(question)
print_to_console(answer)
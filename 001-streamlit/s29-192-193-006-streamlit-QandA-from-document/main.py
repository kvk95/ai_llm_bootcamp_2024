import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import clear_terminal, logger
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st
import sys
from utils.MyVectorStore import chroma_from_documents
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

## Logging ##
# clear_terminal()

## Foundation Model ##
llm: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)


# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import sqlite3


# Input .txt file
# Format file
# Split file
# Create embeddings
# Store embeddings in vector store
# Input query
# Run QA chain
# Output


def generate_response(file, openai_api_key, query):
    # format file
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    # split file
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(formatted_document)
    # create embeddings
    embeddings = SentenceEmbeddingFunction()
    # load to vector database
    store = chroma_from_documents(
        documents=docs, embedding=embeddings, collection_name="chroma_db_thirukural_193"
    )

    # store = FAISS.from_documents(docs, embeddings)

    # create retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=store.as_retriever()
    )
    # run chain with query
    return retrieval_chain.run(query)


st.set_page_config(page_title="Q&A from a long PDF Document")
st.title("Q&A from a long PDF Document")

uploaded_file = st.file_uploader("Upload a .pdf document", type="pdf")

query_text = st.text_input(
    "Enter your question:",
    placeholder="Write your question here",
    disabled=not uploaded_file,
)

result = []
with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key:", type="password", disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit", disabled=not (uploaded_file and query_text)
    )
    if submitted:  # and openai_api_key.startswith("sk-"):
        with st.spinner("Wait, please. I am working on it..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)

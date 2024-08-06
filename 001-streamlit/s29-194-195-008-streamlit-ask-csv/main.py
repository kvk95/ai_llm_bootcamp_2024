import os, sys
parent_dir_path = os.path.abspath(os.curdir)
sys.path.insert(0, parent_dir_path)


## Logging ##
from utils.MyUtils import clear_terminal, logger 
#clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyVectorStore import chroma_from_documents,chroma_get

import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

#LLM and key loading function
def load_LLM(openai_api_key):
    return llm


#Page title and header
st.set_page_config(page_title="Ask from CSV File with FAQs about Napoleon")
st.header("Ask from CSV File with FAQs about Napoleon")

openai_api_key = 'abc'
if openai_api_key:
    embedding = SentenceEmbeddingFunction()

    collection_name = "008-streamlit-ask-csv"
    vectordb_file = "008-streamlit-ask-csv"

    def create_db():
        csv_file = 'data/napoleon-faqs.csv'
        csv_file_path = os.path.abspath(os.path.join(parent_dir_path, csv_file))
        loader = CSVLoader(file_path=csv_file_path, source_column="prompt")
        documents = loader.load()
        vectordb = chroma_from_documents(
            documents=documents, embedding=embedding, 
            collection_name=collection_name,
            persist_directory=vectordb_file
        )
        return vectordb


    def execute_chain():
        # Load the vector database from the local folder
        #vectordb = FAISS.load_local(vectordb_file, embedding)
        vectordb = chroma_get(
                persist_directory=vectordb_file,
                embedding_function=embedding,
                collection_name=collection_name
            )

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(
            score_threshold=0.7,
        )

        template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        llm = load_LLM(openai_api_key=openai_api_key)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return chain

    def main():
        create_db()
        chain = execute_chain()
        question = "where do you come from?"
        response = chain(question)
        logger.info(response)

    # if __name__ == "__main__":
    #     main()

    btn = st.button("Private button: re-create database")
    if btn:
        create_db()

    question = st.text_input("Question: ")
    if(len(question)>0):
        with st.spinner(
                "Checking with available information..."
                ):
            chain = execute_chain()
            response = chain(question)
            logger.info(response)

            st.header("Answer")
            st.write(response["result"])
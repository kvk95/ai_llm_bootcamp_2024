import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import logger
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyVectorStore import chroma_from_documents, chroma_get

## Logging ##
# clear_terminal()

## Foundation Model ##
llm: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

file_path = "./data/test.xlsx"
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def load_xl():

    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()
    return docs


def split_documents(docs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
    )
    documents = text_splitter.split_documents(docs)
    return documents


db_name = "excel_test_01"
embeddings = SentenceEmbeddingFunction()


def create_vector_db(documents):

    chroma_from_documents(
        documents,
        embedding=embeddings,
        collection_name=db_name,
        persist_directory=db_name,
    )


def execute_chain():
    # Load the vector database from the local folder
    # vectordb = FAISS.load_local(vectordb_file, embedding)
    vectordb = chroma_get(
        persist_directory=db_name,
        embedding_function=embeddings,
        collection_name=db_name,
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

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def main():
    docs = load_xl()
    documents = split_documents(docs)
    create_vector_db(documents)
    chain = execute_chain()
    question = "Where is 'Arbutus 'Marina' :: Hybrid Strawberry Tree'?"
    response = chain.invoke(question)
    logger.info(response)


if __name__ == "__main__":
    main()

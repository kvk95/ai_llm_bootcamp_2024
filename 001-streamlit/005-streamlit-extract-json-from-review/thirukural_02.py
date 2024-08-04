import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##
from utils.MyUtils import clear_terminal, logger 
#clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 

#---------------------------------------------------------

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import PyPDFLoader
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction 

# Load documents from filesystem
def load_documents():
    pdf_file = "data/Dharma_in_Tirukkural.pdf"
    pdf_file_path = os.path.abspath(os.path.join(parent_dir_path, pdf_file))
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    logger.info(f"\033[1;31;40m documents count \033[0m \033[0;33;40m {len(documents)} \033[0m \n")  
    return documents

#---------------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)  
    return chunked_documents
 
#---------------------------------------------------------

from utils.MyVectorStore import chroma_from_documents
     
# Set up vector DB
def setup_vectordb(chunked_documents):
    embeddings = SentenceEmbeddingFunction()
    # vectdb = Chroma.from_documents(documents=chunked_documents,
    #     embedding=embedding_function, 
    #     persist_directory="./chroma_db",
    #     client_settings= Settings( anonymized_telemetry=False, is_persistent=True, ))
    vectdb = chroma_from_documents(
        documents=chunked_documents, embedding=embeddings, collection_name="chroma_db_thirukural_02"
    )
    return vectdb

#---------------------------------------------------------
 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set up retriever chain
def setup_retriever_chain(llm, retriever):
    system_template = """
        If the given is not a thirukural tell its not a thirukural
        Only explain if you are completely sure that the information given is accurate. 
        {context}
        
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("user", "{input}"),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Set up document chain
def setup_document_chain(llm):
    template = """
        {input}
        You are a expert tamil pandit and english phd who can give a immense explanation and story for thirukural in both english and tamil
        based on the below context. 
        The thirukural might be given in english or tamil
        If the given is not a thirukural tell its not a thirukural
        Only explain if you are completely sure that the information given is accurate. 
        Refuse to explain otherwise. 
        Make sure your explanation are detailed. 
        Include from which which "அதிகாரம்/Chapter:"
        Make a story explain the topic precisly 
        Format the output as bullet-points text with the following keys:
        - actual_explantion
            - English
            - Tamil
        - அதிகாரம்/Chapter:
            - English
            - Tamil
        - story
            - English
            - Tamil
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Explain thirukural based on the below context:\n\n{context}"),
            ("user", template)
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

# Set up QA chain
def setup_qa_chain(retriever_chain, document_chain):
    qa = create_retrieval_chain(retriever_chain, document_chain)
    return qa

#---------------------------------------------------------

def create_upto_retriever():
    # upload all project files
    documents = load_documents()

    # Split the Document into chunks for embedding and vector storage
    documents = split_documents(documents)

    #transformer_model_path = os.path.abspath(os.path.join(parent_dir_path, os.getenv["TRANSFORMER_MODEL_BASE_PATH"]))
    vectdb = setup_vectordb(documents)
    return vectdb;

def create_qa(vectdb):    

    retriever = vectdb.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    # use gemini or ollama  
    llm: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

    retriever_chain = setup_retriever_chain(llm, retriever)
    document_chain = setup_document_chain(llm)
    qa = setup_qa_chain(retriever_chain, document_chain)
    return qa

#---------------------------------------------------------
def main():    

    vectdb = create_upto_retriever()

    question = "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு"
    # result = qa.invoke({"input": question})
    # logger.info(result["answer"])
    qa = create_qa(vectdb)
    print(qa)
    for chunk in qa.stream({"input": question}):
        if answer_chunk := chunk.get("answer"):
            print(answer_chunk)

# if __name__ == "__main__":
#     main()


import streamlit as st
from streamlit import session_state as ss

#Page title and header
st.set_page_config(page_title="Thirukural Explanation")
st.header("Explain kural")

# Input
st.markdown("## Enter the Kural")

def get_kural():
    kural_text = st.text_area(label="Kural in English or tamil", label_visibility='collapsed', placeholder="Your Product kural...", key="kural_input")
    return kural_text

kural_input = get_kural()

if len(kural_input.split(" ")) > 700:
    st.write("Please enter a shorter product kural. The maximum length is 700 words.")
    st.stop()

    
# Output
st.markdown("### Kural Explanation:")

vectdb = create_upto_retriever()
if 'vector_store' not in st.session_state:
      st.session_state.vector_store = vectdb

if len(kural_input) > 0:    

    question = "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு"
    
    # result = qa.invoke({"input": question})
    vectdb = st.session_state.vector_store;
    qa = create_qa(vectdb=vectdb);
    # st.write_stream(qa.stream({"input": kural_input}))
    st.write(kural_input)
    for chunk in qa.stream({"input": kural_input}):
        if answer_chunk := chunk.get("answer"):
            st.write(answer_chunk)
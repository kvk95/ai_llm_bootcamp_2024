import os, sys
parent_dir_path = os.path.abspath(os.curdir)
sys.path.insert(0, parent_dir_path)
print(parent_dir_path)


## Logging ##
from utils.MyUtils import clear_terminal, logger 
#clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm_model : LlmModel = LlmModel.GEMINI
llm: BaseChatModel = init_llm(llm_model, temperature=0)

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyVectorStore import chroma_from_texts

import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

def generate_response(
    uploaded_file,
    openai_api_key,
    query_text,
    response_text
):
    #format uploaded file
    documents = [uploaded_file.read().decode()]
    
    #break it in small chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    texts = text_splitter.create_documents(documents)
    embeddings = SentenceEmbeddingFunction()

    vectordb_file_path = "resources/chromadb/009-streamlit-evaluate-QandA-from-long-document"
    vectordb_file_path = os.path.abspath(os.path.join(parent_dir_path, vectordb_file_path))
    
    # create a vectorstore and store there the texts
    vectordb = chroma_from_texts(
        texts=documents, embedding=embeddings, 
        collection_name="009-streamlit-evaluate-QandA-from-long-document",
        persist_directory=vectordb_file_path
    )
    #db = FAISS.from_documents(texts, embeddings)
    
    # create a retriever interface
    retriever = vectordb.as_retriever()
    
    # create a real QA dictionary
    real_qa = [
        {
            "question": query_text,
            "answer": response_text
        }
    ]
    
    # regular QA chain
    qachain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )
    
    # predictions
    predictions = qachain.apply(real_qa)
    
    # create an eval chain
    eval_chain = QAEvalChain.from_llm(
        llm=init_llm(llm_model, temperature=0)
    )
    # have it grade itself
    graded_outputs = eval_chain.evaluate(
        real_qa,
        predictions,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )
    
    response = {
        "predictions": predictions,
        "graded_outputs": graded_outputs
    }
    
    return response

st.set_page_config(
    page_title="Evaluate a RAG App"
)
st.title("Evaluate a RAG App")

with st.expander("Evaluate the quality of a RAG APP"):
    st.write("""
        To evaluate the quality of a RAG app, we will
        ask it questions for which we already know the
        real answers.
        
        That way we can see if the app is producing
        the right answers or if it is hallucinating.
    """)

uploaded_file = st.file_uploader(
    "Upload a .txt document",
    type="txt"
)

query_text = st.text_input(
    "Enter a question you have already fact-checked:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    openai_api_key = 'sk-1234'

    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner(
            "Wait, please. I am working on it..."
            ):
            response = generate_response(
                uploaded_file,
                openai_api_key,
                query_text,
                response_text
            )
            result.append(response)
            del openai_api_key
            
print(result)
if len(result):
    st.write("Question")
    st.info(response["predictions"][0]["question"])
    st.write("Real answer")
    st.info(response["predictions"][0]["answer"])
    st.write("Answer provided by the AI App")
    st.info(response["predictions"][0]["result"])
    st.write("Therefore, the AI App answer was")
    st.info(response["graded_outputs"][0]["results"])
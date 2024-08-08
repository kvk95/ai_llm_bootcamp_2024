import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##

# clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm

# ---------------------------------------------------------

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction


# Load documents from filesystem
def load_documents():
    from PyPDF2 import PdfReader

    files = [
        "data/Dharma_in_Tirukkural.pdf",
        "data/tirukkuRaL_Suddhananda_Bharathiyar.pdf",
    ]
    full_pdf_texts = []

    for pdf_file in files:
        pdf_file_path = os.path.abspath(os.path.join(parent_dir_path, pdf_file))

        reader = PdfReader(pdf_file_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]

        # Filter the empty strings
        pdf_texts = [text for text in pdf_texts if text]
        full_pdf_texts += pdf_texts
    return full_pdf_texts


# ---------------------------------------------------------


# Split documents into chunks
def split_documents(pdf_texts):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
    # chunked_documents = text_splitter.split_documents(documents)
    return character_split_texts


# ---------------------------------------------------------

from utils.MyVectorStore import chroma_get


# Set up vector DB
def setup_vectordb(character_split_texts):

    from langchain_text_splitters import SentenceTransformersTokenTextSplitter

    embeddings = SentenceEmbeddingFunction()

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0,
        tokens_per_chunk=256,
        model_name=embeddings.transformer_model_path,
    )
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    vectdb = chroma_get(
        embedding_function=embeddings,
        persist_directory="chroma_db_thirukural_02",
        collection_name="chroma_db_thirukural_02",
    )
    chroma_collection = vectdb._client.get_collection("chroma_db_thirukural_02")

    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)
    return vectdb


# ---------------------------------------------------------

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Set up retriever chain
def setup_retriever_chain(llm, retriever):
    system_template = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation        
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            ("user", system_template),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


# Set up document chain
def setup_document_chain(llm):
    template = """
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
        based on the below context:\n\n{context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("user", "{input}")]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


# Set up QA chain
def setup_qa_chain(retriever_chain, document_chain):
    qa = create_retrieval_chain(retriever_chain, document_chain)
    return qa


# ---------------------------------------------------------


def create_upto_retriever():
    # upload all project files
    pdf_texts = load_documents()

    # Split the Document into chunks for embedding and vector storage
    character_split_texts = split_documents(pdf_texts)

    # transformer_model_path = os.path.abspath(os.path.join(parent_dir_path, os.getenv["TRANSFORMER_MODEL_BASE_PATH"]))
    vectdb = setup_vectordb(character_split_texts)
    return vectdb


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


# ---------------------------------------------------------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def create_qa_new(vectdb):

    retriever = vectdb.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    template = """
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
        based on the below context:\n\n{context}
    """

    # use gemini or ollama
    llm: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("user", "{input}")]
    )

    retrieval_chain = (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


# ---------------------------------------------------------
def main():

    vectdb = create_upto_retriever()

    question = "ஒழுக்கம் விழுப்பந் தரலான் ஒழுக்கம் உயிரினும் ஓம்பப் படும்"
    qa = create_qa(vectdb)
    result = qa.invoke({"input": question})
    print(result.get("answer"))
    # for chunk in qa.stream({"input": question}):
    #     if answer_chunk := chunk.get("answer"):
    #         print(answer_chunk)


if __name__ == "__main__":
    main()


# import streamlit as st

# # Page title and header
# st.set_page_config(page_title="Thirukural Explanation")
# st.header("Explain kural")

# # Input
# st.markdown("## Enter the Kural")


# def get_kural():
#     kural_text = st.text_area(
#         label="Kural in English or tamil",
#         label_visibility="collapsed",
#         placeholder="Your Product kural...",
#         key="kural_input",
#     )
#     return kural_text


# kural_input = get_kural()

# if len(kural_input.split(" ")) > 700:
#     st.write("Please enter a shorter product kural. The maximum length is 700 words.")
#     st.stop()


# # Output
# st.markdown("### Kural Explanation:")

# vectdb = create_upto_retriever()
# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = vectdb

# if len(kural_input) > 0:

#     question = "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு"
#     with st.spinner("Thinking...&..Generating"):
#         # result = qa.invoke({"input": question})
#         vectdb = st.session_state.vector_store
#         qa = create_qa(vectdb)
#         # st.write_stream(qa.stream({"input": kural_input}))
#         st.write(kural_input)
#         result = qa.invoke({"input": kural_input})
#         st.write(result.get("answer"))
#         # for chunk in qa.stream({"input": kural_input}):
#         #     if answer_chunk := chunk.get("answer"):
#         #         st.write(answer_chunk)

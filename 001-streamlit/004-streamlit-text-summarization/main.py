import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##
from utils.MyUtils import clear_terminal, logger 
clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.run(docs)

st.set_page_config(
    page_title = "Writing Text Summarization"
)
st.title("Writing Text Summarization")

txt_input = st.text_area(
    "Enter your text",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value="sk-123456",
        type="password",
        disabled=not txt_input
    )
    submitted = st.form_submit_button("Submit")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(txt_input)
        result.append(response)
        del openai_api_key

if len(result):
    st.info(response)
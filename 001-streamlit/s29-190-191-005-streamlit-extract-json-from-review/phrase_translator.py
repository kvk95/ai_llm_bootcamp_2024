import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import logger

## Logging ##
# clear_terminal()

## Foundation Model ##
llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

logger.info("")
text = """
Need tamil text for the below saying
Do not let difficulties with others break your spirit or distract you from your goals. 
Hate and envy from others are inevitable and draining parts of life, 
but they will only win if they make you hate them back. 
So, by rising above those harmful feelings, you keep your strength and sense of self, 
denying their attempts to drag you down and rejecting the pleasure they seek.
"""
# response = llm.invoke(text)
# logger.info(response)

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import streamlit as st

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """You are expert in translating from {source_language} to {dest_language}.
            Translate the below phrase to "{dest_language}" """
        ),
        HumanMessagePromptTemplate.from_template("{phrase}"),
    ]
)

st.header("Multi language Translater")
col1, col2 = st.columns(2)
with col1:
    option_source_language = st.selectbox("From Language?", ("English", "Tamil"))

with col2:
    option_dest_language = st.selectbox("To Language", ("Tamil", "English"))


def get_phrase():
    phrase_text = st.text_area(
        label="Phrase to convert",
        label_visibility="collapsed",
        placeholder="Your Text...",
        key="input_phrase",
    )
    return phrase_text


input_phrase = get_phrase()

if len(input_phrase.split(" ")) < 20 or len(input_phrase.split(" ")) > 700:
    st.write("Phrase should be between 20 to 700 words")
    st.stop()

chain = prompt | llm

with st.spinner("Translatting..."):
    response = chain.invoke(
        {
            "source_language": option_source_language,
            "dest_language": option_dest_language,
            "phrase": input_phrase,
        }
    )
    st.write(response.content)

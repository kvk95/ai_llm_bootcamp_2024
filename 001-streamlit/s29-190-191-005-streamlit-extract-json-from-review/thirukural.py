import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##
from utils.MyUtils import clear_terminal, logger 
#clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

import streamlit as st
from streamlit import session_state as ss
from langchain_core.prompts import PromptTemplate

# variables
if 'stream' not in ss:
    ss.stream = None

template = """
    You are a expert tamil pandit and english phd who can give a immense explanation for thirukural in both english and tamil. 
    Give a clear and detailed explanation of {kural} considering the below points
    The thirukural might be given in english or tamil
    If the given is not a thirukural tell its not a thirukural
    Only explain if you are completely sure that the information given is accurate. 
    Refuse to explain otherwise. 
    Make sure your explanation are detailed. 
    Include from which "பால்:" and which "அதிகாரம்/Chapter:"
    Make screenplay & script, out of which we can make a video to explain the topic precisly    
    Also include layman explanation so it will easy for non chemistry person to understand.
    Format the output as bullet-points text with the following keys:
    - actual_explantion
     - பால்:
     - அதிகாரம்/Chapter:
     - English
     - Tamil
    - layman_explantion
     - English
     - Tamil
    - screenplay_script
     - English
     - Tamil
    """

#PromptTemplate variables definition
prompt = PromptTemplate(

    input_variables=["kural"],
    template=template,
)


#Page title and header
st.set_page_config(page_title="Thirukural Explanation")
st.header("Explain kural")


# #Intro: instructions
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("Extract key information from a product kural.")
#     st.markdown("""
#         - Sentiment
#         - How long took it to deliver?
#         - How was its price perceived?
#         """)

# with col2:
#     st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


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

if kural_input:
    prompt_with_kural = prompt.format(
        kural=kural_input
    )

    # key_data_extraction = llm.stream(prompt_with_kural)
    # logger.info(key_data_extraction)

    # st.write_stream(key_data_extraction)
    st.write(kural_input)
    st.write_stream(llm.stream(prompt_with_kural))
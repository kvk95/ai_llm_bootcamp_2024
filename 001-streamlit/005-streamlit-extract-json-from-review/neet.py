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
    You are a chemistry profesor specialist whose job is to explain chemistry topics. 
    Only explain if you are completely sure that the information given is accurate. 
    Refuse to explain otherwise. 
    Make sure your explanation are detailed.
    If a topic contains subtopics
     - list subtopics in main topic with one or 2 line explanation
     - after main topic explain each topi in details
    Make screenplay & script, out of which we can make a video to explain the topic precisly
    Give a clear and detailed explanation of {neet_concept}
    Also include layman explanation so it will easy for non chemistry person to understand.
    Format the output as bullet-points text with the following keys:
    - actual_explantion
    - layman_explantion
    - screenplay_script
    """

#PromptTemplate variables definition
prompt = PromptTemplate(
    input_variables=["neet_concept"],
    template=template,
)


#Page title and header
st.set_page_config(page_title="Chemistry Explanation")
st.header("Explain chemistry topics or questions")


# #Intro: instructions
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("Extract key information from a product review.")
#     st.markdown("""
#         - Sentiment
#         - How long took it to deliver?
#         - How was its price perceived?
#         """)

# with col2:
#     st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


# Input
st.markdown("## Enter the Topic or question")

def get_review():
    review_text = st.text_area(label="Topic or Question", label_visibility='collapsed', placeholder="Your Product Review...", key="review_input")
    return review_text

review_input = get_review()

if len(review_input.split(" ")) > 700:
    st.write("Please enter a shorter product review. The maximum length is 700 words.")
    st.stop()

    
# Output
st.markdown("### Key Data Extracted:")

if review_input:
    prompt_with_review = prompt.format(
        neet_concept=review_input
    )

    # key_data_extraction = llm.stream(prompt_with_review)
    # logger.info(key_data_extraction)

    # st.write_stream(key_data_extraction)
    st.write_stream(llm.stream(prompt_with_review))
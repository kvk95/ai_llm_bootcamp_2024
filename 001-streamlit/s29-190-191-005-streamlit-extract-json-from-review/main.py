import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##
from utils.MyUtils import clear_terminal, logger 
clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


template = """\
For the following text, extract the following \
information:

sentiment: Is the customer happy with the product? 
Answer Positive if yes, Negative if \
not, Neutral if either of them, or Unknown if unknown.

delivery_days: How many days did it take \
for the product to arrive? If this \
information is not found, output No information about this.

price_perception: How does it feel the customer about the price? 
Answer Expensive if the customer feels the product is expensive, 
Cheap if the customer feels the product is cheap,
not, Neutral if either of them, or Unknown if unknown.

Format the output as html to place inside a div with the \
following keys:
- Sentiment
- How long took it to deliver?
- How was the price perceived?

Input example:
This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap

text: {review}
"""

#PromptTemplate variables definition
prompt = PromptTemplate(
    input_variables=["review"],
    template=template,
)


#Page title and header
st.set_page_config(page_title="Extract Key Information from Product Reviews")
st.header("Extract Key Information from Product Reviews")


#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("Extract key information from a product review.")
    st.markdown("""
        - Sentiment
        - How long took it to deliver?
        - How was its price perceived?
        """)

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


#Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  
                               value="sk-123456",
                               placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()


# Input
st.markdown("## Enter the product review")

def get_review():
    review_text = st.text_area(label="Product Review", label_visibility='collapsed', placeholder="Your Product Review...", key="review_input")
    return review_text

review_input = get_review()

if len(review_input.split(" ")) > 700:
    st.write("Please enter a shorter product review. The maximum length is 700 words.")
    st.stop()

    
# Output
st.markdown("### Key Data Extracted:")

if review_input:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
        st.stop()

    prompt_with_review = prompt.format(
        review=review_input
    )

    key_data_extraction = llm.invoke(prompt_with_review)

    st.write(key_data_extraction)
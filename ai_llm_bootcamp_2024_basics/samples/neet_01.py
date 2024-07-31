from comm_init import init_llm, LlmModel, print_to_console
import os

from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import os
import streamlit as st
import json

HUGGINGFACE_API_TOKEN = os.getenv("hugging_face_token")

llm = init_llm(LlmModel.LLAMA)

def story_generator(scenario):
    template = """
    You are an expert kids story teller;
    You can generate short stories based on a simple narrative
    Your story should be more than 50 words.

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables = ["scenario"])
    story_llm = LLMChain(llm = llm , prompt=prompt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)
    return story

def neet_story_generator(neet_concept:str) -> str:
    template = """
    Give a clear and detailed explanation of {neet_concept} from an NCERT book along with a detailed story for a 17 years old to understand the concept. 
    Please include the characteristics of each character in the story. 
    The response should be in JSON format with three fields: 
    'actual_concept', 
    'story', 
    'characteristics'.
    """
    prompt = PromptTemplate(template=template, input_variables = ["neet_concept"])
    story_llm = LLMChain(llm = llm , prompt=prompt, verbose=True)
    
    story = story_llm.predict(neet_concept=neet_concept)
    return story



scenario = """
*With alpha begins all alphabets; And the world with the first Bagavan.

with the above quotes, make a short traditional tamil storey for 6 year old, not exceeding 250 words*
"""
#story = story_generator(scenario) # create a story
#TODO: uncomment
#print_to_console(story)

#Daniell cel
neet_concept = """
oxidation states in the first series of the transition elements? Illustrate your answer with examples.
"""
story = neet_story_generator(neet_concept) # create a story
#TODO: uncomment
print_to_console(story)
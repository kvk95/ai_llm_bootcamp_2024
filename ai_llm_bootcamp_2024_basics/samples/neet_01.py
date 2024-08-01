

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

## Logging ##
from utils.MyUtils import clear_terminal, logger 
#clear_terminal()

## Foundation Model ##
from utils.MyModels import BaseChatModel, LlmModel, init_llm 
llm: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import streamlit as st
import json

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
    You are a chemistry profesor specialist whose job is to explain chemistry topics. 
    Only explain if you are completely sure that the information given is accurate. 
    Refuse to explain otherwise. 
    Make sure your explanation are detailed. 
    Give a clear and detailed explanation of {neet_concept}
    Also include layman explanatin so it will easy for non chemistry person to understand. 
    The response should be in JSON format with three fields: 
    'actual_explantion', 
    'layman_explantion'
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
logger.info(story)
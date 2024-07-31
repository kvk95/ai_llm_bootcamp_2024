from comm_init import init_llm, LlmModel, print_to_console
import os

llm = init_llm(LlmModel.MISTRAL)
# question = "Name the most popular 4 U.S. presidents"
# response = llm.invoke(question)
#TODO: uncomment
# print_to_console(response)

from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import os
import streamlit as st
import json

HUGGINGFACE_API_TOKEN = os.getenv("hugging_face_token")

def story_generator(language:str, age: int, scenario: str):
    template = """
    You are an expert kids story teller; Provide story for Thirukural in {language} for a {age}
    #. Give a clear and detailed explanation
    #. Generate short stories based on Thirukural.
    #. Characteristics of each character in the story. 

    CONTEXT: Thirukural {scenario}
    
    The response should be in JSON format with fields (no need to format json): 
    'Thirukural'
    'actual_concept', 
    'story', 
    'characteristics'.
    """
    prompt = PromptTemplate(template=template, input_variables = ["language","age","scenario"])
    story_llm = LLMChain(llm = llm , prompt=prompt, verbose=True)
    
    story = story_llm.predict(language=language, age=age, scenario=scenario)
    return story

#text-to-speech (Hugging Face)
def text2speech(msg):
    #https://huggingface.co/espnet/kan-bayashi_ljspeech_vits
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads = {
         "inputs" : msg
    }
    data = json.dumps(payloads).encode("utf-8")
    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.flac','wb') as f:
        f.write(response.content)

scenario = """
1. அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு
"""
story = story_generator('english', 6, scenario) # create a story
#TODO: uncomment
# Convert the data to a JSON formatted string with 4 spaces of indentation
json_str = json.dumps(story, indent=4)

from pprint import pprint
# Print the pretty-printed JSON string
print_to_console(json_str)

story="""
Once upon a time, in a land far away, there lived a wise and kind Bagavan, the first of all beings. He was the one who taught the world about love, kindness, and respect for all living things.

   One day, Bagavan decided to create the alphabets, the building blocks of language. With a gentle touch, he carved each letter on the leaves of the sacred Bodhi tree. From A to Z, each letter was filled with magic and wisdom.

   As the wind blew through the tree, the letters floated away, landing in every corner of the world. People were amazed as they found these magical letters. They began to use them to communicate, share stories, and learn from one another.

   Bagavan's gift brought peace and unity among the people. They learned to understand each other better, and conflicts faded away. The world became a beautiful place filled with love and harmony, just as Bagavan had always wished.

   And so, every time we write or read, let us remember our first Bagavan and his gift of language that brought us closer together. Let's continue to spread love, kindness, and respect for all living things, just like he taught us.
"""

# text2speech(story) # convert generated text to audio

# Step 1: Install Necessary Libraries
#pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
#!pip install <torch> diffusers accelerate

# Step 2: Load the Pre-trained Model
# import torch
# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, device = torch.cpu, variant="fp16")
# pipe = pipe.to("cuda")

# # Step 3: Generate a Video
# prompt = "Penguine dancing happily"
# # Generate more frames by running the pipeline multiple times
# num_iterations = 4  # Number of times to run the pipeline for more frames
# all_frames = []

# for _ in range(num_iterations):
#     video_frames = pipe(prompt).frames[0]
#     all_frames.extend(video_frames)

# # Step 4: Export the Video
# from diffusers.utils import export_to_video

# video_path = export_to_video(all_frames)
# print(f"Video saved at: {video_path}")
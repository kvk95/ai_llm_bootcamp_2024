import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def print_to_console(message):
    print('----------------------------------------------------------------')
    print(message)
    print('----------------------------------------------------------------')

# Connect with LM
# region gemini
gemini_api_key = os.environ["gemini_api_key_vijay"]
gemini_api_key = os.environ["gemini_api_key"]
from langchain_google_genai import ChatGoogleGenerativeAI
def init_llm_gemini(temperature=0.9):
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", 
                             google_api_key=gemini_api_key, temperature=temperature)
# endregion gemini

# region llama3.1
# pip install -qU langchain-ollama  
from langchain_ollama import ChatOllama
def init_llm_ollama(temperature=0.9):
    llm = ChatOllama(
    model="llama3.1",
    temperature=temperature,
    # other params...
)
    
    return llm
# endregion llama3.1


def init_llm(temperature=0.9):
    return init_llm_gemini(temperature)
    return init_llm_ollama(temperature)

# Prompts
#A prompt is the input we provide to one language model. This input will guide the way the language model will respond.
# There are many types of prompts:
# * Plain instructions.
# * Instructions with a few examples (few-shot examples).
# * Specific context and questions appropiate for a given task.
# * Etc.

#LangChain provides a useful list of prompt recipes [here](https://smith.langchain.com/hub?ref=blog.langchain.dev).

## Prompt Templates
#Prompt templates are pre-defined prompt recipes that usually need some extra pieces to be complete. 
# These extra pieces are variables that the user will provide.

from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model="gemini-1.5-pro", 
                             google_api_key=gemini_api_key)

from langchain.prompts import PromptTemplate
my_template = """
Tell me a {adjective} joke about {topic}
"""

prompt_template = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=my_template
)

user_input = {
    "adjective": "funny",
    "topic": "French"
}

print(user_input["adjective"])

final_prompt = prompt_template.format(
    adjective=user_input["adjective"], 
    topic=user_input["topic"]
)

#TODO: uncomment
#resp = llm(final_prompt)
#print_to_console(resp)

## Chat Prompt Template

from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate

chat = init_llm()
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful assistant"
        ),
        HumanMessagePromptTemplate.from_template(
            "{user_input}"
        )
    ]
)
my_user_input = "How many hours have one year?"
resp = chat(chat_template.format_messages(user_input=my_user_input))
#TODO: uncomment
print_to_console(resp)

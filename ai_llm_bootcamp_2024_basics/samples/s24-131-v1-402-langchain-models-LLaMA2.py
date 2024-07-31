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
    #return init_llm_gemini(temperature)
    return init_llm_ollama(temperature)

# There are a few ways to access LLaMA2.
# To run locally, use Ollama.ai. See [here](https://python.langchain.com/docs/integrations/chat/ollama) for details on installation and setup.
# To use an external API, which is not private, you can use Replicate. You can register and get your REPLICATE_API_TOKEN [here](https://replicate.com/).

# Models
# Langchain provides interfaces and integrations for two types of language models:
# 1. LLMs: input a string of text, output a string of text.
# 2. Chat Models: input a chat message, output a chat message.

#Example of LLM: LLM model

my_llm = init_llm()
question = "Name the most popular 4 U.S. presidents"
response = my_llm.invoke(question)
#TODO: uncomment
#print_to_console(response)

#Example of Chat Model: OpenAI Chat Model
from langchain.schema import HumanMessage, SystemMessage
my_chat = init_llm()
chat_question = [
    SystemMessage(
        content="You are a helpful and concise assistant"
    ),
    HumanMessage(
        content="Name the most popular 4 U.S. First Ladies"
    )
]
chat_response = my_chat.invoke(chat_question)
#TODO: uncomment
print_to_console(chat_response)
#TODO: uncomment
print_to_console(chat_response.content)
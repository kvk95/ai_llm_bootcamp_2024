from enum import Enum

########################################################################
# region Logging

import coloredlogs

coloredlogs.install(level="DEBUG")
# endregion Logging

########################################################################
# region LOAD ENVIRONMENT VARIABLES

# > pip install pip install python-dotenv

from dotenv import load_dotenv, find_dotenv
import os

# Load the environment variables from the .env file
# find_dotenv() ensures the correct path to .env is used
dotenv_path = find_dotenv()
if dotenv_path == "":
    print("No .env file found.")
else:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)

# endregion LOAD ENVIRONMENT VARIABLES

embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
embeddings_model_path = os.getenv("EMBEDDINGS_MODEL_PATH")
chroma_db_path = os.getenv("CHROMA_DB_PATH")

########################################################################
# region Initialise foundation LLM

# > pip install langchain
# > pip install langchain_google_genai
# > pip install langchain_opena000i


# Define the enum
class LlmModel(Enum):
    GEMINI = "gemini-1.5-pro"
    LLAMA = "llama3.1"
    MISTRAL = "mistral"
    OPENAI = "gpt-4o"

from langchain_core.language_models.chat_models import BaseChatModel

# region google llm
from langchain_google_genai import ChatGoogleGenerativeAI

# The `GOOGLE_API_KEY`` environment variable set with your API key, or
# Pass your API key using the google_api_key kwarg to the ChatGoogle constructor.


# Get the value of a specific environment variable
def init_llm_gemini(modelname: str, temperature: float) -> BaseChatModel:
    
    google_api_key = os.environ["gemini_api_key_vijay"]
    google_api_key = os.environ["gemini_api_key"]
    # google_api_key = os.getenv("GOOGLE_API_KEY")

    return ChatGoogleGenerativeAI(
        model=modelname, google_api_key=google_api_key, temperature=0.3
    )


# endregion google llm

# region ollama llm
from langchain_community.llms.ollama import Ollama


def init_llm_ollama(modelname: str, temperature: float) -> BaseChatModel:
    return Ollama(model=modelname, num_gpu=1, temperature=temperature)


# endregion ollama llm

# region openai llm
from langchain_openai import ChatOpenAI


def init_llm_openai(modelname: str, temperature: float) -> BaseChatModel:
    return ChatOpenAI(model=modelname, temperature=temperature)


# endregion openai llm


def init_llm(llmmodel: LlmModel, temperature=0.3) -> BaseChatModel:
    match llmmodel:
        case LlmModel.GEMINI:
            return init_llm_gemini(LlmModel.GEMINI.value, temperature)
        case LlmModel.LLAMA:
            return init_llm_ollama(LlmModel.LLAMA.value, temperature)
        case LlmModel.MISTRAL:
            return init_llm_ollama(LlmModel.MISTRAL.value, temperature)
        case LlmModel.OPENAI:
            return init_llm_openai(LlmModel.OPENAI.value, temperature)
        case _:
            raise ValueError(f"Unsupported LlmModel: {llmmodel}")
    return None

# endregion Initialise foundation LLM

########################################################################

def print_to_console(*message: object):
    print('----------------------------------------------------------------')
    print(message)
    print('----------------------------------------------------------------')

# region add langchain logging

# > pip install langchain

import langchain

# langchain.debug = True
# langchain.verbose = True
# endregion add langchain logging

########################################################################
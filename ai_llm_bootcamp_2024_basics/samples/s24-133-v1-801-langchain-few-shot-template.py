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

## Few Shot Prompt Template

# Basic prompting strategies:
# * Zero Shot Prompt: "Classify the sentiment of this review: ..."
# * Few Shot Prompt: "Classify the sentiment of this review based on these examples: ..."
# * Chain Of Thought Prompt: "Classify the sentiment of this review based on these examples and explanations of the reasoning behind: ..."

## Zero Shot Prompting

from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
my_template = """
Classify the sentiment of this review.
Sentiment: Is the review positive, neutral or negative?

Review: {review}
"""
llm = GoogleGenerativeAI(model="gemini-1.5-pro", 
                             google_api_key=gemini_api_key)

my_prompt = PromptTemplate(
    template=my_template,
    input_variables=["review"]
)

my_prompt.format(review="I love this course!")
question = my_prompt.format(review="I love this course!")

#TODO: uncomment
#resp = llm(question)
#print_to_console(resp)

from langchain.prompts.few_shot import FewShotPromptTemplate

## Few Shot Prompting
#Option 1: in the template

my_few_shot_template_verbose = """
Classify the sentiment of this review.
Sentiment: Is the review positive, neutral or negative?

Review: {review}

Examples:
review: "I love this course!"
sentiment: positive

review: "What a waste of time!"
sentiment: negative

review: "So so. Not so good."
sentiment: neutral
"""

#Option2: out ot the template
my_examples = [
    {
        "review": "I love this course!",
        "response": "sentiment: positive" 
    },
    {
        "review": "What a waste of time!",
        "response": "sentiment: negative" 
    },
    {
        "review": "So so. Not so good.",
        "response": "sentiment: neutral" 
    },
]

my_few_shot_template_ok = """
Classify the sentiment of this review.
Sentiment: Is the review positive, neutral or negative?

Review: {review}
{response}
"""

example_prompt = PromptTemplate(
     input_variables=["review", "response"],
     template=my_few_shot_template_ok
 )

my_few_shot_prompt = FewShotPromptTemplate(
    examples=my_examples,
    example_prompt=example_prompt,
    suffix="Review: {review}",
    input_variables=["review"]
)

#TODO: uncomment
resp = llm(my_few_shot_prompt.format(review="What a piece of shit!"))
print_to_console(resp)
from comm_init import init_llm, LlmModel, print_to_console

## Chains
# Chains are sequences of operations. Usually, a chain combines:
# * a language model (LLM or Chat Model)
# * a prompt
# * other components

### Simple chain with LLMChain

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

chat = init_llm(llmmodel=LlmModel.MISTRAL)

prompt_template = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

llm_chain = prompt_template| chat

product = "AI Applications and AI Courses"

#llm_chain_response = llm_chain.invoke(product)
#TODO: uncomment
#print_to_console(llm_chain_response.content)

# The new way to do it with LangChain Expression Language

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

prompt_template = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

new_llm_chain = prompt_template | chat | StrOutputParser()

# new_llm_chain_response = new_llm_chain.invoke(
#     {"product": product}
# )
#TODO: uncomment
# print_to_console(new_llm_chain_response)

## Sequential Chains with SequentialChain
from langchain.chains import SequentialChain

# prompt template 1: translate to Spanish
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following text to spanish:"
    "\n\n{Text}"
)

# chain 1: input= Text and output= Spanish_Translation
chain_one = LLMChain(
    llm=chat, 
    prompt=first_prompt, 
    output_key="Spanish_Translation"
)

# prompt template 2: summarize
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following text in 1 sentence:"
    "\n\n{Text}"
)

# chain 2: input= Text and output= summary
chain_two = LLMChain(
    llm=chat, 
    prompt=second_prompt, 
    output_key="summary"
)

# prompt template 3: identify language
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following text:\n\n{Spanish_Translation}"
)

# chain 3: input= Spanish_Translation and output= language
chain_three = LLMChain(
    llm=chat, 
    prompt=third_prompt,
    output_key="language"
)

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(
    llm=chat, 
    prompt=fourth_prompt,
    output_key="followup_message"
)

# overall_chain: input= Text 
# and output= Spanish_Translation,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Text"],
    output_variables=["Spanish_Translation", "summary","followup_message"],
    verbose=True
)

review = """
Being in a start-up myself, I read this book to find comfort 
and confirmation about the volatility and emotional roller-coaster 
that comes with a working at the brink of software-as-a service. 
This book was just what it promised - interviews from some of the 
great successes in SV and elsewhere, their humble and sometimes 
difficult beginnings ('against all odds') that I am experiencing 
right now. It's not a literary piece of work - never intended to 
be, on the contrary, I felt the writing style was just what fits 
with a life in the fast lane - little time, easy to read chapters, 
inspiring and thank god, very 'down to earth.'

The one critical point I would like to make: I am somewhat 
perplexed how the companies were chosen - there are so many 
other companies that could have fit the profile which seem 
much more of a success than some of the companies/products in 
the book (gmail? Comm'on, I guess the author wanted to have 
Google in there, but didn't get an interview so she went with 
gmail?). Other great companies are easy to find - they don't
even need to be in the consumer space. How about Salesforce.com? 
I definitely liked the mix of 'new' and 'experienced' start ups. 

This book was a breeze to read and insightful for us start-up 
enterpreneurs.
"""
#overall_chain_response = overall_chain(review)
#TODO: uncomment
#print_to_console(overall_chain_response)

#The new way to do it with LangChain Expression Language
#Option 1: with the simpler syntax we get just the output.

from langchain.schema import StrOutputParser

new_sequence_chain = {"Spanish_Translation": first_prompt | chat | StrOutputParser()} | third_prompt | chat | StrOutputParser()

#overall_chain_response = new_sequence_chain.invoke({"Text": review})
#TODO: uncomment
#print_to_console(overall_chain_response)

#*Option 2: with RunnablePassthrough we get input and output.*
from langchain.schema.runnable import RunnablePassthrough

# input: Text, output: Spanish_Translation
first_chain = first_prompt | chat | StrOutputParser()

# input: Text, output: summary
second_chain = second_prompt | chat | StrOutputParser()

# input: Spanish_Translation, output: language
third_chain = third_prompt | chat | StrOutputParser()

# input: summary, output: followup_message
fourth_chain = fourth_prompt | chat | StrOutputParser()

one_plus_three_sequence_chain = {"Spanish_Translation": first_chain} | RunnablePassthrough.assign(language=third_chain)

#overall_chain_response = one_plus_three_sequence_chain.invoke({"Text": review})
#TODO: uncomment
#print_to_console(overall_chain_response)

## Router Chain

rock_template = """You are a very smart rock and roll professor. \
You are great at answering questions about rock and roll in a concise\
and easy to understand manner.

Here is a question:
{input}"""


politics_template = """You are a very good politics professor. \
You are great at answering politics questions..

Here is a question:
{input}"""


history_template = """You are a very good history teacher. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods.

Here is a question:
{input}"""


sports_template = """ You are a sports teacher.\
You are great at answering sports questions.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "rock", 
        "description": "Good for answering questions about rock and roll", 
        "prompt_template": rock_template
    },
    {
        "name": "politics", 
        "description": "Good for answering questions about politics", 
        "prompt_template": politics_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "sports", 
        "description": "Good for answering questions about sports", 
        "prompt_template": sports_template
    }
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=chat, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]

destinations_str = "\n".join(destinations)
#TODO: uncomment
print_to_console(destinations_str)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=chat, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.

REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(
    llm=chat, 
    prompt=router_prompt
)

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )

#multi_prompt_chain_response = chain.invoke("Who was Joe DiMaggio? Respond in less than 100 words")
#TODO: uncomment
#print_to_console(multi_prompt_chain_response)

#**How to do it with the new LangChain Expression Language**

from langchain.prompts import PromptTemplate

rock_template = """You are a very smart rock and roll professor. \
You are great at answering questions about rock and roll in a concise\
and easy to understand manner.

Here is a question:
{input}"""

rock_prompt = PromptTemplate.from_template(rock_template)

politics_template = """You are a very good politics professor. \
You are great at answering politics questions..

Here is a question:
{input}"""

politics_prompt = PromptTemplate.from_template(politics_template)

history_template = """You are a very good history teacher. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods.

Here is a question:
{input}"""

history_prompt = PromptTemplate.from_template(history_template)

sports_template = """ You are a sports teacher.\
You are great at answering sports questions.

Here is a question:
{input}"""

sports_prompt = PromptTemplate.from_template(sports_template)

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

general_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
)
prompt_branch = RunnableBranch(
  (lambda x: x["topic"] == "rock", rock_prompt),
  (lambda x: x["topic"] == "politics", politics_prompt),
  (lambda x: x["topic"] == "history", history_prompt),
  (lambda x: x["topic"] == "sports", sports_prompt),
  general_prompt
)

from typing import Literal

from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function


class TopicClassifier(BaseModel):
    "Classify the topic of the user question"
    
    topic: Literal["rock", "politics", "history", "sports"]
    "The topic of the user question. One of 'rock', 'politics', 'history', 'sports' or 'general'."


classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
llm = chat.bind(functions=[classifier_function], function_call={"name": "TopicClassifier"}) 
parser = PydanticOutputParser(pydantic_schema=TopicClassifier, attr_name="topic")
classifier_chain = llm | parser

from operator import itemgetter

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


final_chain = (
    RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain) 
    | prompt_branch 
    | init_llm(llmmodel=LlmModel.MISTRAL)
    | StrOutputParser()
)

final_chain_response = final_chain.invoke(
    {"input": "Who was Napoleon Bonaparte?"}
)


#TODO: uncomment
print_to_console(final_chain_response)
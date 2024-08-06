import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import logger

## Logging ##
# clear_terminal()

## Foundation Model ##
model: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

logger.info("")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Write one brief sentence about {politician}")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

# response = chain.invoke({"politician": "JFK"})
# logger.info(response)

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

vectorstore = DocArrayInMemorySearch.from_texts(
    [
        "AI Accelera has provided Generative AI Training and Consulting Services in more than 100 countries",
        "Aceleradora AI is the branch of AI Accelera for the Spanish-Speaking market",
    ],
    embedding=SentenceEmbeddingFunction(),
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

get_question_and_retrieve_relevant_docs = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = get_question_and_retrieve_relevant_docs | prompt | model | output_parser

# response = chain.invoke("In how many countries has AI Accelera provided services?")
# logger.info(response)

from utils.MyVectorStore import chroma_from_texts
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

coll_persist_name = "s32-222-200-main-ops-chains"
vectorstore = chroma_from_texts(
    [
        "AI Accelera has trained more than 7.000 Alumni from all continents and top companies"
    ],
    embedding=SentenceEmbeddingFunction(),
    persist_directory=coll_persist_name,
    collection_name=coll_persist_name,
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# response = retrieval_chain.invoke("who are the Alumni of AI Accelera?")
# logger.info(response)

from operator import itemgetter


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


coll_persist_name_1 = "s32-222-200-main-ops-chains_2"
vectorstore = chroma_from_texts(
    ["AI Accelera has trained more than 3,000 Enterprise Alumni."],
    embedding=SentenceEmbeddingFunction(),
    persist_directory=coll_persist_name_1,
    collection_name=coll_persist_name_1,
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# response = chain.invoke(
#     {
#         "question": "How many Enterprise Alumni has trained AI Accelera?",
#         "language": "Pirate English",
#     }
# )
# logger.info(response)

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    user_input=RunnablePassthrough(),
    transformed_output=lambda x: x["num"] + 1,
)

# response = runnable.invoke({"num": 1})
# logger.info(response)

## Chaining Runnables
# * Remember: almost any component in LangChain (prompts, models, output parsers, etc) can be used as a Runnable.
# * **Runnables can be chained together using the pipe operator `|`. The resulting chains of runnables are also runnables themselves**.

prompt = ChatPromptTemplate.from_template("tell me a sentence about {politician}")
chain = prompt | model | StrOutputParser()
# response = chain.invoke("Chamberlain")
# logger.info(response)

##### Coercion: combine a chain (which is a Runnable) with other Runnables to create a new chain.
# * See how in the `composed_chain` we are including the previous `chain`:

historian_prompt = ChatPromptTemplate.from_template(
    "Was {politician} positive for Humanity?"
)

composed_chain = {"politician": chain} | historian_prompt | model | StrOutputParser()
# response = composed_chain.invoke({"politician": "Lincoln"})
# logger.info(response)

# response = composed_chain.invoke({"politician": "Attila"})
# logger.info(response)

# * **Functions can also be included in Runnables**:
composed_chain_with_lambda = (
    chain
    | (lambda input: {"politician": input})
    | historian_prompt
    | model
    | StrOutputParser()
)
# response = composed_chain_with_lambda.invoke({"politician": "Robespierre"})
# logger.info(response)

## Multiple chains
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt1 = ChatPromptTemplate.from_template("what is the country {politician} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what continent is the country {country} in? respond in {language}"
)

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"country": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

# response = chain2.invoke({"politician": "Miterrand", "language": "French"})
# logger.info(response)

## Nested Chains
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel

prompt = ChatPromptTemplate.from_template(
    "tell me a curious fact about {soccer_player}"
)

output_parser = StrOutputParser()


def russian_lastname_from_dictionary(person):
    return person["name"] + "ovich"


chain = (
    RunnableParallel(
        {
            "soccer_player": RunnablePassthrough()
            | RunnableLambda(russian_lastname_from_dictionary),
            "operation_c": RunnablePassthrough(),
        }
    )
    | prompt
    | model
    | output_parser
)

# response = chain.invoke({"name1": "Jordam", "name": "Abram"})
# logger.info(response)

# ## Fallback for Chains
# * When working with language models, you may often encounter issues from the underlying APIs, whether these be rate limiting or downtime. Therefore, as you go to move your LLM applications into production it becomes more and more important to safeguard against these. That's why LangChain introduced the concept of fallbacks.
# * A fallback is an alternative plan that may be used in an emergency.
# * Fallbacks can be applied not only on the LLM level but on the whole runnable level. This is important because often times different models require different prompts. So if your call to OpenAI fails, you don't just want to send the same prompt to Anthropic - you probably want to use a different prompt template and send a different version there.
# * We can create fallbacks for LCEL chains. Here we do that with two different models: ChatOpenAI (with a bad model name to easily create a chain that will error) and then normal OpenAI (which does not use a chat model). Because OpenAI is NOT a chat model, you likely want a different prompt.

# First let's create a chain with a ChatModel
# We add in a string output parser here so the outputs between the two are the same type
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a funny assistant who always includes a joke in your response",
        ),
        ("human", "Who is the best {sport} player worldwide?"),
    ]
)
# Here we're going to use a bad model name to easily create a chain that will error
chat_model: BaseChatModel = init_llm(LlmModel.GROQ_LLAMA3, temperature=0)

bad_chain = chat_prompt | chat_model | StrOutputParser()

# Now lets create a chain with the normal OpenAI model
from langchain_core.prompts import PromptTemplate

prompt_template = """Instructions: You're a funny assistant who always includes a joke in your response.

Question: Who is the best {sport} player worldwide?"""

prompt = PromptTemplate.from_template(prompt_template)

llm: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

good_chain = prompt | llm

# We can now create a final chain which combines the two
chain = bad_chain.with_fallbacks([good_chain])

response = chain.invoke({"sport": "soccer"})
logger.info(response)

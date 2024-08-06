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
llmModel: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0)

logger.info("")

# response = llmModel.invoke("Tell me one fun fact about the Kennedy family.")
# logger.info(response)

#### Streaming: printing one chunk of text at a time

# for chunk in llmModel.stream("Tell me one fun fact about the Kennedy family."):
#     print(chunk, end="", flush=True)

# llmModel: BaseChatModel = init_llm(LlmModel.LLAMA, temperature=0.9)

# response = llmModel.invoke("Write a short 5 line poem about JFK")
# logger.info(response)

chatModel = init_llm(LlmModel.LLAMA, temperature=0)
messages = [
    ("system", "You are an historian expert in the Kennedy family."),
    ("human", "Tell me one curious thing about JFK."),
]
# response = chatModel.invoke(messages)
# logger.info(response)
# logger.info(response.content)
# logger.info(response.response_metadata)
# logger.info(response.schema())

#### Before the previous one, the old way (but still very popular) of doing this was:
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

messages = [
    SystemMessage(content="You are an historian expert on the Kennedy Family."),
    HumanMessage(content="How many children had Joseph P. Kennedy?"),
]

# response = chatModel.invoke(messages)
# logger.info(response)

# for chunk in chatModel.stream(messages):
#     print(chunk.content, end="", flush=True)

#### Another old way, similar results:
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are expert {profession} in {topic}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | chatModel

# response = chain.invoke(
#     {
#         "profession": "Historian",
#         "topic": "Kennedy Family",
#         "input": "Tell me one fun fact about JFK.",
#     }
# )

## Prompts
# * See the LangChain documentation about prompts [here](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/).
# * Input into LLMs.
# * Prompt templates: easier to use prompts with variables. A prompt template may include:
#     * instructions,
#     * few-shot examples,
#     * and specific context and questions appropriate for a given task.

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}."
)

llmModelPrompt = prompt_template.format(adjective="curious", topic="the Kennedy family")
# response = llmModel.invoke(llmModelPrompt)
# logger.info(response)

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an {profession} expert on {topic}."),
        ("human", "Hello, Mr. {profession}, can you please answer a question?"),
        ("ai", "Sure!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    profession="Historian",
    topic="The Kennedy family",
    user_input="How many grandchildren had Joseph P. Kennedy?",
)

# response = chatModel.invoke(messages)
# logger.info(response.content)

#### Old way:
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("You are an Historian expert on the Kennedy family.")),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

messages = chat_template.format_messages(
    user_input="Name the children and grandchildren of Joseph P. Kennedy?"
)

# response = chatModel.invoke(messages)
# logger.info(response.content)

#### What is the full potential of ChatPromptTemplate?
# * Check the [corresponding page]
# (https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
# in the LangChain API.

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
chain = final_prompt | chatModel
# response = chain.invoke({"input": "Who was JFK?"})
# logger.info(response.content)

## Parsing Outputs
# * See the corresponding LangChain Documentation page [here](https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/).
# * Language models output text. But many times you may want to get more structured information than just text back. This is where output parsers come in.

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()

json_chain = json_prompt | llmModel | json_parser

#### The previous prompt template includes the parser instructions
json_parser.get_format_instructions()
# response = json_chain.invoke({"question": "List the 3 biggest countries"})
# logger.info(response.content)

#### Optionally, you can use Pydantic to define a custom output format

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# Set up a parser
parser = JsonOutputParser(pydantic_object=Joke)

# Inject parser instructions into the prompt template.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a chain with the prompt and the parser
chain = prompt | chatModel | parser

# response = chain.invoke({"query": "Tell me a joke."})
# logger.info(response)

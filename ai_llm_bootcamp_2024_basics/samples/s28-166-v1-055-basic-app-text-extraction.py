from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic app to extract from a ChatMessage the song and artist a user wants to play
# **Define your extraction goal (called "the response schema")**

from langchain.output_parsers import ResponseSchema
response_schemas = [
    ResponseSchema(
        name="singer",
        description="name of the singer"
    ),
    ResponseSchema(
        name="song",
        description="name of the song"
    )
]

# **Create the Output Parser that will extract the data**
from langchain.output_parsers import StructuredOutputParser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# **Create the format instructions**
format_instructions = output_parser.get_format_instructions()
print_to_console(format_instructions)

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate

template = """
Given a command from the user,
extract the artist and song names
{format_instructions}
{user_prompt}
"""

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(template)
    ],
    input_variables={"user_prompt"},
    partial_variables={"format_instructions": format_instructions}
)

#**Enter the chat message from the user**
user_message = prompt.format_prompt(
    user_prompt="I like the song New York, New York by Frank Sinatra"
)

user_chat_message = llm.invoke(user_message.to_messages())
extraction = output_parser.parse(user_chat_message)
print_to_console(extraction)


from comm_init import init_llm, LlmModel, print_to_console

# Models
# Langchain provides interfaces and integrations for two types of language models:
# 1. LLMs: input a string of text, output a string of text.
# 2. Chat Models: input a chat message, output a chat message.

#Example of LLM: LLM model

my_llm = init_llm(LlmModel.GEMINI)
question = "Name the most popular 4 U.S. presidents"
#response = my_llm.invoke(question)
#TODO: uncomment
#print_to_console(response)

#Example of Chat Model: OpenAI Chat Model
from langchain.schema import HumanMessage, SystemMessage
my_chat = init_llm(LlmModel.GEMINI)
chat_question = [
    SystemMessage(
        content="You are a helpful and concise assistant"
    ),
    HumanMessage(
        content="Name the most popular 4 U.S. First Ladies"
    )
]
#chat_response = my_chat.invoke(chat_question)
#TODO: uncomment
#print_to_console(chat_response)
#TODO: uncomment
#print_to_console(chat_response.content)

question = """
*With alpha begins all alphabets; And the world with the first Bagavan.

with the above quotes, make a short traditional tamil storey for 6 year old, not exceeding 250 words*


use the the above prompt in chatgpt. story seems to be worth ful
"""
response = my_llm.invoke(question)
#TODO: uncomment
print_to_console(response.content)
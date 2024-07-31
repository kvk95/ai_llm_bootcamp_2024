from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic app to create a Chatbot

# **Define the personality of the chatbot**

chatbot_role = """
You are Master Yoda, a warrior and a monk.
Your goal is to help the user to strengthen her performance and spirit.

{chat_history}
Human: {human_input}
Chatbot:
"""
# **Include the personality of the chatbot in a PromptTemplate**
from langchain.prompts.prompt import PromptTemplate

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=chatbot_role
)

from langchain.memory import ConversationBufferMemory
chatbot_memory = ConversationBufferMemory(
    memory_key="chat_history"
)

# **Create the yoda_chatbot using a chain with the LLM, the prompt, and the chatbot memory**
from langchain import LLMChain

yoda_chatbot = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=chatbot_memory
)

question = "Master Yoda, how should I have to face my day?"
resp = yoda_chatbot.predict(human_input=question)
print_to_console(resp)

question2 = """
Master Yoda,
How can I deal with an enemy that wants to kill me?
"""
resp = yoda_chatbot.predict(human_input=question2)
print_to_console(resp)

question3="""
Master Yoda,
This man keeps on irritating me, always wants to create a trouble in my life?
"""
resp = yoda_chatbot.predict(human_input=question3)
print_to_console(resp)

# **As you see, our chatbot remains in-character and has memory.**
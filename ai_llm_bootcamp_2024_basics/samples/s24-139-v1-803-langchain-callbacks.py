from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

## Callbacks
# Calling functions in the middle of larger processes.

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["input"], 
    template="Tell me a joke about {input}")

chain = LLMChain(llm=llm, prompt=prompt_template)

#**Without callback**
# resp = chain.run(input="bear")
# #TODO: uncomment
# print_to_console(resp)

#**With callback**

handler = StdOutCallbackHandler()
# resp = chain.run(input="bear", callbacks=[handler])
# #TODO: uncomment
# print_to_console(resp)

## Customized Callback
from langchain.callbacks.base import BaseCallbackHandler
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        print(f"REPONSE: ", response)

resp = chain.run(input="whale", callbacks=[MyCustomHandler()])
# #TODO: uncomment
print_to_console(resp)

## Callback to check costs, token usage, etc
from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
resp = llm.invoke("(x-(x/100)) = 21600, what is x?")
# #TODO: uncomment
print_to_console(resp)
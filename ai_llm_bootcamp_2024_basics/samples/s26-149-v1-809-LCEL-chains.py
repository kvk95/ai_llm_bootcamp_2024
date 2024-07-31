from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

## Chains: Classic Way
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_template(
    "Who won the soccer world cup of {year}? Can you give the answer in tamil"
)

chain = LLMChain(llm=llm, prompt=prompt)
# resp = chain.predict(year="2010")
# print_to_console(resp)

## Chains: New LCEL Way
new_chain = prompt | llm
resp = new_chain.invoke({"year": "2010"})
print_to_console(resp)

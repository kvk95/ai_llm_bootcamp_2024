from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

# Classic Way
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me the name of 3 players who won the soccer world cup of {year}?"
)

# output_parser = StrOutputParser()
# chain = LLMChain(
#     llm=llm, 
#     prompt=prompt,
#     output_parser=output_parser
# )
# resp = chain.predict(year="2010")
# print_to_console(resp)

prompt = ChatPromptTemplate.from_template(
    "Tell me the name 3 players who won the soccer world cup of {year}?"
)

## Arguments: New LCEL Way
new_chain = prompt | llm.bind(stop=["\n3"]) | StrOutputParser()
resp = new_chain.invoke({"year": "2010"})
print_to_console(resp)
from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

## Output Parsers: Classic Way
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Who won the soccer world cup of {year}?"
)

output_parser = StrOutputParser()

chain = LLMChain(
    llm=llm, 
    prompt=prompt,
    output_parser=output_parser
)
resp = chain.predict(year="2010")
print_to_console(resp)

# Output Parsers: New LCEL Way
new_chain = prompt | llm | StrOutputParser()
resp = new_chain.invoke({"year": "2010"})
print_to_console(resp)
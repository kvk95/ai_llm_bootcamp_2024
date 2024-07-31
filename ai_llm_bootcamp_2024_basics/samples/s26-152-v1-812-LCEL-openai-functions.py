from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

from langchain.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_template(
    "Tell me the name of the captain of the team who won the soccer world cup of {year}?"
)

output_parser = JsonOutputParser()

functions = [
    {
      "name": "player_full_name",
      "description": "Name and lastname of the player",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The first name of the player"
          },
          "lastname": {
            "type": "string",
            "description": "The lastname of the player"
          }
        },
        "required": ["name", "lastname"]
      }
    }
  ]

model = OllamaFunctions(model=LlmModel.MISTRAL.value)
model.bind_tools(tools=functions)

# chain =   LLMChain(
#     llm=model, 
#     prompt=prompt, 
#     output_parser = output_parser,
# ) 
# resp = chain.predict(year="2010")
# print_to_console(resp)

## OpenAI Functions: New LCEL Way
new_chain = (
    prompt 
    | model
    | output_parser
)
resp = new_chain.invoke({"year": "2010"})
print_to_console(resp)
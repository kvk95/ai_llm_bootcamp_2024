from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic app with the Pydantic Output Parser
# * Previously we used the StructuredOutputParser to format the output into a JSON dictionary.
# * The StructuredOutputParser is a very simple parser that can only support strings and do not provide options for other data types such as lists or integers.
# * The PydanticOutput Parser is an advanced parser that admits many data types and other features like validators. 

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List 

# **Define the desired output data structure**
class Suggestions_Output_Structure(BaseModel):
    words: List[str] = Field(
        description="list of substitute words based on the context"
    )
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context"
    )

    #Throw error if the substitute word starts with a number
    @validator('words')
    def not_start_with_number(cls, info):
        for item in info:
            if item[0].isnumeric():
                raise ValueError("ERROR: The word cannot start with a number")
        return info

    @validator('reasons')
    def end_with_dot(cls, info):
      for idx, item in enumerate(info):
        if item[-1] != ".":
          info[idx] += "."
      return info
    
# **Create the parser**
my_parser = PydanticOutputParser(
    pydantic_object=Suggestions_Output_Structure
)

# **Determine the input**
from langchain.prompts import PromptTemplate 

my_template = """
Offer a list of suggestions to substitute the specified
target_word based on the present context and the reasoning
for each word.

{format_instructions}

target_word={target_word}
context={context}
"""

my_prompt = PromptTemplate(
    template=my_template,
    input_variables=["target_word", "context"],
    partial_variables={
        "format_instructions": my_parser.get_format_instructions()
    }
)

user_input = my_prompt.format_prompt(
    target_word="loyalty",
    context="""
    The loyalty of the soldier was so great that
    even under severe torture, he refused to betray
    his comrades.
    """
)

output = llm.invoke(user_input.to_string())
print_to_console(output.content)
resp = my_parser.parse(output.content)
print_to_console(resp)
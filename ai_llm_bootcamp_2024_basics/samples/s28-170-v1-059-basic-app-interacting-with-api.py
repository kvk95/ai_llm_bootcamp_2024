from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic app to interact with an API

# **Define the API documentation**
# <br>
# We are going to use a short version of the RestCountries API. 
# <br>
# <br>
# *Important note: our app will only work with the endpoints we define below, althougt the original API 
# has many more endpoints.*

api_docs = """
BASE URL: https://restcountries.com/

API Documentation:

The API endpoint /v3.1/name/{name} Used to find informatin about 
a country. All URL parameters are listed below:
    - name: Name of country - Example: Italy, France
    
The API endpoint /v3.1/currency/{currency} Used to find information 
about a region. All URL parameters are listed below:
    - currency: 3 letter currency. Example: USD, COP

The API endpoint /v3.1/lang/{language} Used to find information 
about the official language of the country. All URL parameters 
are listed below:
    - language: language of the country. Example: English, Spanish
    
"""

# **Create a chain to read the API documentation**
from langchain.chains import APIChain
api_chain = APIChain.from_llm_and_api_docs(
    llm=llm,
    api_docs=api_docs,
    verbose=True,
    limit_to_domains=["https://restcountries.com/"]
)

#**Ask a question about the API**
question = "Give me information about France in less than 100 words."
resp = api_chain.invoke(question)
print_to_console(resp)

question2 = """
List the top 3 biggest countries 
where the official language is French.
"""
resp = api_chain.invoke(question2)
print_to_console(resp)
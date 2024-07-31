from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
sqlite_db_path = "data/street_tree_db.sqlite"
db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

#**Create a chain with de LLM and the database**
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True
)
resp = db_chain.invoke("How many species of trees are in San Francisco?")
print_to_console(resp)

resp = db_chain.run("How many trees of the species Ficus nitida are there in San Francisco?")
print_to_console(resp)

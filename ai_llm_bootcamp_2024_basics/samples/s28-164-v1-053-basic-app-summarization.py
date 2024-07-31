from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic Use Case: Summarize a Text File

with open('data/be-good-and-how-not-to-die.txt', 'r') as file:
    article = file.read()

print_to_console(article[:285])

num_tokens = llm.get_num_tokens(article)
print_to_console(f"There are {num_tokens} in the article.")

# **Split the article in smaller chunks**
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], 
    chunk_size=5000,
    chunk_overlap=350
)
article_chunks = text_splitter.create_documents([article])
print_to_console(f"You have {len(article_chunks)} chunks instead of 1 article.")

# **Use a chain to help the LLM to summarize the 8 chunks**
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce"
)
article_summary = chain.invoke(article_chunks)
print_to_console(article_summary)
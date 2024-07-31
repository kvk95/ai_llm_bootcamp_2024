from comm_init import init_llm, LlmModel, print_to_console

llm = init_llm(llmmodel=LlmModel.MISTRAL)
product = "AI Applications and AI Courses"

#llm_chain_response = llm.invoke(product)
#TODO: uncomment
#print_to_console(llm_chain_response)

# Memory

# * Language models have a short memory. They can only remember the information that is in their context window.
# * Currently, the best way to solve this problem is to use the RAG technique (use an external vector database as memory). We will see this in next chapters.
# * Apart from the RAG technique, LangChain provides a few alternatives to improve the memory of the language models.

# Conversation Buffer Memory
# Stores messages and it loads them in the next prompt.


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

buffer_memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = buffer_memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Julio and I have moved 33 times.")

conversation.predict(input="Sure. If my average moving distance was 100 miles, how many miles took all my moves?")

conversation.predict(input="Do you remember my name?")

print(buffer_memory.buffer)
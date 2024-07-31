from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## LangServe
## Here's a server that deploys an OpenAI chat model and a chain that uses the OpenAI chat model to tell a joke about a topic.
#!pip install langserve
#!pip install sse_starlette

#!/usr/bin/env python
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama
from threading import Thread
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)
#llm = ChatOllama(model=LlmModel.MISTRAL.value)
add_routes(
    app,
    ChatOllama(model=LlmModel.MISTRAL.value),
    path="/ollama",
)

model = llm

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

add_routes(
    app,
    prompt | model,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    @app.get("/")
    async def read_root():
        return {"Hello": "World"}

    # Function to run the Uvicorn server in a thread
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

    # Start the server in a separate thread
    thread = Thread(target=run_server)
    thread.start()
from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.LLAMA)
# #TODO: uncomment
# print_to_console(resp)

## Basic app: a very simple Agent that will decide which external plugin to use
from langchain.utilities import TextRequestsWrapper
requests = TextRequestsWrapper()
# **Set the tools that will use our agent**
from langchain.agents import Tool
from langchain.tools import BaseTool
from playwright.sync_api import sync_playwright

class web_search_tool(BaseTool):
    name = "web_search"
    description = "To search the web"

    def _run(self, query: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://www.google.com/search?q={query}")
            results = page.query_selector_all('h3')
            titles = [result.inner_text() for result in results]
            browser.close()
            return f"Top results for '{query}':\n" + "\n".join(titles[:5])
        
tools = [
    Tool(        
        name="web_search",
        func=web_search_tool(),
        description="""
        useful when you need to To search the web
        """
    ),
    Tool(
        name="request",
        func=requests.get,
        description="""
        useful when you need to make a request to a URL
        """
    )
]

# **Agent initialization and configuration**
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
my_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3
)

# **Ask questions to the agent and see how she decides which tool to use**
question = "Which team won the 2010 soccer world chapionship?"
response = my_agent(question)
print_to_console(response)

question2 = {
    "input": "Tell me about what subject are the comments in this webpage https://www.amazon.in/Refurbished-HP-Touchscreen-Chromebook-Bluetooth/dp/B0CWZSJV9W?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A3V0Z9Z500NB0X"
}
response = my_agent(question2)
print_to_console(response)
from comm_init import init_llm, print_to_console

my_creative_llm = init_llm()

question = "What are the 5 best things to do in life?"

#TODO: uncomment
#print_to_console(my_creative_llm.invoke(question))

# Create a prompt template to wrap the user input
from langchain_core.prompts import PromptTemplate
my_prompt_template = PromptTemplate(
    input_variables = ["destination"],
    template = "What are the 3 best things to do in {destination}?"
)
user_input = "Barcelona"
#TODO: uncomment
#print_to_console(my_creative_llm.invoke(my_prompt_template.format(destination=user_input)))

# Combine instructions in chains
from langchain.chains import LLMChain

my_no_creative_llm = init_llm(temperature=0)
my_prompt_template = PromptTemplate(
    input_variables = ["politician"],
    template = "What are the 3 most remarkable achievements of {politician}?",
)

my_chain = my_no_creative_llm | my_prompt_template
#TODO: uncomment
#print_to_console(my_chain.invoke("Churchill"))

### Agents: give reasoning power to LMs
# 1. Load some external tools (google search, llm-math, etc)
# 2. Initialize an agent type on top of the LM
# 3. Run the agent and she will decide the tools to use based on your input
#https://python.langchain.com/v0.1/docs/integrations/tools/

#> pip install playwright
#> playwright install

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from playwright.sync_api import sync_playwright
from langchain.tools import BaseTool

class web_search_tool(BaseTool):
    name = "web_search"
    description = "To search web"

    def _run(self, query: str):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://www.google.com/search?q={query}")
            results = page.query_selector_all('h3')
            titles = [result.inner_text() for result in results]
            browser.close()
            return f"Top results for '{query}':\n" + "\n".join(titles[:5])

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")
    
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


class PrimeLessThan(BaseTool):
    name = "prime_less_than"
    description = "Returns the largest prime number less than the given number, or None if no such prime exists"

    def _run(self, nmr: int) -> int:
        """
        Returns the largest prime number that is smaller than the given number.

        :param nmr: The number to find the largest prime number less than.
        :return: The largest prime number less than the given number, or None if no such prime exists.
        """
        nmr = int(nmr)

        # If the given number is less than or equal to 2, return None since there are no prime numbers less than 2.
        if nmr <= 2:
            return None

        def is_prime(num):
            """
            Checks if a number is prime.

            :param num: The number to check.
            :return: True if the number is prime, False otherwise.
            """
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True

        # Iterate from the number just less than nmr down to 2 to find the largest prime number.
        for i in range(nmr - 1, 1, -1):
            if is_prime(i):
                return i

        return None

    def _arun(self, nmr: int):
        """
        Raises an error since async operation is not supported for this tool.

        :param nmr: Placeholder parameter.
        """
        raise NotImplementedError("This tool does not support async")


llm = init_llm()
tools = load_tools ([], llm=llm)
tools.insert(0,web_search_tool())
tools.insert(1,PrimeLessThan())
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# agent_response = agent.run("Which year india got independence? What is the largest prime number that is smaller than that?")
#TODO: uncomment
#print_to_console(agent_response)

from langchain import ConversationChain
my_conversation = ConversationChain(llm=my_no_creative_llm, verbose=True)
#conv_resp = my_conversation.invoke("hi there!")
#TODO: uncomment
#print_to_console(conv_resp)
#conv_resp = my_conversation.invoke("My name is Julio. What is your name?")
#TODO: uncomment
#print_to_console(conv_resp)
#conv_resp = my_conversation.invoke("Can i call you as bruno?")
#TODO: uncomment
#print_to_console(conv_resp)

#TODO: uncomment
#print_to_console(my_conversation)
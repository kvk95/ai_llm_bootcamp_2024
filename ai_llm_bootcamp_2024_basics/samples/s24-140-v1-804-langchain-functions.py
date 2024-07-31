from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

# Describe the functions
functions = [
    {
        "name": "get_item_info",
        "description": "Get name and price of a menu item of the chinese restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the menu item, e.g. Chop-Suey",
                },
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "place_order",
        "description": "Place an order for a menu item from the restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item you want to order, e.g. Chop-Suey",
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of items you want to order",
                    "minimum": 1
                },
                "address": {
                    "type": "string",
                    "description": "The address where the food should be delivered",
                },
            },
            "required": ["item_name", "quantity", "address"],
        },
    }
]

# Option 1: The LangChain Solution

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.callbacks import StdOutCallbackHandler

model = OllamaFunctions(model=LlmModel.MISTRAL.value)
model.bind_tools(tools=functions)
from langchain.prompts import PromptTemplate

template = """You are an AI chatbot having a conversation 
with a human.

Human: {human_input}
AI: """

prompt = PromptTemplate(
    input_variables=["human_input"], 
    template=template
)

chain = prompt | model
# resp = chain.invoke("How much does Chop-Suey cost?")
# print_to_console(resp)
# resp = chain.invoke("Who is the current Pope?") # Not-so-good error message.
# print_to_console(resp)
# resp = chain.invoke("I want to order two Chop-Suey to 321 Street")
# print_to_console(resp)

# Let's use a dictionary as a fake database to check full functionality
fake_db = {
    "items": {
        "Chop-Suey": {"price": 15.00, "ingredients": ["chop", "suey", "cheese"]},
        "Lo-Main": {"price": 10.00, "ingredients": ["lo", "main", "basil"]},
        "Chin-Gun": {"price": 12.50, "ingredients": ["chin", "gunu", "tomato sauce"]},
        "Won-Ton": {"price": 11.00, "ingredients": ["won", "ton", "mushrooms"]},
    },
    "orders": []
}

def get_item_info(item_name):
    item = fake_db["items"].get(item_name)
    
    if not item:
        return f"No information available for item: {item_name}"

    return {"name": item_name, "price": item["price"], "ingredients": item["ingredients"]}

def place_order(item_name, quantity, address):
    if item_name not in fake_db["items"]:
        return f"We don't have {item_name}!"
    
    # if quantity < 1:
    #     return "You must order at least one item."
    
    order_id = len(fake_db["orders"]) + 1
    order = {
        "order_id": order_id,
        "item_name": item_name,
        "quantity": quantity,
        "address": address,
        "total_price": fake_db["items"][item_name]["price"] * quantity
    }

    fake_db["orders"].append(order)
    
    return f"Order placed successfully! Your order ID is {order_id}. Total price is ${order['total_price']}."

def process_query(query):

    result = chain.invoke(
        {"human_input": query}, config={"callbacks": [StdOutCallbackHandler()]}
    )

    print_to_console(result)

    if result.tool_calls:
        for tool_call in result.tool_calls:
            function_name = tool_call["name"]
            args = tool_call["args"]
            print_to_console(f"Function call: {function_name}, Args: {args}")

            if function_name == "get_item_info":
                return get_item_info(**args)
            elif function_name == "place_order":
                return place_order(**args)

    return result.content


# result = process_query("I want to order two Chop-Suey to 321 Street")
# print_to_console(result)

# Let´s improve the app making it dynamic (the LLM can continue the conversation after the order has been placed)

import json
from openai import OpenAI

#https://ollama.com/blog/openai-compatibility
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

class ChatBot:
    
    def __init__(self, database):
        self.fake_db = database
        self.model = LlmModel.MISTRAL.value
        
    def chat(self, query):
        initial_response = self.make_openai_request(query)
        
        message = initial_response.choices[0].message
        
        if (hasattr(message, 'function_call') & (message.function_call != None)):
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)
            function_response = getattr(self, function_name)(**arguments)
            
            follow_up_response = self.make_follow_up_request(query, message, function_name, function_response)
            return follow_up_response.choices[0].message.content
        else:
            return message.content
    
    def make_openai_request(self, query):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            functions=functions
        )
        return response

    def make_follow_up_request(self, query, initial_message, function_name, function_response):
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": query},
                initial_message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return response

    def place_order(self, item_name, quantity, address):
        if item_name not in self.fake_db["items"]:
            return f"We don't have {item_name}!"
        
        if quantity < 1:
            return "You must order at least one item."
        
        order_id = len(self.fake_db["orders"]) + 1
        order = {
            "order_id": order_id,
            "item_name": item_name,
            "quantity": quantity,
            "address": address,
            "total_price": self.fake_db["items"][item_name]["price"] * quantity
        }

        self.fake_db["orders"].append(order)
        
        return f"Order placed successfully! Your order ID is {order_id}. Total price is ${order['total_price']}."

    def get_item_info(self, item_name):
        if item_name in self.fake_db["items"]:
            item = self.fake_db["items"][item_name]
            return f"Item: {item['name']}, Price: ${item['price']}"
        else:
            return f"We don't have information about {item_name}."

database = {
    "items": {
        "Chop-Suey": {
            "name": "Chop-Suey",
            "price": 15.0
        },
        "Lo-Mein": {
            "name": "Lo-Mein",
            "price": 12.0
        }
    },
    "orders": []
}

### Let's check the app

bot = ChatBot(database=database)
response = bot.chat("I want to order two Chop-Suey to 321 Street")
print_to_console(response)

## Let's ask for a type of food that is not in the menu
response = bot.chat("I want to order one spring roll to 321 Street")
print_to_console(response)

## Let's ask an off-topic question
response= bot.chat("Who is the current Pope?")
print_to_console(response)

## As you see, the app does not break now.
import google.generativeai as genai
from langchain.chat_models import init_chat_model
import sys
import os
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# async def GetStockPrices(
#         company_name: str
#     ) -> str:
#         """
#         Args : company_name: str
     
#         Returns : str
#         For the company name provided, returns the current stock price.
#         """
#         # Here you could add more complex validation if needed
#         if not company_name or not isinstance(company_name, str):
#             return "Invalid company name provided."
#         # Simulate fetching stock price (replace with actual API call if needed)
#         stock_price = "123.45"  # Placeholder for actual stock price retrieval logic
#         return f"The current stock price of {company_name} is ${stock_price}."  
# agent_tools = [GetStockPrices]

class AgentClass:
    def __init__(self):
        self.name = "Gemini Agent"
        self.description = "An agent that uses Gemini API to answer questions."
        self.api_key = GEMINI_API_KEY
        self.llm = init_chat_model(
                "gemini-2.0-flash",
                model_provider="google_genai",
                api_key=self.api_key,
                temperature=0.1
            ) 
    def chat(self, prompt: str) -> str:
        response = self.llm.invoke(
            input=prompt,
            # tools=agent_tools,

        )
        return response    
@tool
async def GetStockPrices(company_name: str) -> str:
    """For the company name provided, returns the current stock price."""
    if not company_name or not isinstance(company_name, str):
        return "Invalid company name provided."
    stock_price = "123.45"
    print("HAHAHAHAAH")
    return f"The current stock price of {company_name} is ${stock_price}."

# 1. Initialize the LangChain model wrapper
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key = GEMINI_API_KEY, temperature=0.1)

# 2. Bind the LangChain tools to the model
model_with_tools = llm.bind_tools([GetStockPrices])

# 3. Invoke the model using the LangChain interface
if __name__ == "__main__":
    response = model_with_tools.invoke("List the tools available to you and then answer What is the current stock price of Google?")
    # The response object is different in LangChain
    # The tool call and its output will be in the message history
    print(response)

    # if response.tool_calls:
    #     for tool_call in response.tool_calls:
    #         print(f"Tool called: {tool_call.tool_name} with arguments {tool_call.arguments}")
    #         tool_response = GetStockPrices(tool_call.arguments['company_name'])
    #         print(f"Tool response: {tool_response}")
    #         follow_up = chat.send_message(tool_response)
    #         print(f"Follow-up response: {follow_up.text}")
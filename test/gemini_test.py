from langchain.chat_models import init_chat_model
import sys
import os
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
            tools=[GenAITool(google_search={})],

        )
        return response      


if __name__ == "__main__":
    agent = AgentClass()
    print(agent.chat("When is the next total solar eclipse in US? Use google search to find the answer.").content)
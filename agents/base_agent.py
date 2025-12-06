# from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage

'''
Simple Agent wrapper to abstract between OpenAI and Google generative chat models.
The Agent class below chooses the appropriate langchain chat model based on the
'model' argument ('gpt' for OpenAI, anything else -> Google GenAI).

It expects an API key and a model name, constructs the LLM client, and exposes
a generate() method that formats a prompt with placeholders and returns the
model's response content.

Keep this file minimal: imports at the top, configuration here, and the Agent
class implementation following these comments.
'''


class Agent:
    def __init__(self, api_key, model = 'gpt',model_name="gpt-4.1-nano"):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key) if model == 'gpt' else ChatGoogleGenerativeAI(model=self.model_name, api_key=self.api_key)

    def generate(self, prompt: str, prompt_placeholders: dict) -> str:
        formatted_prompt = prompt.format(**prompt_placeholders)
        response = self.llm([HumanMessage(content=formatted_prompt)])
        return response.content
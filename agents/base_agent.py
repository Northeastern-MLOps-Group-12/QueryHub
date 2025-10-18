# from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage

class Agent:
    def __init__(self, api_key, model = 'gpt',model_name="gpt-4.1-nano"):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key) if model == 'gpt' else ChatGoogleGenerativeAI(model=self.model_name, api_key=self.api_key)

    def generate(self, prompt: str, prompt_placeholders: dict) -> str:
        formatted_prompt = prompt.format(**prompt_placeholders)
        response = self.llm([HumanMessage(content=formatted_prompt)])
        return response.content
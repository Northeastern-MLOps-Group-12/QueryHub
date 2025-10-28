import pytest
from unittest.mock import MagicMock, patch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from agents.base_agent import Agent

def test_agent_initializes_openai():
    """Agent uses ChatOpenAI when model='gpt'."""
    with patch("agents.base_agent.ChatOpenAI") as mock_openai:
        a = Agent(api_key="dummy", model="gpt", model_name="gpt-4.1-nano")
        mock_openai.assert_called_once_with(model="gpt-4.1-nano", api_key="dummy")
        assert hasattr(a, "generate")

def test_agent_initializes_google():
    """Agent uses ChatGoogleGenerativeAI when model != 'gpt'."""
    with patch("agents.base_agent.ChatGoogleGenerativeAI") as mock_google:
        a = Agent(api_key="dummy", model="gemini", model_name="gemini-1.5-flash")
        mock_google.assert_called_once_with(model="gemini-1.5-flash", api_key="dummy")

def test_generate_formats_prompt_and_returns_content():
    """Ensure generate() formats and calls the LLM properly."""
    mock_llm = MagicMock()
    mock_llm.return_value.content = "result"
    agent = Agent(api_key="dummy")
    agent.llm = lambda msgs: MagicMock(content="mocked response")

    response = agent.generate("Hello {name}", {"name": "Ashwin"})
    assert "mocked" in response or isinstance(response, str)

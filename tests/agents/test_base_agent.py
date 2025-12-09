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
        mock_openai.assert_called_once_with(model="gpt-4.1-nano", openai_api_key="dummy")
        assert hasattr(a, "generate")

def test_agent_initializes_google():
    """Agent uses ChatGoogleGenerativeAI when model != 'gpt'."""
    with patch("agents.base_agent.ChatGoogleGenerativeAI") as mock_google:
        a = Agent(api_key="dummy", model="gemini", model_name="gemini-1.5-flash")
        # Fixed: ChatGoogleGenerativeAI uses google_api_key, not openai_api_key
        mock_google.assert_called_once_with(model="gemini-1.5-flash", google_api_key="dummy")

def test_generate_formats_prompt_and_returns_content():
    """Ensure generate() formats and calls the LLM properly."""
    # Create a proper mock with stream method
    mock_llm = MagicMock()
    
    # Mock the stream method to return chunks with content
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "mocked"
    mock_chunk2 = MagicMock()
    mock_chunk2.content = " response"
    
    mock_llm.stream.return_value = iter([mock_chunk1, mock_chunk2])
    
    agent = Agent(api_key="dummy")
    agent.llm = mock_llm

    response = agent.generate("Hello {name}", {"name": "Ashwin"})
    
    # Verify the stream method was called
    mock_llm.stream.assert_called_once()
    
    # Verify response contains expected content
    assert "mocked" in response
    assert isinstance(response, str)
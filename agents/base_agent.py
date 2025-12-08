"""
Base Agent for LLM interactions - FIXED API Key Handling
Supports: ChatOpenAI, ChatGoogleGenerativeAI
Tracks: TTFT, Total time, Tokens generated, Tokens/sec, Avg time per token
"""

import os
import time
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage

# Try to import monitoring, but don't crash if not available
try:
    from backend.monitoring import track_llm_generation
    MONITORING_ENABLED = True
except ImportError:
    print("⚠️ Monitoring module not available - running without metrics")
    MONITORING_ENABLED = False
    def track_llm_generation(*args, **kwargs):
        pass  # No-op function


class Agent:
    """
    LLM Agent wrapper with comprehensive token-level monitoring.
    Uses LangChain chat models for both OpenAI and Google Gemini.
    """
    
    def __init__(self, api_key: str = None, model: str = 'gemini', model_name: str = None):
        """
        Initialize the agent with API credentials and model configuration.
        
        Args:
            api_key: (Optional) API key - if not provided, reads from env
            model: Provider type ('gpt'/'openai' or 'gemini'/'google')
            model_name: Specific model name - if not provided, reads from env
        """
        # ✅ FIX 1: Normalize model name
        model_lower = model.lower()
        
        # ✅ FIX 2: Get model name from env if not provided
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        
        self.model = model_lower
        self.model_name = model_name
        
        # ✅ FIX 3: Get correct API key based on provider
        if api_key:
            # Use provided key
            self.api_key = api_key
        else:
            # Read from environment based on provider
            if model_lower in ['gpt', 'openai']:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "❌ OPENAI_API_KEY not set in .env but MODEL=gpt!\n"
                        "Either:\n"
                        "  1. Set OPENAI_API_KEY in .env\n"
                        "  2. Change MODEL=gemini in .env"
                    )
            else:
                self.api_key = os.getenv("LLM_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "❌ LLM_API_KEY (Gemini) not set in .env!\n"
                        "Set LLM_API_KEY=<your-gemini-key> in .env"
                    )
        
        # ✅ FIX 4: Initialize correct LangChain chat model
        if model_lower in ['gpt', 'openai']:
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key
            )
            self.provider = "openai"
            print(f"✅ Initialized ChatOpenAI")
            print(f"   Model: {model_name}")
            print(f"   API Key: {self.api_key[:20]}...")
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key
            )
            self.provider = "gemini"
            print(f"✅ Initialized ChatGoogleGenerativeAI")
            print(f"   Model: {model_name}")
            print(f"   API Key: {self.api_key[:20]}...")
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to default tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        Uses tiktoken for accurate counting.
        
        Args:
            text: Input text to count tokens
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def generate(self, prompt: str, prompt_placeholders: dict, operation: str = "generation") -> str:
        """
        Generate text using LLM with comprehensive monitoring.
        
        Tracks:
        - Time to first token (TTFT)
        - Total generation time
        - Tokens generated
        - Tokens per second
        - Average time per token
        
        Args:
            prompt: The prompt template with {placeholders}
            prompt_placeholders: Dictionary of values to fill in the prompt
            operation: Name of the operation (for metrics labeling)
            
        Returns:
            Generated text from the LLM
        """
        # Format the prompt with placeholders
        formatted_prompt = prompt.format(**prompt_placeholders)
        
        # Track timing
        start_time = time.time()
        first_token_time = None
        response_text = ""
        
        try:
            # ================================================================
            # STREAMING GENERATION WITH TTFT TRACKING
            # ================================================================
            
            # LangChain streaming - works for both OpenAI and Gemini
            for chunk in self.llm.stream([HumanMessage(content=formatted_prompt)]):
                # Mark time to first token
                if first_token_time is None and chunk.content:
                    first_token_time = time.time()
                
                # Accumulate response
                response_text += chunk.content
            
            # If no tokens were generated, mark first token time as now
            if first_token_time is None:
                first_token_time = time.time()
            
            # ================================================================
            # CALCULATE METRICS
            # ================================================================
            
            end_time = time.time()
            total_time = end_time - start_time
            time_to_first_token = first_token_time - start_time if first_token_time else 0
            
            # Count tokens in response
            total_tokens = self.count_tokens(response_text)
            
            # Track metrics in Prometheus (if monitoring is enabled)
            if MONITORING_ENABLED:
                track_llm_generation(
                    operation=operation,
                    model=self.model_name,
                    time_to_first_token=time_to_first_token,
                    total_time=total_time,
                    total_tokens=total_tokens
                )
            
            # Print metrics for debugging
            print(f"📊 LLM Metrics ({operation}):")
            print(f"   Provider: {self.provider.upper()}")
            print(f"   Model: {self.model_name}")
            print(f"   Time to First Token: {time_to_first_token:.3f}s")
            print(f"   Total Time: {total_time:.3f}s")
            print(f"   Total Tokens: {total_tokens}")
            if total_time > 0:
                print(f"   Tokens/Second: {total_tokens/total_time:.2f}")
            else:
                print(f"   Tokens/Second: N/A")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error in LLM generation ({self.provider}): {e}")
            
            # Still track the error with zero tokens (if monitoring enabled)
            if MONITORING_ENABLED:
                end_time = time.time()
                track_llm_generation(
                    operation=operation,
                    model=self.model_name,
                    time_to_first_token=0,
                    total_time=end_time - start_time,
                    total_tokens=0
                )
            raise
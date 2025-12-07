"""
Base Agent for LLM interactions - Using LangChain Chat Models
Supports: ChatOpenAI, ChatGoogleGenerativeAI
Tracks: TTFT, Total time, Tokens generated, Tokens/sec, Avg time per token
"""

import time
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from backend.monitoring import track_llm_generation


class Agent:
    """
    LLM Agent wrapper with comprehensive token-level monitoring.
    Uses LangChain chat models for both OpenAI and Google Gemini.
    """
    
    def __init__(self, api_key: str, model: str = 'gpt', model_name: str = "gpt-4o-mini"):
        """
        Initialize the agent with API credentials and model configuration.
        
        Args:
            api_key: API key for the LLM provider
            model: Provider type ('gpt' for OpenAI, 'gemini' for Google)
            model_name: Specific model name (e.g., 'gpt-4', 'gemini-2.5-flash')
        """
        self.model = model
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize LangChain chat model
        if model == 'gpt':
            self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key)
            self.provider = "openai"
            print(f"✅ Initialized ChatOpenAI with model: {model_name}")
        else:
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, api_key=self.api_key)
            self.provider = "gemini"
            print(f"✅ Initialized ChatGoogleGenerativeAI with model: {model_name}")
        
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
            
            # Track metrics in Prometheus
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
            
            # Still track the error with zero tokens
            end_time = time.time()
            track_llm_generation(
                operation=operation,
                model=self.model_name,
                time_to_first_token=0,
                total_time=end_time - start_time,
                total_tokens=0
            )
            raise
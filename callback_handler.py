# berisi fungsi untuk menghitung token dan biaya penggunaan LLM Gemini.

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List
import tiktoken
import streamlit as st

class GeminiCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.total_tokens = 0
        self.total_cost = 0
        self.input_price_per_million_tokens = 0.35 
        self.output_price_per_million_tokens = 1.05
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            for prompt in prompts:
                tokens = len(encoding.encode(prompt))
                self.total_tokens += tokens
                self.total_cost += (tokens / 1_000_000) * self.input_price_per_million_tokens
        except Exception:
            pass
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            for generation_list in response.generations:
                for generation in generation_list:
                    tokens = len(encoding.encode(generation.text))
                    self.total_tokens += tokens
                    self.total_cost += (tokens / 1_000_000) * self.output_price_per_million_tokens
        except Exception:
            pass
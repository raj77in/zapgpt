"""
ZapGPT - A command-line tool for interacting with various LLM providers

This package provides both CLI and programmatic interfaces for querying
multiple LLM providers including OpenAI, OpenRouter, Together, and more.

CLI Usage:
    zapgpt "Your question here"
    zapgpt --provider openrouter "Your question"
    zapgpt --use-prompt coding "Write a function"
    zapgpt --quiet "Your question"  # Suppress all output except response

Programmatic Usage:
    from zapgpt import query_llm
    response = query_llm("What is Python?", provider="openai")

    # With custom prompt
    response = query_llm(
        "Debug this code",
        provider="openai",
        use_prompt="coding",
        model="gpt-4o"
    )
"""

__version__ = "3.0.0"
__author__ = "Amit Agarwal"
__email__ = "amit@example.com"

# Import main functionality for programmatic use
try:
    from .main import main, query_llm

    __all__ = ["query_llm", "main"]
except ImportError:
    # Handle case where dependencies might not be available
    __all__ = ["main"]

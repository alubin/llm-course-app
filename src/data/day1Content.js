export const day1Content = {
  id: 1,
  title: "LLM Fundamentals + Building Your First AI CLI Assistant",
  subtitle: "Learn how Large Language Models work and build a production-ready CLI tool",
  duration: "4-8 hours",
  difficulty: "Beginner to Intermediate",
  
  objectives: [
    "Explain how Large Language Models work at a conceptual level",
    "Understand tokens, context windows, and temperature parameters",
    "Set up and use the OpenAI and Anthropic Python SDKs",
    "Build a production-quality CLI application with Click",
    "Implement streaming responses for better UX",
    "Handle errors gracefully and manage API costs",
    "Structure a Python project for GitHub publication"
  ],

  prerequisites: [
    { name: "Python", details: "Version 3.9+ installed" },
    { name: "Git", details: "Basic Git knowledge (clone, commit, push)" },
    { name: "Terminal", details: "Comfortable with command line" },
    { name: "API Key", details: "OpenAI or Anthropic API key (free tier works)" },
    { name: "Code Editor", details: "VS Code, PyCharm, or similar" }
  ],

  technologies: [
    { name: "Python 3.9+", purpose: "Core programming language" },
    { name: "OpenAI SDK", purpose: "Access to GPT models" },
    { name: "Anthropic SDK", purpose: "Access to Claude models" },
    { name: "Click", purpose: "Building CLI interfaces" },
    { name: "Rich", purpose: "Beautiful terminal formatting" },
    { name: "python-dotenv", purpose: "Environment variable management" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — Understanding LLMs",
      estimatedTime: "1-1.5 hours",
      modules: [
        {
          id: "what-is-llm",
          title: "What is a Large Language Model?",
          content: `
A Large Language Model (LLM) is a type of AI that has been trained on vast amounts of text data to understand and generate human-like text. Think of it as a very sophisticated autocomplete system that can:

- Complete sentences
- Answer questions
- Write code
- Translate languages
- Summarize documents
- And much more...

### How LLMs Work (Simplified)

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                        LLM ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   INPUT                    PROCESSING                OUTPUT     │
│   ─────                    ──────────                ──────     │
│                                                                 │
│   "What is              ┌──────────────┐           "Python is   │
│    Python?"  ────────►  │  Transformer │  ────────► a high-     │
│                         │   Neural     │            level..."   │
│                         │   Network    │                        │
│                         └──────────────┘                        │
│                               │                                 │
│                    ┌──────────┴──────────┐                      │
│                    │                     │                      │
│               [Attention]          [Prediction]                 │
│               Understands          Generates next               │
│               context &            most probable                 │
│               relationships        token                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### The Training Process

LLMs learn through a process called **self-supervised learning**:

1. **Pre-training:** The model reads billions of text documents (books, websites, code, etc.)
2. **Pattern Learning:** It learns patterns in language—grammar, facts, reasoning styles
3. **Fine-tuning:** Additional training on specific tasks (like following instructions)
4. **RLHF:** Reinforcement Learning from Human Feedback makes responses more helpful

> **Key Insight:** LLMs don't "understand" text the way humans do. They predict the most likely next word (token) based on patterns learned during training. This statistical approach, at scale, produces remarkably intelligent-seeming behavior.
          \`
        },
        {
          id: "tokens",
          title: "Tokens — The Building Blocks",
          content: \`
### What Are Tokens?

Tokens are the fundamental units that LLMs process. A token can be:
- A word: \`"hello"\` → 1 token
- Part of a word: \`"understanding"\` → \`["under", "standing"]\` → 2 tokens
- A character: In some cases, individual characters
- Punctuation: \`"!"\` → 1 token

### Why Tokens Matter

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    TOKENIZATION EXAMPLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Text: "Hello, I'm learning about LLMs!"                 │
│                                                                 │
│   Tokens:                                                       │
│   ┌───────┬───┬───────┬──────────┬───────┬───────┬───┐         │
│   │ Hello │ , │  I'm  │ learning │ about │ LLMs  │ ! │         │
│   └───────┴───┴───────┴──────────┴───────┴───────┴───┘         │
│       1     2     3        4         5       6     7            │
│                                                                 │
│   Total: 7 tokens                                               │
│                                                                 │
│   💰 Cost Impact:                                               │
│   - GPT-4: ~$0.00021 (input) + ~$0.00063 (output)              │
│   - Claude: ~$0.000021 (input) + ~$0.000105 (output)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Token Limits and Context Windows

Every LLM has a **context window**—the maximum number of tokens it can process at once:

| Model | Context Window | Approximate Words |
|-------|---------------|-------------------|
| GPT-3.5 Turbo | 16,385 tokens | ~12,000 words |
| GPT-4 | 8,192 tokens | ~6,000 words |
| GPT-4 Turbo | 128,000 tokens | ~96,000 words |
| Claude 3 Sonnet | 200,000 tokens | ~150,000 words |

### Rule of Thumb 📏

> **1 token ≈ 4 characters ≈ 0.75 words** (in English)

### Code: Counting Tokens

\`\`\`python
# Using tiktoken (OpenAI's tokenizer)
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "Hello, I'm learning about Large Language Models!"
print(f"Token count: {count_tokens(text)}")  # Output: ~11 tokens
\`\`\`
          \`
        },
        {
          id: "parameters",
          title: "Key Parameters",
          content: \`
When calling an LLM API, you can control the output using several parameters:

### Temperature 🌡️

Controls randomness/creativity in responses.

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPERATURE SCALE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   0.0          0.3          0.7          1.0          2.0       │
│    │            │            │            │            │        │
│    ▼            ▼            ▼            ▼            ▼        │
│ ┌──────┐   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐       │
│ │DETER-│   │CONSER│    │BALAN-│    │CREAT-│    │ WILD │       │
│ │MINIS-│   │VATIVE│    │ CED  │    │ IVE  │    │      │       │
│ │ TIC  │   │      │    │      │    │      │    │      │       │
│ └──────┘   └──────┘    └──────┘    └──────┘    └──────┘       │
│                                                                 │
│ Best for:  Best for:   Best for:   Best for:   Best for:       │
│ - Math     - Code      - General   - Stories   - Brainstorm    │
│ - Facts    - Analysis  - Chat      - Poetry    - Wild ideas    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Max Tokens

Limits the length of the response. Useful for:
- Controlling costs
- Getting concise answers
- Preventing rambling

### System Prompt

Sets the behavior and personality of the assistant:

\`\`\`python
system_prompt = """You are a helpful coding assistant. 
You write clean, well-commented code.
You explain your solutions step by step.
You prefer Python but can work with any language."""
\`\`\`

### Top-p (Nucleus Sampling)

An alternative to temperature. Controls diversity by limiting the token pool:
- \`top_p=0.1\`: Only considers tokens in the top 10% probability
- \`top_p=0.9\`: Considers tokens in the top 90% probability

> **Recommendation:** For most applications, adjust **temperature** and leave **top_p** at default. Adjusting both can lead to unpredictable results.
          \`
        },
        {
          id: "api-architecture",
          title: "API Architecture",
          content: \`
### Request/Response Pattern

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    API COMMUNICATION FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   YOUR APP                    API                    LLM        │
│   ────────                    ───                    ───        │
│                                                                 │
│   ┌──────────┐  HTTP POST   ┌──────────┐  Process  ┌────────┐  │
│   │          │ ──────────►  │          │ ────────► │        │  │
│   │  Client  │              │  Server  │           │ Model  │  │
│   │          │ ◄──────────  │          │ ◄──────── │        │  │
│   └──────────┘   Response   └──────────┘  Generate └────────┘  │
│                                                                 │
│   Request Contains:          Response Contains:                 │
│   - API Key                  - Generated text                   │
│   - Model name               - Token usage                      │
│   - Messages array           - Finish reason                    │
│   - Parameters               - Request ID                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Message Roles

LLM APIs use a conversation format with different roles:

| Role | Purpose | Example |
|------|---------|---------|
| \`system\` | Sets assistant behavior | "You are a helpful assistant..." |
| \`user\` | Human's message | "What is Python?" |
| \`assistant\` | AI's response | "Python is a programming language..." |

### Conversation History

To maintain context, you send the entire conversation history:

\`\`\`python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    {"role": "user", "content": "What can I build with it?"}  # New message
]
\`\`\`
          \`
        },
        {
          id: "streaming",
          title: "Streaming vs. Non-Streaming",
          content: \`
### Non-Streaming (Standard)

\`\`\`
User sends request ──────────────────────► API processes
                                                 │
                                                 │ (wait 2-10 seconds)
                                                 │
User receives complete response ◄────────────────┘
\`\`\`

**Pros:** Simple to implement  
**Cons:** User waits with no feedback

### Streaming

\`\`\`
User sends request ──────────────────────► API processes
                                                 │
User sees "The"    ◄─────────────────────────────┤
User sees "The answer" ◄─────────────────────────┤
User sees "The answer is" ◄──────────────────────┤
User sees "The answer is Python" ◄───────────────┘
\`\`\`

**Pros:** Immediate feedback, better UX  
**Cons:** Slightly more complex to implement

### Code Comparison

\`\`\`python
# Non-streaming
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
\`\`\`
          \`
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building the CLI Assistant",
      estimatedTime: "3-5 hours",
      tasks: [
        {
          id: "task-1",
          title: "Project Setup",
          description: "Create the project structure and install dependencies",
          content: \`
### 1.1 Create Project Structure

\`\`\`bash
# Create project directory
mkdir sage-cli && cd sage-cli

# Create directory structure
mkdir -p src/sage tests docs
touch src/sage/__init__.py
touch src/sage/cli.py
touch src/sage/llm.py
touch src/sage/config.py
touch src/sage/utils.py
touch tests/__init__.py
touch tests/test_llm.py
touch .env.example
touch .gitignore
touch README.md
touch pyproject.toml
\`\`\`

Your structure should look like:

\`\`\`
sage-cli/
├── src/
│   └── sage/
│       ├── __init__.py
│       ├── cli.py          # CLI commands
│       ├── llm.py          # LLM interactions
│       ├── config.py       # Configuration
│       └── utils.py        # Helper functions
├── tests/
│   ├── __init__.py
│   └── test_llm.py
├── docs/
├── .env.example
├── .gitignore
├── README.md
└── pyproject.toml
\`\`\`

### 1.2 Create pyproject.toml

\`\`\`toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sage-cli"
version = "0.1.0"
description = "An AI-powered command-line assistant"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["ai", "cli", "llm", "assistant", "openai", "anthropic"]

dependencies = [
    "click>=8.1.0",
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "tiktoken>=0.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
sage = "sage.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
\`\`\`

### 1.3 Create .gitignore

\`\`\`gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
.venv/
env/

# Environment variables
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# Distribution / packaging
build/
dist/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
\`\`\`

### 1.4 Create .env.example

\`\`\`bash
# Copy this file to .env and fill in your API keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
DEFAULT_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.7
\`\`\`

### 1.5 Set Up Virtual Environment

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\\venv\\Scripts\\activate

# Install dependencies
pip install -e ".[dev]"

# Create your .env file
cp .env.example .env
# Now edit .env and add your API keys
\`\`\`
          \`
        },
        {
          id: "task-2",
          title: "Configuration Module",
          description: "Create the configuration management system",
          content: \`
### Create src/sage/config.py

\`\`\`python
"""
Configuration management for Sage CLI.

This module handles loading configuration from environment variables
and provides sensible defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Application configuration."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model settings
    default_model: str = "gpt-4"
    default_temperature: float = 0.7
    max_tokens: int = 2048
    
    # Available models
    OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    ANTHROPIC_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4"),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        )
    
    def get_provider(self, model: str) -> str:
        """Determine the provider for a given model."""
        if model in self.OPENAI_MODELS or model.startswith("gpt"):
            return "openai"
        elif model in self.ANTHROPIC_MODELS or model.startswith("claude"):
            return "anthropic"
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def validate(self, model: Optional[str] = None) -> None:
        """Validate that required API keys are present."""
        model = model or self.default_model
        provider = self.get_provider(model)
        
        if provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in your .env file or environment."
            )
        elif provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY in your .env file or environment."
            )


# Global configuration instance
config = Config.from_env()
\`\`\`
          \`
        },
        {
          id: "task-3",
          title: "LLM Client Module",
          description: "Build the unified LLM client interface",
          content: \`
### Create src/sage/llm.py

\`\`\`python
"""
LLM client module for Sage CLI.

This module provides a unified interface for interacting with
multiple LLM providers (OpenAI, Anthropic).
"""

from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from openai import OpenAI
from anthropic import Anthropic

from sage.config import config, Config


# Type alias for messages
Message = dict[str, str]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat request and return the response."""
        pass
    
    @abstractmethod
    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Send a chat request and stream the response."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat request and return the response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Send a chat request and stream the response."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def _convert_messages(self, messages: List[Message]) -> tuple[str, List[Message]]:
        """Convert messages to Anthropic format, extracting system prompt."""
        system_prompt = ""
        converted = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                converted.append(msg)
        
        return system_prompt, converted
    
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a chat request and return the response."""
        system_prompt, converted_messages = self._convert_messages(messages)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else None,
            messages=converted_messages,
        )
        
        return response.content[0].text
    
    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Send a chat request and stream the response."""
        system_prompt, converted_messages = self._convert_messages(messages)
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt if system_prompt else None,
            messages=converted_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text


def get_client(
    model: Optional[str] = None,
    config_instance: Optional[Config] = None,
) -> BaseLLMClient:
    """Factory function to get the appropriate LLM client."""
    cfg = config_instance or config
    model = model or cfg.default_model
    
    cfg.validate(model)
    
    provider = cfg.get_provider(model)
    
    if provider == "openai":
        return OpenAIClient(api_key=cfg.openai_api_key, model=model)
    elif provider == "anthropic":
        return AnthropicClient(api_key=cfg.anthropic_api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class Conversation:
    """Manages conversation history."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[Message] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Message]:
        return self.messages.copy()
    
    def clear(self) -> None:
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages
    
    def __len__(self) -> int:
        return len(self.messages)
\`\`\`
          \`
        },
        {
          id: "task-4",
          title: "Utility Functions",
          description: "Create helper functions for the CLI",
          content: \`
### Create src/sage/utils.py

\`\`\`python
"""
Utility functions for Sage CLI.
"""

import sys
from pathlib import Path
from typing import Optional

import tiktoken
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax


console = Console()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if file_path.stat().st_size > 100_000:
        raise ValueError(f"File too large: {path} (max 100KB)")
    
    return file_path.read_text(encoding="utf-8")


def print_markdown(text: str) -> None:
    """Print text as formatted markdown."""
    md = Markdown(text)
    console.print(md)


def print_code(code: str, language: str = "python") -> None:
    """Print code with syntax highlighting."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_panel(content: str, title: str = "", style: str = "blue") -> None:
    console.print(Panel(content, title=title, border_style=style))


def stream_print(text_generator, end: str = "") -> str:
    """Print streaming text and return the complete response."""
    full_response = ""
    
    for chunk in text_generator:
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print(end, flush=True)
    return full_response


def get_stdin() -> Optional[str]:
    """Read from stdin if available."""
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None
\`\`\`
          \`
        },
        {
          id: "task-5",
          title: "CLI Commands",
          description: "Build the command-line interface with Click",
          content: \`
### Create src/sage/cli.py

This is the main CLI file. Due to its length, here are the key commands:

\`\`\`python
"""CLI interface for Sage."""

import sys
from typing import Optional

import click
from rich.console import Console

from sage.config import config
from sage.llm import get_client, Conversation
from sage.utils import (
    console, print_markdown, print_error, 
    print_success, print_panel, stream_print, get_stdin,
)


SYSTEM_PROMPTS = {
    "default": """You are Sage, a helpful AI assistant in the terminal. 
You provide clear, concise answers.""",
    "code": """You are Sage, an expert programming assistant. 
You write clean, efficient, well-commented code.""",
    "concise": """You are Sage, a brief AI assistant. 
Give short, direct answers.""",
}


@click.group()
@click.version_option(version="0.1.0", prog_name="sage")
def main():
    """🔮 Sage - Your AI-powered command-line assistant."""
    pass


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.option("-m", "--model", default=None, help="Model to use")
@click.option("-t", "--temperature", default=0.7, type=float)
@click.option("--no-stream", is_flag=True)
def ask(question: tuple, model: Optional[str], temperature: float, no_stream: bool):
    """Ask a single question and get a response."""
    question_text = " ".join(question)
    
    try:
        client = get_client(model=model)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": question_text},
        ]
        
        console.print()
        
        if no_stream:
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response = client.chat(messages, temperature=temperature)
            print_markdown(response)
        else:
            response = stream_print(
                client.chat_stream(messages, temperature=temperature)
            )
        
        console.print()
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


@main.command()
@click.option("-m", "--model", default=None)
@click.option("-t", "--temperature", default=0.7, type=float)
@click.option("-s", "--system", default="default", 
              type=click.Choice(["default", "code", "concise"]))
def chat(model: Optional[str], temperature: float, system: str):
    """Start an interactive chat session."""
    try:
        client = get_client(model=model)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)
    
    conversation = Conversation(system_prompt=SYSTEM_PROMPTS[system])
    
    print_panel(
        "Welcome to Sage Chat! Type /help for commands, /quit to exit.",
        title="🔮 Sage Chat",
        style="green"
    )
    
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                command = user_input.lower()
                if command in ("/quit", "/exit"):
                    print_success("Goodbye!")
                    break
                elif command == "/clear":
                    conversation.clear()
                    print_success("Conversation cleared.")
                    continue
                elif command == "/help":
                    console.print("Commands: /clear, /quit, /help")
                    continue
            
            conversation.add_user_message(user_input)
            console.print("[bold green]Sage:[/bold green] ", end="")
            
            response = stream_print(
                client.chat_stream(
                    conversation.get_messages(),
                    temperature=temperature
                )
            )
            
            conversation.add_assistant_message(response)
            console.print()
            
        except KeyboardInterrupt:
            console.print()
            print_success("Goodbye!")
            break


# Additional commands: summarize, code, translate, models
# See full implementation in the course materials


if __name__ == "__main__":
    main()
\`\`\`
          \`
        },
        {
          id: "task-6",
          title: "Package Initialization",
          description: "Update the package __init__.py",
          content: \`
### Update src/sage/__init__.py

\`\`\`python
"""
Sage - An AI-powered command-line assistant.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from sage.llm import get_client, Conversation
from sage.config import config

__all__ = ["get_client", "Conversation", "config", "__version__"]
\`\`\`
          \`
        },
        {
          id: "task-7",
          title: "Write Tests",
          description: "Create unit tests for the LLM module",
          content: \`
### Create tests/test_llm.py

\`\`\`python
"""Tests for the LLM module."""

import pytest
from unittest.mock import Mock, patch

from sage.config import Config
from sage.llm import OpenAIClient, Conversation, get_client


class TestConversation:
    """Tests for the Conversation class."""
    
    def test_init_empty(self):
        conv = Conversation()
        assert len(conv) == 0
    
    def test_init_with_system_prompt(self):
        conv = Conversation(system_prompt="You are helpful.")
        assert len(conv) == 1
        assert conv.get_messages()[0]["role"] == "system"
    
    def test_add_messages(self):
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there!")
        
        messages = conv.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_clear_keeps_system_prompt(self):
        conv = Conversation(system_prompt="You are helpful.")
        conv.add_user_message("Hello")
        conv.clear()
        
        messages = conv.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestConfig:
    """Tests for configuration."""
    
    def test_get_provider_openai(self):
        cfg = Config()
        assert cfg.get_provider("gpt-4") == "openai"
    
    def test_get_provider_anthropic(self):
        cfg = Config()
        assert cfg.get_provider("claude-3-sonnet-20240229") == "anthropic"
    
    def test_get_provider_unknown(self):
        cfg = Config()
        with pytest.raises(ValueError, match="Unknown model"):
            cfg.get_provider("unknown-model")
\`\`\`

### Run Tests

\`\`\`bash
pytest tests/ -v
pytest tests/ --cov=sage --cov-report=term-missing
\`\`\`
          \`
        },
        {
          id: "task-8",
          title: "Create README",
          description: "Write project documentation",
          content: \`
### Create README.md

\`\`\`markdown
# 🔮 Sage CLI

An AI-powered command-line assistant.

## ✨ Features

- 🤖 **Multiple AI Providers** - OpenAI (GPT-4) and Anthropic (Claude)
- 💬 **Interactive Chat** - Conversation context
- ⚡ **Streaming Responses** - Real-time output
- 📝 **Summarization** - Summarize text and files
- 💻 **Code Generation** - Any programming language
- 🌐 **Translation** - Between languages

## 🚀 Installation

\\\`\\\`\\\`bash
git clone https://github.com/yourusername/sage-cli.git
cd sage-cli
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
\\\`\\\`\\\`

## 📖 Usage

\\\`\\\`\\\`bash
sage ask "What is Python?"
sage chat
sage summarize -f article.txt
sage code "fizzbuzz in python"
sage translate "Hello" -t spanish
\\\`\\\`\\\`

## 📄 License

MIT
\`\`\`
          \`
        },
        {
          id: "task-9",
          title: "Test Your CLI",
          description: "Manual testing checklist",
          content: \`
### Manual Testing Checklist

Run each command and verify it works:

\`\`\`bash
# 1. Check version
sage --version

# 2. List models
sage models

# 3. Ask a simple question
sage ask "What is 2 + 2?"

# 4. Ask with different temperature
sage ask "Write a creative opening line" -t 1.0

# 5. Generate code
sage code "fizzbuzz in python"

# 6. Generate code in another language
sage code "hello world" -l rust

# 7. Summarize text
sage summarize "The quick brown fox..."

# 8. Start a chat session
sage chat
# Try: ask question, follow-up, /clear, /quit
\`\`\`
          \`
        },
        {
          id: "task-10",
          title: "Publish to GitHub",
          description: "Push your project to GitHub",
          content: \`
### Initialize Git Repository

\`\`\`bash
git init
git add .
git commit -m "Initial commit: Sage CLI"
\`\`\`

### Create GitHub Repository

1. Go to github.com/new
2. Name: \`sage-cli\`
3. Description: "An AI-powered command-line assistant"
4. Make it public
5. Don't initialize with README
6. Click "Create repository"

### Push to GitHub

\`\`\`bash
git remote add origin https://github.com/yourusername/sage-cli.git
git branch -M main
git push -u origin main
\`\`\`

### Create a Release Tag

\`\`\`bash
git tag -a v0.1.0 -m "First release"
git push origin v0.1.0
\`\`\`

## 🎉 Congratulations!

You've completed Day 1 and built your first AI-powered CLI tool!

### What You Learned

✅ How Large Language Models work conceptually  
✅ Understanding tokens, context windows, and parameters  
✅ Using OpenAI and Anthropic Python SDKs  
✅ Building production-quality CLI with Click
✅ Implementing streaming responses
✅ Structuring a Python project for GitHub
          `
        }
      ]
    }
  ]
};

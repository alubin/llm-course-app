export const pythonPrimerContent = {
  title: "Python Primer for LLM Engineering",
  subtitle: "Understand every line of code before you write it",
  duration: "1-2 hours",
  
  sections: [
    {
      id: "basics",
      title: "Python Basics Refresher",
      content: `
### Variables and Basic Types

Python is dynamically typed — you don't declare types explicitly.

\`\`\`python
# Variables - no type declaration needed
name = "Sage"           # str (string)
version = 0.1           # float
max_tokens = 2048       # int
is_streaming = True     # bool
nothing = None          # NoneType (like null)

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0

# Constants (by convention, ALL_CAPS)
MAX_RETRIES = 3
DEFAULT_MODEL = "gpt-4"
\`\`\`

### String Formatting (f-strings)

\`\`\`python
name = "Claude"
version = 3.5

# F-string syntax: f"text {expression}"
message = f"Hello, {name}!"           # "Hello, Claude!"
info = f"Model: {name} v{version}"    # "Model: Claude v3.5"

# Format specifiers
price = 19.99
formatted = f"Price: \${price:.2f}"   # "Price: $19.99"

# Multi-line f-strings
error = f"""
Error occurred:
  Model: {name}
  Version: {version}
"""
\`\`\`

### Comparison to Other Languages

\`\`\`python
# Python                      # Java/JS equivalent
name = "Sage"                 # String name = "Sage";
numbers = [1, 2, 3]           # int[] numbers = {1, 2, 3};
if x > 0:                     # if (x > 0) {
    print("positive")         #     print("positive");
                              # }
for item in items:            # for (item of items) {
    process(item)             #     process(item); }
\`\`\`
      \`
    },
    {
      id: "type-hints",
      title: "Type Hints",
      content: \`
Type hints make code more readable and enable IDE autocompletion. They're optional but highly recommended.

### Basic Type Hints

\`\`\`python
# Variable annotations
name: str = "Sage"
count: int = 42
temperature: float = 0.7
enabled: bool = True

# Function annotations
def greet(name: str) -> str:
    """The -> str indicates return type."""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process_data(data: str) -> None:
    """-> None means no return value."""
    print(data)
\`\`\`

### Complex Types

\`\`\`python
from typing import List, Dict, Optional, Union, Any

# List of strings
def get_models() -> List[str]:
    return ["gpt-4", "gpt-3.5-turbo"]

# Dictionary
def get_config() -> Dict[str, Any]:
    return {"model": "gpt-4", "temperature": 0.7}

# Optional means "this type OR None"
def find_user(id: int) -> Optional[str]:
    if id == 1:
        return "alice"
    return None

# Union means "either type A or type B"
def process(value: Union[str, int]) -> str:
    return str(value)
\`\`\`

### Type Aliases

\`\`\`python
from typing import Dict, List

# Create a type alias for readability
Message = Dict[str, str]  # {"role": "user", "content": "Hello"}

def send_messages(messages: List[Message]) -> str:
    """Now it's clear what 'messages' should look like."""
    pass
\`\`\`
      \`
    },
    {
      id: "data-structures",
      title: "Data Structures",
      content: \`
### Lists

\`\`\`python
# Creating lists
models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
empty_list = []
numbers = list(range(5))  # [0, 1, 2, 3, 4]

# Accessing elements
first = models[0]         # "gpt-4"
last = models[-1]         # "claude-3" (negative = from end)

# Modifying
models.append("claude-2")           # Add to end
models.insert(0, "gpt-4-turbo")     # Insert at index
models.remove("gpt-3.5-turbo")      # Remove by value
popped = models.pop()               # Remove last

# Slicing
first_two = models[0:2]    # First two elements
last_two = models[-2:]     # Last two elements

# List comprehension
lengths = [len(m) for m in models]
gpt_models = [m for m in models if m.startswith("gpt")]
\`\`\`

### Dictionaries

\`\`\`python
# Creating dictionaries
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2048
}

# Accessing values
model = config["model"]              # Raises KeyError if missing
model = config.get("model")          # Returns None if missing
model = config.get("model", "gpt-3.5")  # Returns default

# Modifying
config["stream"] = True              # Add or update
del config["stream"]                 # Delete key

# Iterating
for key, value in config.items():
    print(f"{key}: {value}")
\`\`\`

### The Message Format (LLM APIs)

\`\`\`python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is..."},
]

# Adding a new message
messages.append({"role": "user", "content": "Tell me more"})

# Filtering by role
user_msgs = [m for m in messages if m["role"] == "user"]
\`\`\`
      \`
    },
    {
      id: "functions",
      title: "Functions Deep Dive",
      content: \`
### Basic Functions

\`\`\`python
def greet(name: str) -> str:
    """
    Generate a greeting message.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting string
    """
    return f"Hello, {name}!"
\`\`\`

### Default Arguments

\`\`\`python
def chat(
    message: str,
    model: str = "gpt-4",           # Default value
    temperature: float = 0.7,        # Default value
    stream: bool = False             # Default value
) -> str:
    """Parameters with defaults are optional."""
    pass

# Calling with defaults
chat("Hello")                        # Uses all defaults
chat("Hello", model="claude-3")      # Override model
chat("Hello", temperature=0.9)       # Override temperature
\`\`\`

### *args and **kwargs

\`\`\`python
# *args: Accept any number of positional arguments
def join_words(*words) -> str:
    """words becomes a tuple of all arguments."""
    return " ".join(words)

join_words("Hello", "world")  # "Hello world"

# **kwargs: Accept any number of keyword arguments
def configure(**options) -> dict:
    """options becomes a dict."""
    return options

configure(model="gpt-4", temp=0.7)
# Returns: {"model": "gpt-4", "temp": 0.7}
\`\`\`
      \`
    },
    {
      id: "classes",
      title: "Classes and OOP",
      content: \`
### Basic Class

\`\`\`python
class Dog:
    """A simple Dog class."""
    
    # Class variable (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name: str, age: int):
        """Constructor - called when creating an instance."""
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self) -> str:
        """Instance method."""
        return f"{self.name} says woof!"

# Creating instances
buddy = Dog("Buddy", 3)
print(buddy.name)      # "Buddy"
print(buddy.bark())    # "Buddy says woof!"
\`\`\`

### Inheritance

\`\`\`python
class Animal:
    """Base class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def speak(self) -> str:
        return "Some sound"

class Dog(Animal):
    """Dog inherits from Animal."""
    
    def speak(self) -> str:
        """Override parent method."""
        return f"{self.name} says woof!"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())
\`\`\`
      \`
    },
    {
      id: "abc",
      title: "Abstract Base Classes",
      content: \`
Abstract Base Classes define interfaces that subclasses must implement.

\`\`\`python
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    Abstract base class - cannot be instantiated.
    Forces subclasses to implement certain methods.
    """
    
    @abstractmethod
    def chat(self, messages: list) -> str:
        """Subclasses MUST implement this."""
        pass
    
    @abstractmethod
    def chat_stream(self, messages: list):
        """Another required method."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Regular method - optional to override."""
        return len(text.split())

# This would raise an error:
# client = BaseLLMClient()  # TypeError!

# This works:
class OpenAIClient(BaseLLMClient):
    def chat(self, messages: list) -> str:
        return "Response from OpenAI"
    
    def chat_stream(self, messages: list):
        yield "Streaming response"

client = OpenAIClient()  # Works!
\`\`\`

### Why Use ABCs?

\`\`\`python
# Ensures all clients have the same interface
def send_message(client: BaseLLMClient, message: str) -> str:
    """Works with ANY client that implements BaseLLMClient."""
    messages = [{"role": "user", "content": message}]
    return client.chat(messages)

# All these work:
send_message(OpenAIClient(), "Hello")
send_message(AnthropicClient(), "Hello")
\`\`\`
      \`
    },
    {
      id: "dataclasses",
      title: "Dataclasses",
      content: \`
Dataclasses reduce boilerplate for data-holding classes.

\`\`\`python
from dataclasses import dataclass
from typing import Optional

# Without dataclass (verbose)
class ConfigOld:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
    
    def __repr__(self):
        return f"Config(model={self.model})"

# With dataclass (clean!)
@dataclass
class Config:
    """Automatically generates __init__, __repr__, __eq__."""
    model: str
    temperature: float
    api_key: Optional[str] = None  # Default value
    max_tokens: int = 2048         # Default value

# Usage
config = Config(model="gpt-4", temperature=0.7)
print(config)  
# Config(model='gpt-4', temperature=0.7, api_key=None, max_tokens=2048)
\`\`\`

### Dataclass with Methods

\`\`\`python
@dataclass
class Config:
    model: str
    temperature: float = 0.7
    
    VALID_MODELS = ["gpt-4", "gpt-3.5-turbo"]  # Class variable
    
    def validate(self) -> bool:
        """You can still add methods."""
        return self.model in self.VALID_MODELS
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create from environment variables."""
        import os
        return cls(
            model=os.getenv("MODEL", "gpt-4"),
            temperature=float(os.getenv("TEMP", "0.7")),
        )
\`\`\`
      \`
    },
    {
      id: "decorators",
      title: "Decorators",
      content: \`
Decorators modify or enhance functions. They're the \`@something\` syntax.

### Understanding Decorators

\`\`\`python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("World")
# Before function call
# Hello, World!
# After function call
\`\`\`

### Common Built-in Decorators

\`\`\`python
class MyClass:
    
    @staticmethod
    def utility_function():
        """Doesn't need self or cls."""
        return "Static method"
    
    @classmethod
    def from_string(cls, data: str):
        """Receives class as first argument."""
        return cls()
    
    @property
    def full_name(self):
        """Makes method act like attribute."""
        return f"{self.first} {self.last}"
\`\`\`

### Click Decorators (Used in Course)

\`\`\`python
import click

@click.command()
@click.argument("name")
@click.option("-g", "--greeting", default="Hello")
@click.option("-v", "--verbose", is_flag=True)
def greet(name: str, greeting: str, verbose: bool):
    """Greet someone."""
    if verbose:
        click.echo(f"About to greet {name}...")
    click.echo(f"{greeting}, {name}!")

# CLI usage:
# python script.py World
# python script.py World -g "Hi"
# python script.py World -v
\`\`\`
      \`
    },
    {
      id: "generators",
      title: "Generators and Yield",
      content: \`
Generators produce values one at a time — perfect for streaming LLM responses.

### Basic Generators

\`\`\`python
# Regular function - returns all at once
def get_numbers_list(n: int) -> list:
    result = []
    for i in range(n):
        result.append(i)
    return result  # All in memory

# Generator - yields one at a time
def get_numbers_generator(n: int):
    for i in range(n):
        yield i  # One value, pause, resume

# Usage looks the same
for num in get_numbers_generator(5):
    print(num)
\`\`\`

### Why Generators for Streaming

\`\`\`python
from typing import Generator

def stream_response() -> Generator[str, None, None]:
    """
    Generator type hint: Generator[YieldType, SendType, ReturnType]
    """
    words = ["Hello", " ", "world", "!"]
    for word in words:
        yield word  # Send one word, pause

# Consuming a generator
for chunk in stream_response():
    print(chunk, end="", flush=True)  # Prints as they arrive
\`\`\`

### Used in Course (Streaming LLM)

\`\`\`python
def chat_stream(self, messages: list) -> Generator[str, None, None]:
    """Stream response tokens one at a time."""
    stream = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Print tokens as they arrive
for token in client.chat_stream(messages):
    print(token, end="", flush=True)
\`\`\`
      \`
    },
    {
      id: "context-managers",
      title: "Context Managers",
      content: \`
Context managers handle setup and cleanup with \`with\` statements.

### Basic Usage

\`\`\`python
# File handling - automatically closes file
with open("file.txt", "r") as f:
    content = f.read()
# File is automatically closed here

# Without context manager (error-prone)
f = open("file.txt", "r")
try:
    content = f.read()
finally:
    f.close()  # Easy to forget!
\`\`\`

### Creating Context Managers

\`\`\`python
from contextlib import contextmanager

@contextmanager
def timer():
    """Times code execution."""
    import time
    start = time.time()
    yield  # Code inside 'with' runs here
    end = time.time()
    print(f"Elapsed: {end - start:.2f}s")

with timer():
    result = sum(range(1000000))
# Prints: "Elapsed: 0.05s"
\`\`\`

### Used in Course

\`\`\`python
# Anthropic streaming
with self.client.messages.stream(...) as stream:
    for text in stream.text_stream:
        yield text
# Connection closed after block

# Rich console spinner
with console.status("[bold blue]Thinking...[/bold blue]"):
    response = client.chat(messages)
# Spinner stops automatically
\`\`\`
      \`
    },
    {
      id: "exceptions",
      title: "Exception Handling",
      content: \`
### Basic Try/Except

\`\`\`python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exception types
try:
    value = int("not a number")
except ValueError:
    print("Invalid number")
except TypeError:
    print("Wrong type")

# Catch any exception
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")

# With else and finally
try:
    result = some_operation()
except SomeError:
    print("Error occurred")
else:
    print("Success!")  # Only if no exception
finally:
    print("Cleanup")   # Always runs
\`\`\`

### Raising Exceptions

\`\`\`python
def validate_temperature(temp: float) -> None:
    if temp < 0.0 or temp > 2.0:
        raise ValueError(f"Temperature must be 0.0-2.0, got {temp}")

def get_api_key() -> str:
    key = os.getenv("API_KEY")
    if not key:
        raise ValueError("API key not found. Set API_KEY in .env")
    return key
\`\`\`
      \`
    },
    {
      id: "files",
      title: "Working with Files",
      content: \`
### Using pathlib (Modern Approach)

\`\`\`python
from pathlib import Path

# Create Path objects
file_path = Path("documents/report.txt")
home = Path.home()  # User's home directory
cwd = Path.cwd()    # Current working directory

# Path operations
full_path = home / "projects" / "sage"  # Join with /
print(full_path.parent)      # Parent directory
print(full_path.name)        # "sage"
print(full_path.suffix)      # File extension

# Check existence
if file_path.exists():
    print("File exists")
if file_path.is_file():
    print("It's a file")

# Read and write
content = file_path.read_text(encoding="utf-8")
file_path.write_text("Hello!", encoding="utf-8")

# Get file info
size = file_path.stat().st_size  # Size in bytes

# Find files
for py_file in Path(".").glob("**/*.py"):
    print(py_file)
\`\`\`
      \`
    },
    {
      id: "env",
      title: "Environment Variables",
      content: \`
### Using os.getenv

\`\`\`python
import os

# Get variable (None if not found)
api_key = os.getenv("OPENAI_API_KEY")

# Get with default
model = os.getenv("DEFAULT_MODEL", "gpt-4")

# Check if set
if os.getenv("DEBUG"):
    print("Debug mode enabled")
\`\`\`

### Using python-dotenv

\`\`\`python
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()

# Now os.getenv() finds them
api_key = os.getenv("OPENAI_API_KEY")
\`\`\`

**.env file:**
\`\`\`bash
# Don't commit to git!
OPENAI_API_KEY=sk-your-key-here
DEFAULT_MODEL=gpt-4
\`\`\`

**.env.example (commit this!):**
\`\`\`bash
# Copy to .env and fill in values
OPENAI_API_KEY=sk-your-key-here
DEFAULT_MODEL=gpt-4
\`\`\`
      \`
    },
    {
      id: "packages",
      title: "Package Structure",
      content: \`
### Directory Layout

\`\`\`
my_package/
├── src/
│   └── my_package/
│       ├── __init__.py      # Makes it a package
│       ├── main.py
│       ├── utils.py
│       └── config.py
├── tests/
│   └── test_main.py
├── pyproject.toml
├── .env
└── README.md
\`\`\`

### __init__.py

\`\`\`python
# src/my_package/__init__.py
"""Package docstring."""

__version__ = "0.1.0"

# Import for convenient access
from my_package.main import main_function
from my_package.config import config

__all__ = ["main_function", "config"]
\`\`\`

### Imports

\`\`\`python
# Absolute imports (preferred)
from my_package.utils import helper
from my_package.config import Config

# Relative imports (within same package)
from .utils import helper      # Same level
from ..other import something  # Parent level

# Import with alias
import numpy as np
import pandas as pd
\`\`\`
      \`
    },
    {
      id: "venv",
      title: "Virtual Environments",
      content: \`
Virtual environments isolate project dependencies.

### Creating and Using venv

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
.\\venv\\Scripts\\activate

# Your prompt changes to show (venv)
(venv) $ 

# Install packages
pip install click openai

# See installed packages
pip list

# Save dependencies
pip freeze > requirements.txt

# Deactivate
deactivate
\`\`\`

### pyproject.toml Dependencies

\`\`\`toml
[project]
dependencies = [
    "click>=8.1.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
]
\`\`\`

\`\`\`bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
\`\`\`
      \`
    },
    {
      id: "click",
      title: "Click Library",
      content: \`
Click builds command-line interfaces.

### Basic Command

\`\`\`python
import click

@click.command()
def hello():
    """Greets the user."""
    click.echo("Hello, World!")

if __name__ == "__main__":
    hello()
\`\`\`

### Arguments and Options

\`\`\`python
@click.command()
@click.argument("name")
@click.option("-c", "--count", default=1)
@click.option("-v", "--verbose", is_flag=True)
def greet(name: str, count: int, verbose: bool):
    """Greet NAME COUNT times."""
    for _ in range(count):
        click.echo(f"Hello, {name}!")

# python script.py Alice -c 3
\`\`\`

### Command Groups

\`\`\`python
@click.group()
def cli():
    """My CLI application."""
    pass

@cli.command()
def init():
    """Initialize project."""
    click.echo("Initializing...")

@cli.command()
@click.argument("name")
def greet(name):
    """Greet someone."""
    click.echo(f"Hello, {name}!")

# python script.py init
# python script.py greet Alice
\`\`\`
      \`
    },
    {
      id: "rich",
      title: "Rich Library",
      content: \`
Rich makes terminal output beautiful.

### Basic Output

\`\`\`python
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Styled text
console.print("Hello", style="bold red")
console.print("[bold blue]Blue text[/bold blue]")
console.print("[green]Success![/green]")

# Markdown
md = Markdown("# Title\\n\\n**Bold** and *italic*")
console.print(md)

# Panel (box)
console.print(Panel("Content", title="Title"))

# Code with syntax highlighting
code = 'def hello():\\n    print("Hi!")'
syntax = Syntax(code, "python", theme="monokai")
console.print(syntax)
\`\`\`

### Progress and Status

\`\`\`python
# Spinner while waiting
with console.status("[bold blue]Processing..."):
    time.sleep(2)
console.print("[green]Done![/green]")

# Progress bar
from rich.progress import track

for item in track(range(100), description="Loading"):
    process(item)
\`\`\`

### Input

\`\`\`python
name = console.input("[bold cyan]Your name:[/bold cyan] ")
console.print(f"Hello, {name}!")
\`\`\`
      `
    }
  ]
};

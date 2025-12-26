export const day6Content = {
  id: 6,
  title: "AI Agents with Tool Use",
  subtitle: "Build autonomous agents that can use tools and APIs",
  duration: "6-8 hours",
  difficulty: "Advanced",

  objectives: [
    "Understand the difference between chatbots and agents",
    "Learn the ReAct (Reasoning + Acting) pattern",
    "Implement function calling with OpenAI",
    "Build a tool/plugin system",
    "Create an agent loop for autonomous task execution",
    "Deploy a multi-tool AI agent"
  ],

  prerequisites: [
    { name: "Day 1-3", details: "Experience with LLM APIs and building applications" },
    { name: "Python", details: "Async/await and OOP patterns" },
    { name: "APIs", details: "Understanding of REST APIs" }
  ],

  technologies: [
    { name: "OpenAI Function Calling", purpose: "Structured tool invocation" },
    { name: "LangChain Agents", purpose: "Agent framework and tools" },
    { name: "Requests", purpose: "HTTP client for API calls" },
    { name: "Python AsyncIO", purpose: "Asynchronous execution" },
    { name: "Pydantic", purpose: "Tool schema validation" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — AI Agents",
      estimatedTime: "2-2.5 hours",
      modules: [
        {
          id: "chatbot-vs-agent",
          title: "Chatbot vs Agent",
          content: `
### The Key Difference

**Chatbot:**
- Responds to user messages
- Purely conversational
- No external actions

**Agent:**
- Takes actions to achieve goals
- Uses tools/APIs
- Autonomous decision-making

### Example Comparison

\`\`\`
User: "What's the weather in New York?"

CHATBOT:
"I don't have access to real-time weather data."

AGENT:
1. Recognizes need for weather tool
2. Calls weather API with location="New York"
3. Returns: "It's 72°F and sunny in New York"
\`\`\`

### Agent Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                        AI AGENT LOOP                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Goal: "Book a flight to Paris"                          │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────┐                                           │
│   │ Agent (LLM)     │  Thinks: "I need to search flights"       │
│   │  - Reasoning    │                                           │
│   │  - Planning     │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Tool Selection  │  Chooses: search_flights()                │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Tool Execution  │  Calls API, gets results                  │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Agent (LLM)     │  Thinks: "Now I can answer"               │
│   │  - Synthesis    │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   Response: "I found 3 flights to Paris..."                     │
│                                                                 │
│   (Loop continues until task complete)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Agent Components

1. **Brain (LLM)** - Reasoning and decision-making
2. **Tools** - Functions the agent can call
3. **Memory** - Conversation history and context
4. **Loop** - Iterative think-act-observe cycle
          `
        },
        {
          id: "function-calling",
          title: "Function Calling APIs",
          content: `
### OpenAI Function Calling

Allows LLMs to output structured function calls:

\`\`\`python
from openai import OpenAI

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Call with tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=tools,
    tool_choice="auto"  # Let model decide when to use tools
)

# Check if model wants to call a function
message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    print(f"Function: {function_name}")
    print(f"Arguments: {function_args}")
    # Output: Function: get_weather
    #         Arguments: {'location': 'NYC', 'unit': 'fahrenheit'}
\`\`\`

### Function Call Flow

\`\`\`
1. User: "What's the weather in NYC?"

2. LLM decides to call function:
   {
     "name": "get_weather",
     "arguments": {"location": "NYC"}
   }

3. Your code executes the actual function:
   result = get_weather("NYC")
   # {"temp": 72, "condition": "sunny"}

4. Send result back to LLM:
   messages.append({
     "role": "tool",
     "tool_call_id": "...",
     "content": json.dumps(result)
   })

5. LLM synthesizes final answer:
   "It's currently 72°F and sunny in NYC"
\`\`\`

### Anthropic Tool Use

Similar concept with Claude:

\`\`\`python
from anthropic import Anthropic

client = Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}]

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=tools,
    max_tokens=1024
)
\`\`\`
          `
        },
        {
          id: "react-pattern",
          title: "ReAct Pattern",
          content: `
### ReAct: Reasoning + Acting

Combines reasoning (thought) with actions in an iterative loop.

### ReAct Loop Example

\`\`\`
Task: "What's the population of the capital of France?"

Step 1 - THINK:
"I need to find the capital of France first"

Step 2 - ACT:
Action: search("capital of France")
Observation: "Paris is the capital of France"

Step 3 - THINK:
"Now I need to find Paris's population"

Step 4 - ACT:
Action: search("population of Paris")
Observation: "Paris has a population of 2.16 million"

Step 5 - THINK:
"I now have the answer"

Step 6 - FINISH:
"The capital of France is Paris, which has a population of 2.16 million"
\`\`\`

### Implementation Pattern

\`\`\`python
def react_loop(task: str, max_iterations: int = 10):
    """ReAct agent loop."""
    messages = [{"role": "user", "content": task}]

    for i in range(max_iterations):
        # THINK: Get LLM's next action
        response = llm.call(messages, tools=available_tools)

        # Check if done
        if response.finish_reason == "stop":
            return response.content

        # ACT: Execute tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)

                # OBSERVE: Add result to context
                messages.append({
                    "role": "tool",
                    "content": result
                })
        else:
            # No more actions needed
            return response.content

    return "Max iterations reached"
\`\`\`

### Why ReAct Works

1. **Transparency** - See the agent's reasoning
2. **Debugging** - Understand decision-making process
3. **Error Correction** - Agent can recover from mistakes
4. **Multi-step** - Solve complex tasks step-by-step
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building an AI Agent",
      estimatedTime: "3.5-5.5 hours",
      tasks: [
        {
          id: "task-1",
          title: "Project Setup",
          description: "Initialize agent project",
          content: `
### Create Project Structure

\`\`\`bash
mkdir ai-agent && cd ai-agent
mkdir -p agent/{tools,core}
touch agent/__init__.py
touch agent/tools/{__init__.py,base.py,calculator.py,web_search.py,weather.py}
touch agent/core/{__init__.py,llm_client.py,agent.py}
touch main.py config.py
\`\`\`

### Install Dependencies

\`\`\`bash
pip install openai anthropic requests python-dotenv pydantic
\`\`\`

### Create .env

\`\`\`bash
OPENAI_API_KEY=sk-your-key-here
SERPAPI_API_KEY=your-serpapi-key  # For web search
WEATHER_API_KEY=your-weather-key  # For weather
\`\`\`

### Project Structure

\`\`\`
ai-agent/
├── agent/
│   ├── core/
│   │   ├── llm_client.py    # LLM integration
│   │   └── agent.py         # Agent loop
│   └── tools/
│       ├── base.py          # Tool base class
│       ├── calculator.py    # Math tool
│       ├── web_search.py    # Search tool
│       └── weather.py       # Weather tool
├── config.py
└── main.py
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "Tool Base Class",
          description: "Create reusable tool framework",
          content: `
### Create agent/tools/base.py

\`\`\`python
"""Base tool class."""
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, Field

class ToolParameter(BaseModel):
    """Tool parameter definition."""
    type: str
    description: str
    enum: list = None
    required: bool = True

class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ToolParameter]:
        """Tool parameters schema."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool."""
        pass

    def to_openai_function(self) -> dict:
        """Convert tool to OpenAI function format."""
        properties = {}
        required = []

        for param_name, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param_name] = prop

            if param.required:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "Build Tools",
          description: "Create calculator and web search tools",
          content: `
### Create agent/tools/calculator.py

\`\`\`python
"""Calculator tool."""
from agent.tools.base import Tool, ToolParameter

class CalculatorTool(Tool):
    """Performs mathematical calculations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluates mathematical expressions. Use for any math calculations."

    @property
    def parameters(self):
        return {
            "expression": ToolParameter(
                type="string",
                description="Math expression to evaluate, e.g. '2 + 2' or '(10 * 5) / 2'"
            )
        }

    def execute(self, expression: str) -> str:
        """Execute calculation."""
        try:
            # Safe eval with limited scope
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
\`\`\`

### Create agent/tools/web_search.py

\`\`\`python
"""Web search tool using SerpAPI."""
import os
import requests
from agent.tools.base import Tool, ToolParameter

class WebSearchTool(Tool):
    """Searches the web for information."""

    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for current information. Use when you need up-to-date facts."

    @property
    def parameters(self):
        return {
            "query": ToolParameter(
                type="string",
                description="Search query"
            )
        }

    def execute(self, query: str) -> str:
        """Execute web search."""
        if not self.api_key:
            return "Web search not configured (missing SERPAPI_API_KEY)"

        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": 3
            }

            response = requests.get(url, params=params)
            results = response.json()

            # Extract top results
            snippets = []
            for result in results.get("organic_results", [])[:3]:
                snippets.append(f"{result['title']}: {result['snippet']}")

            return "\\n\\n".join(snippets) if snippets else "No results found"

        except Exception as e:
            return f"Search error: {str(e)}"
\`\`\`
          `
        },
        {
          id: "task-4",
          title: "LLM Client",
          description: "Build client with function calling",
          content: `
### Create agent/core/llm_client.py

\`\`\`python
"""LLM client with tool support."""
import json
from typing import List, Dict, Optional
from openai import OpenAI
from agent.tools.base import Tool

class LLMClient:
    """LLM client with function calling."""

    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model

    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Tool]] = None
    ) -> dict:
        """
        Send chat with optional tools.

        Returns:
            {
                "content": str,
                "tool_calls": list or None,
                "finish_reason": str
            }
        """
        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [tool.to_openai_function() for tool in tools]

        # Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto" if tools else None
        )

        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                }
                for tc in message.tool_calls
            ]

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "finish_reason": response.choices[0].finish_reason
        }
\`\`\`
          `
        },
        {
          id: "task-5",
          title: "Agent Loop",
          description: "Implement the ReAct agent",
          content: `
### Create agent/core/agent.py

\`\`\`python
"""AI Agent with ReAct loop."""
from typing import List, Dict
from agent.core.llm_client import LLMClient
from agent.tools.base import Tool

class Agent:
    """ReAct agent with tools."""

    def __init__(self, tools: List[Tool], model: str = "gpt-4"):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = LLMClient(model=model)
        self.max_iterations = 10

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Run agent on a task.

        Args:
            task: The task/question to solve
            verbose: Print intermediate steps

        Returns:
            Final answer
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant with access to tools. Use tools when needed to answer questions accurately."
            },
            {
                "role": "user",
                "content": task
            }
        ]

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\\n--- Iteration {iteration + 1} ---")

            # Get LLM response
            response = self.llm.chat(
                messages=messages,
                tools=list(self.tools.values())
            )

            # Check if done (no tool calls)
            if not response["tool_calls"]:
                if verbose:
                    print(f"✓ Final Answer: {response['content']}")
                return response["content"]

            # Execute tools
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

                if verbose:
                    print(f"🔧 Calling tool: {tool_name}")
                    print(f"   Arguments: {tool_args}")

                # Execute tool
                tool = self.tools.get(tool_name)
                if not tool:
                    result = f"Error: Unknown tool {tool_name}"
                else:
                    result = tool.execute(**tool_args)

                if verbose:
                    print(f"   Result: {result[:200]}...")

                # Add tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": str(tool_args)
                            }
                        }
                    ]
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                })

        return "Max iterations reached without completing task"
\`\`\`
          `
        },
        {
          id: "task-6",
          title: "Main Application",
          description: "Create CLI interface for the agent",
          content: `
### Create main.py

\`\`\`python
"""AI Agent CLI."""
from agent.core.agent import Agent
from agent.tools.calculator import CalculatorTool
from agent.tools.web_search import WebSearchTool

def main():
    # Initialize tools
    tools = [
        CalculatorTool(),
        WebSearchTool(),
    ]

    # Create agent
    agent = Agent(tools=tools, model="gpt-4")

    print("🤖 AI Agent Ready!")
    print("Available tools:", ", ".join([t.name for t in tools]))
    print("\\nType 'quit' to exit\\n")

    while True:
        # Get user input
        task = input("\\n👤 You: ").strip()

        if task.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not task:
            continue

        # Run agent
        try:
            answer = agent.run(task, verbose=True)
            print(f"\\n🤖 Agent: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
\`\`\`

### Run the Agent

\`\`\`bash
python main.py
\`\`\`

### Example Interactions

\`\`\`
You: What's 234 * 567?
🔧 Calling tool: calculator
   Arguments: {'expression': '234 * 567'}
   Result: 132678
🤖 Agent: 234 * 567 = 132,678

You: Who won the latest Nobel Prize in Physics?
🔧 Calling tool: web_search
   Arguments: {'query': 'latest Nobel Prize in Physics winner'}
   Result: [Search results...]
🤖 Agent: The 2024 Nobel Prize in Physics was awarded to...
\`\`\`

🎉 **Congratulations!** You've built an autonomous AI agent!
          `
        }
      ]
    }
  ]
};

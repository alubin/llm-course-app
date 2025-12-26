import type { DayContent } from './types';

export const day2Content: DayContent = {
  id: 2,
  title: "Chatbot with Memory & Context",
  subtitle: "Build a stateful chatbot with conversation persistence",
  duration: "5-7 hours",
  difficulty: "Intermediate",

  objectives: [
    "Understand why LLMs are stateless and how to add memory",
    "Implement conversation history management",
    "Handle context window limits with truncation strategies",
    "Persist conversations to a database with SQLAlchemy",
    "Build a web-based chat interface",
    "Deploy a production-ready chatbot API"
  ],

  prerequisites: [
    { name: "Day 1", details: "Completed CLI assistant project" },
    { name: "Python", details: "Comfortable with classes and async/await" },
    { name: "SQL Basics", details: "Understanding of databases" },
    { name: "HTML/JS", details: "Basic web development knowledge" }
  ],

  technologies: [
    { name: "FastAPI", purpose: "High-performance async web framework" },
    { name: "SQLAlchemy", purpose: "Database ORM for conversation storage" },
    { name: "OpenAI/Anthropic SDK", purpose: "LLM integration" },
    { name: "Uvicorn", purpose: "ASGI server" },
    { name: "SQLite", purpose: "Embedded database" },
    { name: "Jinja2", purpose: "HTML templating" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — Memory & Context",
      estimatedTime: "1.5-2 hours",
      modules: [
        {
          id: "stateless-llms",
          title: "Why LLMs Are Stateless",
          content: `
### The Stateless Nature of LLMs

Every API call to an LLM is independent. The model has **no memory** of previous conversations.

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    STATELESS LLM BEHAVIOR                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Request 1: "What's Python?"                                   │
│   ────────────────────────────►                                 │
│                                 LLM responds about Python       │
│   ◄────────────────────────────                                 │
│                                                                 │
│   Request 2: "Show me an example"                               │
│   ────────────────────────────►                                 │
│                                 LLM has NO IDEA what you        │
│   ◄────────────────────────────  want an example of!            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Why This Design?

1. **Scalability** - No need to maintain state across millions of users
2. **Simplicity** - Each request is independent and parallelizable
3. **Reliability** - No session management, no state corruption

### The Solution: Send History

\`\`\`python
# First message
messages = [
    {"role": "user", "content": "What's Python?"}
]
response1 = client.chat(messages)

# Second message - include ENTIRE history
messages = [
    {"role": "user", "content": "What's Python?"},
    {"role": "assistant", "content": response1},
    {"role": "user", "content": "Show me an example"}
]
response2 = client.chat(messages)
\`\`\`

> **Key Insight:** The application (not the LLM) is responsible for maintaining conversation state.
          `
        },
        {
          id: "context-window",
          title: "Context Window Limits",
          content: `
### The Context Window Problem

Every model has a maximum token limit:

| Model | Context Window | Cost Impact |
|-------|---------------|-------------|
| GPT-3.5 Turbo | 16K tokens | Low |
| GPT-4 | 8K tokens | High per token |
| GPT-4 Turbo | 128K tokens | Medium |
| Claude 3 Sonnet | 200K tokens | Low |

### What Happens When You Exceed?

\`\`\`python
# If total tokens > context_window:
# ❌ API Error: "maximum context length exceeded"
\`\`\`

### The Math

\`\`\`
Total Tokens = System Prompt + All Messages + Response Buffer

Example:
- System prompt: 50 tokens
- 10 message pairs (user + assistant): 200 tokens each = 2000 tokens
- Response buffer: 500 tokens
- Total: 2,550 tokens ✅ (fits in 8K)

After 50 message pairs:
- Total: 10,050 tokens ❌ (exceeds 8K!)
\`\`\`

### Strategies to Handle Limits

1. **Fixed Window (Simple)**
   - Keep only last N messages
   - Drop oldest when limit reached

2. **Token-Based Truncation (Better)**
   - Count tokens dynamically
   - Remove oldest messages until under limit

3. **Summarization (Advanced)**
   - Summarize old messages periodically
   - Keep summary + recent messages

4. **Hybrid**
   - Summarize very old messages
   - Keep recent N messages verbatim
          `
        },
        {
          id: "truncation-strategies",
          title: "Implementing Truncation",
          content: `
### Strategy 1: Fixed Window

\`\`\`python
MAX_MESSAGES = 20

def truncate_fixed(messages: list) -> list:
    """Keep only the last MAX_MESSAGES messages."""
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    # Keep system + last N messages
    return system_msgs + other_msgs[-MAX_MESSAGES:]
\`\`\`

**Pros:** Simple, predictable
**Cons:** Doesn't account for message length

### Strategy 2: Token-Based Truncation

\`\`\`python
import tiktoken

MAX_TOKENS = 4000

def truncate_by_tokens(messages: list, model: str = "gpt-4") -> list:
    """Remove oldest messages until under token limit."""
    encoding = tiktoken.encoding_for_model(model)

    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    # Always keep system prompt
    total_tokens = sum(len(encoding.encode(m["content"])) for m in system_msgs)

    # Add messages from newest to oldest
    kept_messages = []
    for msg in reversed(other_msgs):
        msg_tokens = len(encoding.encode(msg["content"]))
        if total_tokens + msg_tokens <= MAX_TOKENS:
            kept_messages.insert(0, msg)
            total_tokens += msg_tokens
        else:
            break

    return system_msgs + kept_messages
\`\`\`

**Pros:** Precise, cost-effective
**Cons:** Slightly more complex

### Strategy 3: Summarization

\`\`\`python
def summarize_old_messages(messages: list) -> list:
    """Summarize messages older than threshold."""
    KEEP_RECENT = 10

    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    if len(other_msgs) <= KEEP_RECENT:
        return messages

    # Split into old and recent
    old_msgs = other_msgs[:-KEEP_RECENT]
    recent_msgs = other_msgs[-KEEP_RECENT:]

    # Summarize old messages
    summary_prompt = f"Summarize this conversation:\\n{old_msgs}"
    summary = llm_client.chat([{"role": "user", "content": summary_prompt}])

    # Create summary message
    summary_msg = {
        "role": "system",
        "content": f"Previous conversation summary: {summary}"
    }

    return system_msgs + [summary_msg] + recent_msgs
\`\`\`

**Pros:** Preserves important context
**Cons:** Costs extra LLM call, may lose details
          `
        },
        {
          id: "persistence",
          title: "Database Persistence",
          content: `
### Why Persist Conversations?

1. **User Experience** - Resume conversations across sessions
2. **Analytics** - Analyze conversation patterns
3. **Debugging** - Trace issues in production
4. **Compliance** - Audit trails for regulated industries

### Database Schema

\`\`\`sql
-- Conversations table
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Messages table
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER,
    role TEXT CHECK(role IN ('system', 'user', 'assistant')),
    content TEXT,
    tokens INTEGER,
    created_at TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
\`\`\`

### SQLAlchemy Models

\`\`\`python
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(20))
    content = Column(Text)
    tokens = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
\`\`\`

### Usage Pattern

\`\`\`python
# Create new conversation
conversation = Conversation(user_id="user123", title="Python Help")
session.add(conversation)
session.commit()

# Add messages
msg1 = Message(
    conversation_id=conversation.id,
    role="user",
    content="What is Python?",
    tokens=5
)
session.add(msg1)
session.commit()

# Retrieve conversation
conv = session.query(Conversation).filter_by(id=conv_id).first()
messages = [{"role": m.role, "content": m.content} for m in conv.messages]
\`\`\`
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building the Chatbot",
      estimatedTime: "3.5-5 hours",
      tasks: [
        {
          id: "task-1",
          title: "Project Setup",
          description: "Initialize the FastAPI project structure",
          content: `
### Create Project Structure

\`\`\`bash
mkdir chatbot-memory && cd chatbot-memory

# Create directories
mkdir -p {app/{models,services,api,db},static,templates,tests}
touch app/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/api/__init__.py
touch app/db/__init__.py
\`\`\`

### Project Structure

\`\`\`
chatbot-memory/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py      # SQLAlchemy models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py   # LLM integration
│   │   └── chat_service.py  # Chat logic
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   └── db/
│       ├── __init__.py
│       └── database.py      # DB connection
├── static/
│   └── chat.js              # Frontend JS
├── templates/
│   └── chat.html            # Chat UI
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
\`\`\`

### Create requirements.txt

\`\`\`txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
openai>=1.0.0
anthropic>=0.18.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
jinja2>=3.1.0
python-multipart>=0.0.6
\`\`\`

### Install Dependencies

\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

### Create .env

\`\`\`bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
DATABASE_URL=sqlite:///./chatbot.db
DEFAULT_MODEL=gpt-4
MAX_CONTEXT_TOKENS=4000
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "Configuration Module",
          description: "Set up application configuration",
          content: `
### Create app/config.py

\`\`\`python
"""Application configuration."""
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

    # LLM Settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
    default_temperature: float = 0.7

    # Application
    app_name: str = "Memory Chatbot"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    class Config:
        env_file = ".env"

settings = Settings()
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "Database Models",
          description: "Create SQLAlchemy models for conversations",
          content: `
### Create app/db/database.py

\`\`\`python
"""Database setup and session management."""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
\`\`\`

### Create app/models/database.py

\`\`\`python
"""SQLAlchemy models."""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), index=True)
    title = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages)
        }

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    role = Column(String(20))
    content = Column(Text)
    tokens = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "created_at": self.created_at.isoformat()
        }
\`\`\`
          `
        },
        {
          id: "task-4",
          title: "Context Manager",
          description: "Build the conversation context truncation logic",
          content: `
### Create app/services/context_manager.py

\`\`\`python
"""Context window management and truncation."""
from typing import List, Dict
import tiktoken
from app.config import settings

class ContextManager:
    """Manages conversation context within token limits."""

    def __init__(self, model: str = settings.default_model):
        self.model = model
        self.max_tokens = settings.max_context_tokens
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in a message list."""
        total = 0
        for message in messages:
            # Count message overhead (role, formatting)
            total += 4  # Every message has role, content, etc.
            total += self.count_tokens(message.get("content", ""))
        total += 2  # Assistant response priming
        return total

    def truncate_by_tokens(
        self,
        messages: List[Dict[str, str]],
        reserve_tokens: int = 500
    ) -> List[Dict[str, str]]:
        """
        Truncate messages to fit within token limit.

        Args:
            messages: List of message dicts
            reserve_tokens: Tokens to reserve for response

        Returns:
            Truncated message list
        """
        max_tokens = self.max_tokens - reserve_tokens

        # Separate system messages (always keep)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Count system message tokens
        system_tokens = self.count_messages_tokens(system_msgs)
        remaining_tokens = max_tokens - system_tokens

        # Add messages from newest to oldest
        kept_messages = []
        current_tokens = 0

        for msg in reversed(other_msgs):
            msg_tokens = self.count_tokens(msg.get("content", "")) + 4
            if current_tokens + msg_tokens <= remaining_tokens:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        return system_msgs + kept_messages

    def create_summary_message(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """Create a summary of old messages (placeholder for now)."""
        message_count = len(messages)
        return {
            "role": "system",
            "content": f"[Previous conversation with {message_count} messages summarized]"
        }
\`\`\`
          `
        },
        {
          id: "task-5",
          title: "Chat Service",
          description: "Implement the core chat logic with memory",
          content: `
### Create app/services/chat_service.py

\`\`\`python
"""Chat service with conversation management."""
from typing import List, Dict, Optional, Generator
from sqlalchemy.orm import Session
from openai import OpenAI
from anthropic import Anthropic

from app.config import settings
from app.models.database import Conversation, Message
from app.services.context_manager import ContextManager

class ChatService:
    """Handles chat operations with memory."""

    def __init__(self, db: Session):
        self.db = db
        self.context_manager = ContextManager()
        self.openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

    def create_conversation(self, user_id: str, title: str = "New Chat") -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(user_id=user_id, title=title)
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self.db.query(Conversation).filter(Conversation.id == conversation_id).first()

    def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """Get all conversations for a user."""
        return self.db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc()).all()

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str
    ) -> Message:
        """Add a message to a conversation."""
        tokens = self.context_manager.count_tokens(content)
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens=tokens
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_messages(self, conversation_id: int) -> List[Dict[str, str]]:
        """Get all messages for a conversation in LLM format."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation.messages
        ]
        return messages

    def chat(
        self,
        conversation_id: int,
        user_message: str,
        model: str = settings.default_model,
        temperature: float = settings.default_temperature
    ) -> str:
        """Send a message and get a response."""
        # Add user message
        self.add_message(conversation_id, "user", user_message)

        # Get conversation history
        messages = self.get_messages(conversation_id)

        # Truncate if needed
        messages = self.context_manager.truncate_by_tokens(messages)

        # Get LLM response
        if model.startswith("gpt"):
            response = self._chat_openai(messages, model, temperature)
        elif model.startswith("claude"):
            response = self._chat_anthropic(messages, model, temperature)
        else:
            raise ValueError(f"Unknown model: {model}")

        # Save assistant response
        self.add_message(conversation_id, "assistant", response)

        return response

    def _chat_openai(self, messages: List[Dict], model: str, temperature: float) -> str:
        """Chat with OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, messages: List[Dict], model: str, temperature: float) -> str:
        """Chat with Anthropic."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")

        # Extract system message
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        other_msgs = [m for m in messages if m["role"] != "system"]

        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_msg if system_msg else None,
            messages=other_msgs
        )
        return response.content[0].text
\`\`\`
          `
        },
        {
          id: "task-6",
          title: "API Routes",
          description: "Create FastAPI endpoints",
          content: `
### Create app/api/routes.py

\`\`\`python
"""API routes for the chatbot."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from app.db.database import get_db
from app.services.chat_service import ChatService

router = APIRouter()

# Request/Response models
class ConversationCreate(BaseModel):
    user_id: str
    title: str = "New Chat"

class MessageRequest(BaseModel):
    message: str
    model: str = "gpt-4"
    temperature: float = 0.7

class MessageResponse(BaseModel):
    response: str
    conversation_id: int

@router.post("/conversations")
def create_conversation(
    conv: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    service = ChatService(db)
    conversation = service.create_conversation(conv.user_id, conv.title)
    return conversation.to_dict()

@router.get("/conversations/{user_id}")
def get_conversations(user_id: str, db: Session = Depends(get_db)):
    """Get all conversations for a user."""
    service = ChatService(db)
    conversations = service.get_user_conversations(user_id)
    return [c.to_dict() for c in conversations]

@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int, db: Session = Depends(get_db)):
    """Get all messages in a conversation."""
    service = ChatService(db)
    conversation = service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [m.to_dict() for m in conversation.messages]

@router.post("/conversations/{conversation_id}/chat")
def chat(
    conversation_id: int,
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    """Send a message and get a response."""
    service = ChatService(db)

    # Check conversation exists
    conversation = service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        response = service.chat(
            conversation_id,
            request.message,
            request.model,
            request.temperature
        )
        return MessageResponse(response=response, conversation_id=conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete a conversation."""
    service = ChatService(db)
    conversation = service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted"}
\`\`\`
          `
        },
        {
          id: "task-7",
          title: "Web Interface",
          description: "Build the HTML/JS chat UI",
          content: `
### Create templates/chat.html

\`\`\`html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; display: flex; gap: 20px; height: 100vh; }
        .sidebar { width: 300px; background: #2a2a2a; border-radius: 8px; padding: 20px; }
        .main { flex: 1; display: flex; flex-direction: column; }
        .chat-area { flex: 1; overflow-y: auto; padding: 20px; background: #2a2a2a; border-radius: 8px; margin-bottom: 20px; }
        .message { margin-bottom: 16px; padding: 12px; border-radius: 8px; }
        .user { background: #3a7bd5; margin-left: 20%; }
        .assistant { background: #2a4a2a; margin-right: 20%; }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 12px; border: none; border-radius: 8px; background: #3a3a3a; color: #fff; }
        button { padding: 12px 24px; border: none; border-radius: 8px; background: #3a7bd5; color: #fff; cursor: pointer; }
        button:hover { background: #2a5bc5; }
        .conv-list { max-height: calc(100vh - 200px); overflow-y: auto; }
        .conv-item { padding: 12px; margin-bottom: 8px; background: #3a3a3a; border-radius: 8px; cursor: pointer; }
        .conv-item:hover { background: #4a4a4a; }
        .conv-item.active { background: #3a7bd5; }
        h1 { margin-bottom: 20px; }
        h2 { margin-bottom: 16px; font-size: 18px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Conversations</h2>
            <button onclick="newConversation()">+ New Chat</button>
            <div class="conv-list" id="conversations"></div>
        </div>
        <div class="main">
            <h1>Memory Chatbot</h1>
            <div class="chat-area" id="messages"></div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type a message..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    <script src="/static/chat.js"></script>
</body>
</html>
\`\`\`

### Create static/chat.js

\`\`\`javascript
let userId = 'demo-user';
let currentConversationId = null;

async function loadConversations() {
    const response = await fetch(\`/api/conversations/\${userId}\`);
    const conversations = await response.json();

    const list = document.getElementById('conversations');
    list.innerHTML = conversations.map(c => \`
        <div class="conv-item \${c.id === currentConversationId ? 'active' : ''}"
             onclick="loadConversation(\${c.id})">
            <strong>\${c.title}</strong><br>
            <small>\${c.message_count} messages</small>
        </div>
    \`).join('');
}

async function newConversation() {
    const response = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, title: 'New Chat' })
    });
    const conversation = await response.json();
    currentConversationId = conversation.id;
    document.getElementById('messages').innerHTML = '';
    loadConversations();
}

async function loadConversation(id) {
    currentConversationId = id;
    const response = await fetch(\`/api/conversations/\${id}/messages\`);
    const messages = await response.json();

    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = messages.map(m => \`
        <div class="message \${m.role}">
            <strong>\${m.role === 'user' ? 'You' : 'Assistant'}:</strong><br>
            \${m.content}
        </div>
    \`).join('');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    loadConversations();
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    if (!message || !currentConversationId) return;

    // Show user message immediately
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML += \`<div class="message user"><strong>You:</strong><br>\${message}</div>\`;
    input.value = '';
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Send to API
    const response = await fetch(\`/api/conversations/\${currentConversationId}/chat\`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    const data = await response.json();

    // Show assistant response
    messagesDiv.innerHTML += \`<div class="message assistant"><strong>Assistant:</strong><br>\${data.response}</div>\`;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') sendMessage();
}

// Initialize
loadConversations();
\`\`\`
          `
        },
        {
          id: "task-8",
          title: "Main Application",
          description: "Create the FastAPI main app",
          content: `
### Create app/main.py

\`\`\`python
"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.database import init_db
from app.config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Include API routes
app.include_router(router, prefix="/api")

# Initialize database
@app.on_event("startup")
def startup_event():
    init_db()

# Root route
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

### Run the Application

\`\`\`bash
# Development
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
\`\`\`

Visit http://localhost:8000
          `
        },
        {
          id: "task-9",
          title: "Testing",
          description: "Test the chatbot functionality",
          content: `
### Manual Testing Checklist

1. **Create New Conversation**
   - Click "New Chat"
   - Verify conversation appears in sidebar

2. **Send Messages**
   - Type "What is Python?" and send
   - Verify response appears
   - Send follow-up: "Show me an example"
   - Verify it remembers context

3. **Test Memory**
   - Have a 10-message conversation
   - Refresh the page
   - Load the conversation
   - Verify all messages persist

4. **Test Context Limits**
   - Have a very long conversation (20+ exchanges)
   - Verify older messages get truncated
   - Verify chat still works

5. **Multiple Conversations**
   - Create 3 different conversations
   - Switch between them
   - Verify each maintains separate history

### API Testing with curl

\`\`\`bash
# Create conversation
curl -X POST http://localhost:8000/api/conversations \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": "test-user", "title": "API Test"}'

# Send message (use conversation_id from above)
curl -X POST http://localhost:8000/api/conversations/1/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello!"}'

# Get messages
curl http://localhost:8000/api/conversations/1/messages
\`\`\`
          `
        },
        {
          id: "task-10",
          title: "Deployment",
          description: "Deploy to production",
          content: `
### Create Dockerfile

\`\`\`dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

### Create docker-compose.yml

\`\`\`yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - DATABASE_URL=sqlite:///./data/chatbot.db
    volumes:
      - ./data:/app/data
    restart: unless-stopped
\`\`\`

### Deploy

\`\`\`bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
\`\`\`

### Push to GitHub

\`\`\`bash
git init
git add .
git commit -m "Initial commit: Memory chatbot"
git branch -M main
git remote add origin https://github.com/yourusername/chatbot-memory.git
git push -u origin main
\`\`\`

🎉 **Congratulations!** You've built a production-ready chatbot with memory!
          `
        }
      ]
    }
  ]
};

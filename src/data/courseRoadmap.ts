import { CourseDay } from './types';

export interface PythonPrimerTopic {
  id: string;
  title: string;
}

export const courseRoadmap: CourseDay[] = [
  {
    id: 0,
    title: "Why Python for AI?",
    description: "Understand why Python dominates AI development and how it leverages C/C++ for performance",
    duration: "30-45 minutes",
    language: "Python",
    status: "available",
    topics: ["Python vs C++", "Two-Language Problem", "NumPy & PyTorch Internals", "The AI Stack", "Performance Reality"],
    project: "Hands-on benchmarks",
    icon: "BookOpen"
  },
  {
    id: 1,
    title: "LLM Fundamentals + CLI Assistant",
    description: "Learn how LLMs work and build your first AI-powered CLI tool",
    duration: "4-8 hours",
    language: "Python",
    status: "available",
    topics: ["Tokens & Context Windows", "Temperature & Parameters", "OpenAI/Anthropic SDKs", "Click CLI Framework", "Streaming Responses"],
    project: "sage - AI CLI Assistant",
    icon: "Terminal"
  },
  {
    id: 2,
    title: "Chatbot with Memory & Context",
    description: "Build a conversational chatbot that remembers past interactions",
    duration: "4-6 hours",
    language: "Python",
    status: "available",
    topics: ["Conversation Management", "Context Windowing", "Memory Strategies", "Persistence", "Session Handling"],
    project: "Memory-enabled Chatbot",
    icon: "MessageSquare"
  },
  {
    id: 3,
    title: "RAG: Chat with Your Documents",
    description: "Create a document Q&A system using Retrieval Augmented Generation",
    duration: "5-7 hours",
    language: "Python",
    status: "available",
    topics: ["Embeddings", "Vector Databases", "Chunking Strategies", "Retrieval", "Context Injection"],
    project: "Document Q&A App",
    icon: "FileSearch"
  },
  {
    id: 4,
    title: "AI-Powered REST API",
    description: "Build a production-ready AI microservice with Spring Boot",
    duration: "5-7 hours",
    language: "Java",
    status: "available",
    topics: ["Spring Boot Setup", "REST Controllers", "AI Integration", "Error Handling", "API Documentation"],
    project: "Spring Boot AI Service",
    icon: "Server"
  },
  {
    id: 5,
    title: "Understanding Transformers & Fine-tuning",
    description: "Deep dive into transformer architecture and model customization",
    duration: "6-8 hours",
    language: "Python",
    status: "available",
    topics: ["Attention Mechanism", "Transformer Architecture", "Fine-tuning Basics", "LoRA", "Training Pipeline"],
    project: "Fine-tuned Model",
    icon: "Brain"
  },
  {
    id: 6,
    title: "AI Agents with Tool Use",
    description: "Build autonomous agents that can use tools and take actions",
    duration: "5-7 hours",
    language: "Python",
    status: "available",
    topics: ["Agent Architecture", "Tool Definition", "Function Calling", "ReAct Pattern", "Multi-step Reasoning"],
    project: "Autonomous AI Agent",
    icon: "Bot"
  },
  {
    id: 7,
    title: "Data Pipeline with AI Enrichment",
    description: "Create a data processing pipeline with ML predictions",
    duration: "5-7 hours",
    language: "Python",
    status: "available",
    topics: ["Data Ingestion", "Batch Processing", "AI Classification", "Data Validation", "Output Generation"],
    project: "ML Data Pipeline",
    icon: "Workflow"
  },
  {
    id: 8,
    title: "ML Systems for Production",
    description: "Design and deploy production ML systems: search, recommendations, and serving",
    duration: "6-8 hours",
    language: "Python",
    status: "available",
    topics: ["Search Ranking", "Recommendation Systems", "Model Serving", "Inference Optimization", "ML System Design"],
    project: "Production ML System",
    icon: "Layers"
  }
];

export const pythonPrimerTopics: PythonPrimerTopic[] = [
  { id: "basics", title: "Python Basics Refresher" },
  { id: "type-hints", title: "Type Hints" },
  { id: "data-structures", title: "Data Structures" },
  { id: "functions", title: "Functions Deep Dive" },
  { id: "classes", title: "Classes and OOP" },
  { id: "abc", title: "Abstract Base Classes" },
  { id: "dataclasses", title: "Dataclasses" },
  { id: "decorators", title: "Decorators" },
  { id: "generators", title: "Generators and Yield" },
  { id: "context-managers", title: "Context Managers" },
  { id: "exceptions", title: "Exception Handling" },
  { id: "files", title: "Working with Files" },
  { id: "env", title: "Environment Variables" },
  { id: "packages", title: "Package Structure" },
  { id: "venv", title: "Virtual Environments" },
  { id: "click", title: "Click Library" },
  { id: "rich", title: "Rich Library" },
];

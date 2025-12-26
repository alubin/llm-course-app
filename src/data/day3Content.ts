import type { DayContent } from './types';

export const day3Content: DayContent = {
  id: 3,
  title: "RAG - Chat with Your Documents",
  subtitle: "Build a Retrieval-Augmented Generation system",
  duration: "6-8 hours",
  difficulty: "Intermediate to Advanced",

  objectives: [
    "Understand Retrieval-Augmented Generation (RAG)",
    "Learn about embeddings and vector similarity",
    "Implement document chunking strategies",
    "Build a vector database with ChromaDB",
    "Create a complete RAG pipeline",
    "Deploy a document Q&A application"
  ],

  prerequisites: [
    { name: "Day 1 & 2", details: "Completed previous projects" },
    { name: "Python", details: "Comfortable with file I/O and data structures" },
    { name: "Linear Algebra", details: "Basic understanding of vectors (helpful)" }
  ],

  technologies: [
    { name: "ChromaDB", purpose: "Vector database for embeddings" },
    { name: "LangChain", purpose: "RAG framework and utilities" },
    { name: "OpenAI Embeddings", purpose: "Text-to-vector conversion" },
    { name: "PyPDF2", purpose: "PDF document parsing" },
    { name: "Gradio", purpose: "Quick web UI prototyping" },
    { name: "FAISS (optional)", purpose: "Alternative vector store" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — RAG Fundamentals",
      estimatedTime: "2-2.5 hours",
      modules: [
        {
          id: "what-is-rag",
          title: "What is RAG?",
          content: `
### The Problem RAG Solves

LLMs have three key limitations:

1. **Knowledge Cutoff** - Only knows information from training data
2. **No Private Data** - Can't access your company documents
3. **Hallucination** - Makes up plausible-sounding facts

### Retrieval-Augmented Generation (RAG)

RAG combines retrieval with generation to ground LLM responses in real documents.

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query: "What is our refund policy?"                      │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                              │
│   │  Convert to  │  Embedding Model                             │
│   │   Vector     │  ────────────►  [0.23, 0.45, -0.12, ...]    │
│   └──────────────┘                                              │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                              │
│   │   Search     │  Vector DB                                   │
│   │ Similar Docs │  ────────────►  Top 3 matching chunks        │
│   └──────────────┘                                              │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                              │
│   │  Inject into │  Prompt:                                     │
│   │    Prompt    │  "Given these docs: [chunks]                 │
│   └──────────────┘   Answer: [query]"                           │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                              │
│   │     LLM      │  ────────────►  "Our refund policy is..."    │
│   └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Benefits

✅ **Accuracy** - Grounded in actual documents
✅ **Up-to-date** - Add new documents anytime
✅ **Private** - Use confidential company data
✅ **Traceable** - Can cite source documents
          `
        },
        {
          id: "embeddings",
          title: "Understanding Embeddings",
          content: `
### What Are Embeddings?

Embeddings convert text into high-dimensional vectors that capture semantic meaning.

\`\`\`python
# Text
text = "The cat sat on the mat"

# Embedding (simplified - real ones have 1536+ dimensions)
embedding = [0.23, -0.45, 0.67, 0.12, -0.89, ...]
\`\`\`

### Semantic Similarity

Similar meanings = similar vectors:

\`\`\`
"dog"        →  [0.8, 0.2, 0.1]
"puppy"      →  [0.7, 0.3, 0.2]  ← Close to "dog"
"car"        →  [0.1, 0.9, 0.8]  ← Far from "dog"
\`\`\`

### How Similarity Works

Use **cosine similarity** to compare vectors:

\`\`\`python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate similarity (-1 to 1, 1 = identical)."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example
query_vec = [0.8, 0.2, 0.1]
doc1_vec = [0.7, 0.3, 0.2]  # Similar
doc2_vec = [0.1, 0.9, 0.8]  # Different

print(cosine_similarity(query_vec, doc1_vec))  # 0.95 (high)
print(cosine_similarity(query_vec, doc2_vec))  # 0.31 (low)
\`\`\`

### Popular Embedding Models

| Model | Dimensions | Best For |
|-------|-----------|----------|
| OpenAI text-embedding-3-small | 1536 | General purpose, fast |
| OpenAI text-embedding-3-large | 3072 | Highest quality |
| Sentence-BERT | 384-768 | Open source, fast |
| Cohere Embed | 1024 | Multilingual |

### Creating Embeddings

\`\`\`python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Usage
text = "Large Language Models are powerful"
vector = get_embedding(text)
print(f"Dimensions: {len(vector)}")  # 1536
\`\`\`
          `
        },
        {
          id: "chunking",
          title: "Document Chunking Strategies",
          content: `
### Why Chunk Documents?

1. **Token Limits** - Can't fit entire books in context
2. **Precision** - Smaller chunks = more focused retrieval
3. **Cost** - Less tokens = lower API costs

### Chunking Strategies

**1. Fixed-Size Chunks**

\`\`\`python
def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap prevents cutting sentences
    return chunks
\`\`\`

**Pros:** Simple, predictable
**Cons:** May split mid-sentence

**2. Sentence-Based Chunks**

\`\`\`python
import nltk
nltk.download('punkt')

def chunk_by_sentences(text: str, sentences_per_chunk: int = 5):
    """Chunk by complete sentences."""
    sentences = nltk.sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks
\`\`\`

**Pros:** Preserves sentence boundaries
**Cons:** Variable chunk sizes

**3. Semantic Chunks (Advanced)**

\`\`\`python
# Group sentences by topic using embeddings
# Detect topic changes based on embedding similarity
\`\`\`

**Pros:** Maintains topical coherence
**Cons:** Computationally expensive

### Recommended Settings

\`\`\`python
CHUNK_SIZE = 500       # tokens
CHUNK_OVERLAP = 50     # tokens overlap between chunks
MAX_CHUNKS = 3         # chunks to retrieve per query
\`\`\`

### Chunking with LangChain

\`\`\`python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)

chunks = splitter.split_text(document_text)
\`\`\`
          `
        },
        {
          id: "vector-databases",
          title: "Vector Databases",
          content: `
### What is a Vector Database?

A specialized database optimized for:
- Storing high-dimensional vectors
- Fast similarity search
- Metadata filtering

### Popular Vector DBs

| Database | Best For | Deployment |
|----------|----------|------------|
| ChromaDB | Local dev, prototypes | Embedded |
| Pinecone | Production, scale | Cloud SaaS |
| Weaviate | Self-hosted production | Docker/K8s |
| FAISS | In-memory, research | Library |
| Qdrant | Modern features | Docker/Cloud |

### ChromaDB Basics

\`\`\`python
import chromadb

# Initialize
client = chromadb.Client()

# Create collection
collection = client.create_collection("my_docs")

# Add documents
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "doc1.txt"}, {"source": "doc2.txt"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["tell me about document"],
    n_results=2
)

print(results['documents'])
print(results['metadatas'])
\`\`\`

### How Vector Search Works

\`\`\`
1. Query arrives: "What is Python?"
2. Convert to embedding: [0.23, 0.45, ...]
3. Find k-nearest neighbors (KNN) in vector space
4. Return top k most similar document chunks
\`\`\`

### Approximate Nearest Neighbor (ANN)

For speed, vector DBs use approximation algorithms:
- **HNSW** - Hierarchical Navigable Small World graphs
- **IVF** - Inverted File Index
- **LSH** - Locality-Sensitive Hashing

Trade-off: ~99% accuracy, 100x faster than exact search

### Metadata Filtering

\`\`\`python
# Search only in specific documents
results = collection.query(
    query_texts=["refund policy"],
    where={"source": "policies.pdf"},
    n_results=3
)
\`\`\`
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building RAG System",
      estimatedTime: "3.5-5.5 hours",
      tasks: [
        {
          id: "task-1",
          title: "Project Setup",
          description: "Initialize RAG project",
          content: `
### Create Project Structure

\`\`\`bash
mkdir rag-docqa && cd rag-docqa
mkdir -p {app,data/documents,data/vectorstore,tests}
touch app/{__init__.py,config.py,chunker.py,embedder.py,vectorstore.py,rag.py,main.py}
\`\`\`

### Install Dependencies

\`\`\`bash
pip install openai chromadb langchain pypdf2 python-dotenv tiktoken gradio
\`\`\`

### Create requirements.txt

\`\`\`txt
openai>=1.0.0
chromadb>=0.4.0
langchain>=0.1.0
pypdf2>=3.0.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
gradio>=4.0.0
nltk>=3.8.0
\`\`\`

### Create .env

\`\`\`bash
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4
CHUNK_SIZE=500
CHUNK_OVERLAP=50
COLLECTION_NAME=documents
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "Document Chunker",
          description: "Build document processing and chunking",
          content: `
### Create app/chunker.py

\`\`\`python
"""Document chunking utilities."""
from typing import List
import PyPDF2
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    """Process and chunk documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )

    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def load_txt(self, file_path: str) -> str:
        """Load text file."""
        return Path(file_path).read_text(encoding='utf-8')

    def load_document(self, file_path: str) -> str:
        """Load document based on extension."""
        path = Path(file_path)

        if path.suffix == '.pdf':
            return self.load_pdf(file_path)
        elif path.suffix in ['.txt', '.md']:
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)

    def process_document(self, file_path: str) -> List[dict]:
        """
        Load and chunk a document.

        Returns list of dicts with 'text' and 'metadata'.
        """
        text = self.load_document(file_path)
        chunks = self.chunk_text(text)

        return [
            {
                "text": chunk,
                "metadata": {
                    "source": Path(file_path).name,
                    "chunk_index": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "Vector Store",
          description: "Set up ChromaDB vector storage",
          content: `
### Create app/vectorstore.py

\`\`\`python
"""Vector store operations with ChromaDB."""
import chromadb
from typing import List, Dict
from openai import OpenAI
from app.config import settings

class VectorStore:
    """Manages vector storage and retrieval."""

    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path="./data/vectorstore")
        self.collection_name = collection_name
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to vector store.

        Args:
            chunks: List of dicts with 'text' and 'metadata'
        """
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"{chunk['metadata']['source']}_chunk_{i}" for i, chunk in enumerate(chunks)]

        # Add to collection (ChromaDB auto-generates embeddings with OpenAI)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 3, filter_metadata: Dict = None) -> List[Dict]:
        """
        Query vector store for similar documents.

        Returns:
            List of dicts with 'text', 'metadata', and 'score'
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata if filter_metadata else None
        )

        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else None
            })

        return formatted_results

    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "count": self.collection.count()
        }
\`\`\`
          `
        },
        {
          id: "task-4",
          title: "RAG Pipeline",
          description: "Build the complete RAG pipeline",
          content: `
### Create app/rag.py

\`\`\`python
"""RAG (Retrieval-Augmented Generation) pipeline."""
from typing import List, Dict
from openai import OpenAI
from app.vectorstore import VectorStore
from app.chunker import DocumentChunker
from app.config import settings

class RAGPipeline:
    """Complete RAG pipeline for document Q&A."""

    def __init__(self):
        self.vectorstore = VectorStore()
        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.llm_client = OpenAI(api_key=settings.openai_api_key)

    def ingest_document(self, file_path: str) -> Dict:
        """
        Process and add document to vector store.

        Returns:
            Statistics about the ingestion
        """
        # Chunk document
        chunks = self.chunker.process_document(file_path)

        # Add to vector store
        self.vectorstore.add_documents(chunks)

        return {
            "file": file_path,
            "chunks_created": len(chunks),
            "status": "success"
        }

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant document chunks."""
        return self.vectorstore.query(query, n_results=n_results)

    def generate_answer(
        self,
        query: str,
        contexts: List[Dict],
        model: str = None
    ) -> str:
        """
        Generate answer using retrieved contexts.

        Args:
            query: User's question
            contexts: Retrieved document chunks
            model: LLM model to use

        Returns:
            Generated answer
        """
        model = model or settings.llm_model

        # Build context string
        context_str = "\\n\\n".join([
            f"Source: {ctx['metadata']['source']}\\n{ctx['text']}"
            for ctx in contexts
        ])

        # Create prompt
        prompt = f"""Answer the question based on the context below. If the context doesn't contain enough information to answer the question, say so.

Context:
{context_str}

Question: {query}

Answer:"""

        # Generate response
        response = self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for factual answers
        )

        return response.choices[0].message.content

    def query(
        self,
        question: str,
        n_contexts: int = 3,
        return_sources: bool = False
    ) -> Dict:
        """
        Complete RAG query: retrieve + generate.

        Args:
            question: User's question
            n_contexts: Number of context chunks to retrieve
            return_sources: Whether to return source documents

        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        # Retrieve relevant contexts
        contexts = self.retrieve(question, n_results=n_contexts)

        # Generate answer
        answer = self.generate_answer(question, contexts)

        result = {"answer": answer}

        if return_sources:
            result["sources"] = [
                {
                    "source": ctx["metadata"]["source"],
                    "text": ctx["text"][:200] + "..."  # Preview
                }
                for ctx in contexts
            ]

        return result
\`\`\`
          `
        },
        {
          id: "task-5",
          title: "Gradio UI",
          description: "Create web interface with Gradio",
          content: `
### Create app/main.py

\`\`\`python
"""Main application with Gradio UI."""
import gradio as gr
from pathlib import Path
from app.rag import RAGPipeline
from app.config import settings

# Initialize RAG pipeline
rag = RAGPipeline()

def upload_document(file):
    """Handle document upload."""
    try:
        result = rag.ingest_document(file.name)
        return f"✅ Success! Processed {result['chunks_created']} chunks from {Path(file.name).name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ask_question(question, num_sources):
    """Handle question answering."""
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = rag.query(question, n_contexts=num_sources, return_sources=True)

        # Format sources
        sources_text = "\\n\\n".join([
            f"**Source: {s['source']}**\\n{s['text']}"
            for s in result['sources']
        ])

        return result['answer'], sources_text
    except Exception as e:
        return f"Error: {str(e)}", ""

# Build Gradio interface
with gr.Blocks(title="RAG Document Q&A") as demo:
    gr.Markdown("# 📚 RAG Document Q&A System")
    gr.Markdown("Upload documents and ask questions!")

    with gr.Tab("📤 Upload Documents"):
        upload_file = gr.File(label="Upload PDF or TXT file", file_types=[".pdf", ".txt", ".md"])
        upload_btn = gr.Button("Process Document")
        upload_output = gr.Textbox(label="Status")

        upload_btn.click(
            fn=upload_document,
            inputs=upload_file,
            outputs=upload_output
        )

    with gr.Tab("❓ Ask Questions"):
        question_input = gr.Textbox(label="Your Question", placeholder="What is the refund policy?")
        num_sources = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of sources to use")

        ask_btn = gr.Button("Ask")

        answer_output = gr.Textbox(label="Answer", lines=5)
        sources_output = gr.Markdown(label="Sources")

        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, num_sources],
            outputs=[answer_output, sources_output]
        )

    with gr.Tab("ℹ️ Info"):
        stats = rag.vectorstore.get_stats()
        gr.Markdown(f"""
        ### System Info
        - **Collection**: {stats['collection_name']}
        - **Document chunks**: {stats['count']}
        - **Embedding model**: {settings.embedding_model}
        - **LLM model**: {settings.llm_model}
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
\`\`\`

### Run the Application

\`\`\`bash
python -m app.main
\`\`\`

Visit http://localhost:7860
          `
        },
        {
          id: "task-6",
          title: "Testing & Evaluation",
          description: "Test the RAG system",
          content: `
### Testing Checklist

1. **Upload Documents**
   - Upload a PDF file
   - Upload a TXT file
   - Verify chunk count is reasonable

2. **Ask Questions**
   - Ask a question answerable from documents
   - Verify answer is accurate
   - Check that sources are cited

3. **Test Edge Cases**
   - Ask about something NOT in documents
   - Verify it says "not enough information"
   - Ask very specific questions
   - Ask broad questions

4. **Evaluate Quality**
   - **Relevance**: Are retrieved chunks relevant?
   - **Accuracy**: Is the answer factually correct?
   - **Completeness**: Does it fully answer the question?

### Create Evaluation Script

\`\`\`python
"""Evaluate RAG system."""
from app.rag import RAGPipeline

rag = RAGPipeline()

# Test questions
test_cases = [
    {
        "question": "What is the main topic of the document?",
        "expected_keywords": ["main", "topic", "subject"]
    },
    {
        "question": "What are the key features?",
        "expected_keywords": ["feature", "capability"]
    }
]

for case in test_cases:
    result = rag.query(case["question"], return_sources=True)
    print(f"Q: {case['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    print("-" * 50)
\`\`\`

### Performance Metrics

\`\`\`python
import time

def benchmark_retrieval(rag, query, n_runs=10):
    """Benchmark retrieval speed."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        rag.retrieve(query)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Average retrieval time: {avg_time:.3f}s")
\`\`\`

🎉 **Congratulations!** You've built a production RAG system!
          `
        }
      ]
    }
  ]
};

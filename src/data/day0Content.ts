import { DayContent } from './types';

export const day0Content: DayContent = {
  id: 0,
  title: "Why Python for AI? Understanding the Stack",
  subtitle: "Learn why Python dominates AI development and how it actually works under the hood",
  duration: "30-45 minutes",
  difficulty: "Beginner",
  objectives: [
    "Understand Python's role in the AI/ML ecosystem",
    "Learn how Python libraries leverage C/C++ for performance",
    "Discover the 'two-language problem' and its solution",
    "Explore the Python AI ecosystem and key libraries"
  ],
  prerequisites: [
    {
      name: "Basic Programming",
      details: "Understanding of any programming language"
    },
    {
      name: "Curiosity",
      details: "Interest in how AI tools actually work"
    }
  ],
  technologies: [
    {
      name: "Python",
      purpose: "High-level interface for AI development"
    },
    {
      name: "C/C++",
      purpose: "Performance-critical computation backends"
    },
    {
      name: "CUDA/cuBLAS",
      purpose: "GPU acceleration for neural networks"
    }
  ],
  sections: [
    {
      id: "theory",
      title: "Understanding the Python AI Stack",
      estimatedTime: "20-30 minutes",
      modules: [
        {
          id: "the-truth",
          title: "The Truth: Python is 'Slow' (And That's Okay)",
          content: `
## Why Python Gets Chosen Despite Being "Slow"

You're absolutely right - most Python AI libraries **are wrappers around C/C++ code**. Let's understand why this architecture is actually brilliant.

### The Performance Reality

Python is an **interpreted language**, which makes it inherently slower than compiled languages like C++ or Rust:

\`\`\`python
# This simple Python loop is 50-100x slower than equivalent C++
result = 0
for i in range(1000000):
    result += i
\`\`\`

**But here's the key insight:** When doing AI/ML work, you're rarely running pure Python for computation-heavy tasks.

### The Two-Language Solution

The AI community solved this with a **two-language architecture**:

| Layer | Language | Purpose |
|-------|----------|---------|
| **User Interface** | Python | Write, experiment, prototype |
| **Computation Engine** | C/C++/CUDA | Execute heavy math operations |

\`\`\`python
import torch

# You write this in Python (easy to read/write)
model = torch.nn.Linear(1024, 512)
output = model(input_data)

# But PyTorch executes this in optimized C++/CUDA
# The actual matrix multiplication happens at near-C speed!
\`\`\`

### How It Actually Works

When you call \`torch.matmul()\` or \`np.dot()\`:

1. **Python layer** validates inputs and prepares data
2. **Hands off** to C++ extension module
3. **C++ code** does the actual computation (sometimes on GPU)
4. **Results return** to Python as familiar objects

> Think of Python as the "steering wheel" and C++ as the "engine". You drive with Python, but C++ provides the power.
`
        },
        {
          id: "why-python",
          title: "Why Python Became the AI Standard",
          content: `
## The Perfect Storm: Why Python Won AI/ML

Python didn't dominate AI by accident. Several factors converged:

### 1. **Simplicity & Readability**

Compare implementing the same neural network forward pass:

**Python (PyTorch):**
\`\`\`python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
\`\`\`

**Pure C++ (without libraries):**
\`\`\`cpp
class SimpleNet {
private:
    Matrix w1, b1, w2, b2;
public:
    SimpleNet(int input, int hidden, int output) {
        w1 = Matrix::random(input, hidden);
        b1 = Matrix::zeros(1, hidden);
        w2 = Matrix::random(hidden, output);
        b2 = Matrix::zeros(1, output);
    }

    Matrix forward(const Matrix& x) {
        Matrix h1 = (x.dot(w1) + b1).relu();
        return h1.dot(w2) + b2;
    }
};
// Plus you need to implement Matrix class, CUDA kernels, etc.
\`\`\`

The Python version is **dramatically simpler** while being just as fast (because PyTorch uses C++ underneath).

### 2. **Rapid Prototyping**

AI research requires **constant experimentation**:

\`\`\`python
# Try different architectures in minutes
for layers in [2, 3, 4, 5]:
    for hidden_size in [64, 128, 256]:
        model = build_model(layers, hidden_size)
        accuracy = train_and_evaluate(model)
        print(f"Layers: {layers}, Hidden: {hidden_size} → {accuracy}")
\`\`\`

In C++, this experimentation would take **hours to days** due to:
- Compile times
- Manual memory management
- Verbose syntax
- Debugging difficulty

### 3. **The Ecosystem Effect**

Once a few key libraries chose Python, network effects kicked in:

\`\`\`
NumPy (1995) → SciPy (2001) → scikit-learn (2007)
    ↓
TensorFlow (2015) → PyTorch (2016)
    ↓
Hugging Face, LangChain, OpenAI SDK, Anthropic SDK...
\`\`\`

**Everyone builds for Python** because everyone uses Python.

### 4. **Interactive Development (REPL)**

Python's interactive shell is perfect for data exploration:

\`\`\`python
>>> import pandas as pd
>>> df = pd.read_csv('data.csv')
>>> df.head()  # Instantly see your data
>>> df.describe()  # Quick statistics
>>> df.plot()  # Visualize in seconds
\`\`\`

This iterative workflow is **essential** for data science and ML.
`
        },
        {
          id: "the-stack",
          title: "The Modern AI Stack: Layers Explained",
          content: `
## How Your Python Code Actually Runs

Let's trace what happens when you run a typical AI workload:

### Example: Running a GPT Model

\`\`\`python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_ids = tokenizer.encode("Hello, I'm a", return_tensors='pt')
output = model.generate(input_ids, max_length=50)
\`\`\`

### The Execution Stack:

| Layer | What Happens | Language |
|-------|--------------|----------|
| **Your Code** | \`model.generate()\` called | Python |
| **Transformers Library** | Manages model architecture, tokenization | Python |
| **PyTorch** | Tensor operations, autograd, training loops | Python API |
| **ATen (PyTorch Backend)** | Core tensor library, dispatches operations | C++ |
| **cuDNN/cuBLAS** | Optimized GPU kernels for conv/matmul | C++/CUDA |
| **Hardware** | GPU executes parallel operations | CUDA Cores |

### Performance Breakdown

For a single transformer layer forward pass:

- **Python overhead:** ~0.1ms (negligible)
- **PyTorch coordination:** ~0.5ms
- **C++/CUDA computation:** ~5-50ms (99% of time)

> **Key Insight:** Python adds ~1-2% overhead, but gives 10-100x faster development time.

### Real-World Example: NumPy

NumPy is a perfect example of this architecture:

\`\`\`python
import numpy as np

# Create two large matrices (1000x1000)
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

# Matrix multiplication
c = np.dot(a, b)  # This is fast!
\`\`\`

**What actually happens:**

1. **Python:** Validates shapes, prepares pointers
2. **NumPy (C):** Calls BLAS (Basic Linear Algebra Subprograms)
3. **BLAS:** Highly optimized C/Fortran/Assembly code
4. **CPU:** SIMD instructions (AVX-512, etc.) for parallelism

The result: **Near-native speed** from Python!
`
        },
        {
          id: "ecosystem",
          title: "The Python AI Ecosystem",
          content: `
## Key Libraries and Their Backends

Understanding what's actually under the hood:

### Core Numerical Computing

**NumPy** - Foundation for everything
- **Backend:** C + BLAS/LAPACK (Fortran)
- **Purpose:** Fast array operations
- **Example:**
\`\`\`python
import numpy as np
# C backend handles all the heavy lifting
arr = np.array([1, 2, 3, 4, 5])
mean = arr.mean()  # Optimized C code
\`\`\`

**SciPy** - Scientific computing
- **Backend:** C, C++, Fortran
- **Purpose:** Advanced algorithms (optimization, signal processing)

### Machine Learning Frameworks

**PyTorch**
- **Backend:** C++ (ATen), CUDA
- **Why it's popular:** Dynamic computation graphs, Pythonic API
- **Used by:** Meta, OpenAI, most research labs
\`\`\`python
import torch
# Python API, but executes in optimized C++/CUDA
tensor = torch.randn(1000, 1000, device='cuda')
result = torch.matmul(tensor, tensor)
\`\`\`

**TensorFlow**
- **Backend:** C++, CUDA, XLA (compiler)
- **Why it's popular:** Production deployment, mobile support
- **Used by:** Google, industry applications

**JAX** (Newer)
- **Backend:** XLA (Accelerated Linear Algebra)
- **Why it's interesting:** JIT compilation, functional programming
- **Used by:** Google Research, DeepMind

### LLM-Specific Libraries

**Hugging Face Transformers**
- **Backend:** PyTorch or TensorFlow
- **Purpose:** Pre-trained models, easy fine-tuning
\`\`\`python
from transformers import pipeline
# Abstracts away complexity
classifier = pipeline('sentiment-analysis')
result = classifier('I love this!')
\`\`\`

**LangChain**
- **Backend:** Pure Python (orchestration layer)
- **Purpose:** Chains together LLM calls
- **Note:** Orchestration layer, actual LLM calls happen via APIs

**OpenAI/Anthropic SDKs**
- **Backend:** Pure Python (HTTP client)
- **Purpose:** API communication
- **Note:** The heavy computation happens on their servers!

### When Python IS the Bottleneck

Pure Python is slow for:
- **Tight loops** with many operations
- **Custom algorithms** not in libraries
- **Data preprocessing** at massive scale

**Solution:** Use Numba, Cython, or Rust extensions:

\`\`\`python
from numba import jit

@jit  # Just-In-Time compile to machine code
def fast_function(x):
    result = 0
    for i in range(len(x)):
        result += x[i] ** 2
    return result
\`\`\`

### The Bottom Line

For AI/ML work:
1. ✅ **Use Python** for high-level logic, experimentation
2. ✅ **Leverage libraries** that use C/C++/CUDA underneath
3. ✅ **Vectorize operations** - avoid Python loops for number crunching
4. ❌ **Don't write** matrix multiplication in pure Python
5. ❌ **Don't worry** about Python overhead for model training

> "Python is the second-best language for everything" - but for AI, it might just be the best because it lets you **use the best tools** (C++, CUDA) with the **best interface** (Python).
`
        }
      ]
    },
    {
      id: "hands-on",
      title: "Hands-On: See It In Action",
      estimatedTime: "10-15 minutes",
      tasks: [
        {
          id: "task-benchmark",
          title: "Benchmark: Python vs NumPy",
          description: "Experience the performance difference between pure Python and NumPy's C backend",
          content: `
## Seeing the Speed Difference

Let's demonstrate why we use NumPy (C backend) instead of pure Python:

### The Experiment

Create a file \`benchmark.py\`:

\`\`\`python
import numpy as np
import time

# Pure Python implementation
def pure_python_sum(n):
    result = 0
    for i in range(n):
        result += i ** 2
    return result

# NumPy implementation
def numpy_sum(n):
    arr = np.arange(n)
    return np.sum(arr ** 2)

# Benchmark
n = 1_000_000

# Pure Python
start = time.time()
result_python = pure_python_sum(n)
python_time = time.time() - start

# NumPy (C backend)
start = time.time()
result_numpy = numpy_sum(n)
numpy_time = time.time() - start

print(f"Pure Python: {python_time:.4f}s")
print(f"NumPy (C):   {numpy_time:.4f}s")
print(f"Speedup:     {python_time / numpy_time:.1f}x faster")
print(f"Both correct: {result_python == result_numpy}")
\`\`\`

### Run It

\`\`\`bash
python benchmark.py
\`\`\`

### Expected Output

\`\`\`
Pure Python: 0.2847s
NumPy (C):   0.0031s
Speedup:     91.8x faster
Both correct: True
\`\`\`

### What You Learned

1. **Same result**, dramatically different performance
2. NumPy's C backend makes it **50-100x faster**
3. This is why we use libraries like NumPy, PyTorch, TensorFlow
4. Your AI code will be fast **if you use the right tools**

### Key Takeaway

> Write Python code that calls optimized C/C++ libraries. Don't try to write high-performance numerical code in pure Python!
`
        },
        {
          id: "task-inspect",
          title: "Inspect: What's Under the Hood?",
          description: "Explore the C extensions in popular Python AI libraries",
          content: `
## Looking Inside Your AI Libraries

Let's see the actual C/C++ code backing your Python libraries:

### 1. Find NumPy's C Extensions

\`\`\`python
import numpy as np
import os

# Find where NumPy is installed
numpy_path = os.path.dirname(np.__file__)
print(f"NumPy location: {numpy_path}")

# List compiled extensions (C/C++ modules)
core_path = os.path.join(numpy_path, 'core')
if os.path.exists(core_path):
    extensions = [f for f in os.listdir(core_path) if f.endswith('.so') or f.endswith('.pyd')]
    print(f"\\nCompiled C extensions found: {len(extensions)}")
    for ext in extensions[:5]:  # Show first 5
        print(f"  - {ext}")
\`\`\`

### 2. Check PyTorch Backend

\`\`\`python
import torch

# PyTorch backend information
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Backend:", torch.version.cuda if torch.cuda.is_available() else "CPU only")

# Show that operations use C++ backend
x = torch.randn(3, 3)
print(f"\\nTensor type: {type(x)}")
print(f"Tensor backend: {x.device}")
\`\`\`

### 3. Time API Call vs Local Computation

\`\`\`python
import time
import openai  # or anthropic

# Time an API call (computation happens on remote servers)
start = time.time()
# response = openai.ChatCompletion.create(...)  # Would make API call
api_time = time.time() - start
print(f"API call (network + remote GPU): ~1-3 seconds")

# Time local PyTorch operation (your machine)
import torch
start = time.time()
x = torch.randn(1000, 1000)
y = torch.matmul(x, x)  # Matrix multiply on your hardware
local_time = time.time() - start
print(f"Local computation (C++/CUDA): {local_time:.4f}s")
\`\`\`

### What This Reveals

1. **\`.so\`/\`.pyd\` files** are compiled C/C++ shared libraries
2. These are **NOT Python** - they're machine code
3. When you \`import numpy\`, Python loads these C modules
4. Your Python code is just **calling into C** for heavy work

### Bonus: See the Actual C Code

NumPy is open source! Check out the C code:
- GitHub: \`https://github.com/numpy/numpy/tree/main/numpy/core/src\`
- You'll see files like \`multiarray.c\`, \`ufunc.c\` - this is what runs!

**For PyTorch:**
- GitHub: \`https://github.com/pytorch/pytorch/tree/main/aten/src/ATen\`
- The \`ATen\` folder is the C++ tensor library

> You're not "just" writing Python - you're orchestrating highly optimized C/C++/CUDA code!
`
        }
      ]
    }
  ]
};

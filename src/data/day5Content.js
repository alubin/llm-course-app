export const day5Content = {
  id: 5,
  title: "Transformers & Fine-tuning LLMs",
  subtitle: "Deep dive into transformer architecture and model customization",
  duration: "8-10 hours",
  difficulty: "Advanced",

  objectives: [
    "Understand transformer architecture in depth",
    "Learn attention mechanisms and multi-head attention",
    "Implement fine-tuning with Hugging Face Transformers",
    "Use LoRA and QLoRA for efficient fine-tuning",
    "Train models on custom datasets",
    "Evaluate and deploy fine-tuned models"
  ],

  prerequisites: [
    { name: "Python & PyTorch", details: "Experience with deep learning frameworks" },
    { name: "Machine Learning", details: "Understanding of neural networks" },
    { name: "Linear Algebra", details: "Vectors, matrices, dot products" },
    { name: "GPU Access", details: "Google Colab or local GPU recommended" }
  ],

  technologies: [
    { name: "Hugging Face Transformers", purpose: "Pre-trained models and training" },
    { name: "PyTorch", purpose: "Deep learning framework" },
    { name: "PEFT (LoRA/QLoRA)", purpose: "Parameter-efficient fine-tuning" },
    { name: "Datasets", purpose: "Data loading and preprocessing" },
    { name: "Accelerate", purpose: "Distributed training" },
    { name: "bitsandbytes", purpose: "Model quantization" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — Transformer Architecture",
      estimatedTime: "3-4 hours",
      modules: [
        {
          id: "transformer-overview",
          title: "Transformer Architecture Overview",
          content: `
### The Transformer Revolution

Introduced in "Attention is All You Need" (2017), transformers replaced RNNs with:
- **Parallel processing** - Process entire sequences at once
- **Self-attention** - Understand relationships between all tokens
- **Positional encoding** - Maintain sequence order information

### Architecture Components

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: "The cat sat on the mat"                              │
│      │                                                          │
│      ▼                                                          │
│   Tokenization: [101, 2003, 4937, 2938, ...]                  │
│      │                                                          │
│      ▼                                                          │
│   ┌──────────────────┐                                         │
│   │ Token Embeddings │ (learnable vectors for each token)      │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────┐                                      │
│   │ Positional Encoding │ (add position information)           │
│   └────────┬────────────┘                                      │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────────────────────┐                         │
│   │  Multi-Head Self-Attention       │                         │
│   │  - Query, Key, Value projections │                         │
│   │  - Attention scores               │                         │
│   │  - 12-96 attention heads          │                         │
│   └────────┬─────────────────────────┘                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Add & Normalize  │ (residual connection)                   │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Feed-Forward Net │ (2-layer MLP)                           │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Add & Normalize  │                                         │
│   └────────┬─────────┘                                         │
│            │                                                    │
│   (Repeat 12-96 times - transformer layers)                    │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Output Layer     │ → Next token probabilities              │
│   └──────────────────┘                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Key Innovation: Self-Attention

Allows each token to "look at" all other tokens:

\`\`\`python
# Simplified attention
scores = softmax(Q @ K.T / sqrt(d_k))  # Attention weights
output = scores @ V                     # Weighted sum of values
\`\`\`
          `
        },
        {
          id: "attention-mechanism",
          title: "Attention Mechanism Deep Dive",
          content: `
### Scaled Dot-Product Attention

\`\`\`
Attention(Q, K, V) = softmax(QK^T / √d_k) V
\`\`\`

**Components:**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I have?"

### Step-by-Step Example

\`\`\`python
import torch
import torch.nn.functional as F

# Input: [batch, seq_len, d_model]
d_model = 512
seq_len = 10
x = torch.randn(1, seq_len, d_model)

# Linear projections
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

Q = x @ W_q  # [1, 10, 512]
K = x @ W_k  # [1, 10, 512]
V = x @ W_v  # [1, 10, 512]

# Attention scores
scores = (Q @ K.transpose(-2, -1)) / (d_model ** 0.5)  # [1, 10, 10]

# Apply softmax
attention_weights = F.softmax(scores, dim=-1)

# Apply to values
output = attention_weights @ V  # [1, 10, 512]
\`\`\`

### Multi-Head Attention

Run attention multiple times in parallel with different learned projections:

\`\`\`python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Linear projections split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose for attention: [batch, heads, seq, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention = F.softmax(scores, dim=-1)
        output = attention @ V

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)

        # Final linear projection
        return self.W_o(output)
\`\`\`

### Why Multiple Heads?

Each head can learn different relationships:
- Head 1: Grammar and syntax
- Head 2: Semantic meaning
- Head 3: Long-range dependencies
- etc.
          `
        },
        {
          id: "fine-tuning-approaches",
          title: "Fine-tuning Approaches",
          content: `
### Full Fine-tuning

Update **all** model parameters:

**Pros:**
- Maximum performance potential
- Complete adaptation to task

**Cons:**
- Requires huge GPU memory (7B model = 28GB+)
- Risk of catastrophic forgetting
- Expensive to train

\`\`\`python
# Full fine-tuning
for param in model.parameters():
    param.requires_grad = True  # Update all parameters
\`\`\`

### LoRA (Low-Rank Adaptation)

Freeze base model, add small trainable matrices:

\`\`\`
W' = W + ∆W
∆W = B × A  (low-rank decomposition)
\`\`\`

**Example:**
- Original weight: 4096 × 4096 = 16M parameters
- LoRA: (4096 × 8) + (8 × 4096) = 65K parameters
- **250x fewer parameters!**

\`\`\`python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4.7M || all params: 7B || trainable%: 0.067%
\`\`\`

**Pros:**
- 10-100x less memory
- Fast training
- Multiple LoRA adapters for different tasks

**Cons:**
- Slightly lower performance than full fine-tuning

### QLoRA (Quantized LoRA)

LoRA + 4-bit quantization:

\`\`\`python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
\`\`\`

**Pros:**
- Train 7B model on single consumer GPU (24GB)
- Minimal performance loss
- Production-ready technique

### When to Use What?

| Method | GPU Needed | Best For |
|--------|-----------|----------|
| Full Fine-tuning | 80GB+ | Maximum performance, unlimited resources |
| LoRA | 40GB | Good balance of performance and efficiency |
| QLoRA | 16-24GB | Consumer GPUs, experimentation |
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Fine-tuning LLMs",
      estimatedTime: "5-6 hours",
      tasks: [
        {
          id: "task-1",
          title: "Setup Environment",
          description: "Prepare training environment",
          content: `
### Install Dependencies

\`\`\`bash
pip install transformers datasets peft accelerate bitsandbytes
pip install torch torchvision torchaudio  # CUDA version
pip install wandb  # Optional: experiment tracking
\`\`\`

### Check GPU

\`\`\`python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
\`\`\`

### Login to Hugging Face

\`\`\`bash
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
\`\`\`

### Project Structure

\`\`\`bash
mkdir llm-finetuning && cd llm-finetuning
touch train.py dataset.py config.py evaluate.py
mkdir data models outputs
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "Prepare Dataset",
          description: "Load and format training data",
          content: `
### Create dataset.py

\`\`\`python
"""Dataset preparation for fine-tuning."""
from datasets import load_dataset, Dataset
import json

def load_custom_dataset(file_path: str):
    """
    Load custom JSONL dataset.

    Format:
    {"instruction": "...", "input": "...", "output": "..."}
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def format_instruction(example):
    """Format example as instruction-following prompt."""
    if example['input']:
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

    return {"text": prompt}

def prepare_dataset(dataset_name="databricks/databricks-dolly-15k"):
    """Load and prepare dataset."""
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Format for instruction tuning
    dataset = dataset.map(format_instruction, remove_columns=dataset['train'].column_names)

    # Split
    train_test = dataset['train'].train_test_split(test_size=0.1)

    return train_test['train'], train_test['test']

if __name__ == "__main__":
    train_ds, eval_ds = prepare_dataset()
    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples: {len(eval_ds)}")
    print(f"Example:\\n{train_ds[0]['text']}")
\`\`\`

### Create Sample Data

\`\`\`bash
cat > data/sample.jsonl << 'EOF'
{"instruction": "Explain what machine learning is", "input": "", "output": "Machine learning is a subset of AI..."}
{"instruction": "Write a Python function", "input": "to calculate fibonacci", "output": "def fibonacci(n):\\n    if n <= 1:\\n        return n..."}
EOF
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "Configure Training",
          description: "Set up model and training parameters",
          content: `
### Create config.py

\`\`\`python
"""Training configuration."""
from dataclasses import dataclass
from transformers import TrainingArguments
from peft import LoraConfig

@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

@dataclass
class LoRAConfig:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

def get_training_args():
    """Get training arguments."""
    return TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        disable_tqdm=False,
        report_to="none"  # or "wandb" for experiment tracking
    )
\`\`\`
          `
        },
        {
          id: "task-4",
          title: "Training Script",
          description: "Implement the training loop",
          content: `
### Create train.py

\`\`\`python
"""Fine-tuning script with QLoRA."""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from config import ModelConfig, LoRAConfig, get_training_args
from dataset import prepare_dataset

def load_model_and_tokenizer(config: ModelConfig):
    """Load model with quantization."""

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def main():
    # Configs
    model_config = ModelConfig()
    lora_config_dict = LoRAConfig()

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_config)

    # LoRA configuration
    peft_config = LoraConfig(
        r=lora_config_dict.lora_r,
        lora_alpha=lora_config_dict.lora_alpha,
        lora_dropout=lora_config_dict.lora_dropout,
        target_modules=lora_config_dict.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Prepare dataset
    print("Loading dataset...")
    train_dataset, eval_dataset = prepare_dataset()

    # Training arguments
    training_args = get_training_args()

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print("Saving model...")
    trainer.model.save_pretrained("./outputs/final_model")
    tokenizer.save_pretrained("./outputs/final_model")

    print("Training complete!")

if __name__ == "__main__":
    main()
\`\`\`

### Run Training

\`\`\`bash
python train.py
\`\`\`
          `
        },
        {
          id: "task-5",
          title: "Inference & Evaluation",
          description: "Test the fine-tuned model",
          content: `
### Create evaluate.py

\`\`\`python
"""Inference with fine-tuned model."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_finetuned_model(base_model_name, adapter_path):
    """Load base model with LoRA adapter."""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text=""):
    """Generate response from instruction."""
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response = response.split("### Response:")[-1].strip()

    return response

def main():
    # Load model
    print("Loading model...")
    model, tokenizer = load_finetuned_model(
        base_model_name="meta-llama/Llama-2-7b-hf",
        adapter_path="./outputs/final_model"
    )

    # Test examples
    examples = [
        ("Explain what Python is", ""),
        ("Write a function", "to calculate factorial"),
        ("What is machine learning?", "")
    ]

    for instruction, input_text in examples:
        print(f"\\n{'='*60}")
        print(f"Instruction: {instruction}")
        if input_text:
            print(f"Input: {input_text}")

        response = generate_response(model, tokenizer, instruction, input_text)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
\`\`\`

### Push to Hugging Face Hub

\`\`\`python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./outputs/final_model",
    repo_id="your-username/llama2-finetuned",
    repo_type="model"
)
\`\`\`

🎉 **Congratulations!** You've fine-tuned an LLM!
          `
        }
      ]
    }
  ]
};

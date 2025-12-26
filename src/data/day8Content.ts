import type { DayContent } from './types';

export const day8Content: DayContent = {
  id: 8,
  title: "ML Systems for Production",
  subtitle: "Design and deploy production-ready ML systems",
  duration: "6-8 hours",
  difficulty: "Advanced",

  objectives: [
    "Understand ML system design patterns for production",
    "Learn search ranking and recommendation architectures",
    "Master model serving and inference optimization",
    "Design scalable ML pipelines",
    "Apply ethical AI and bias mitigation techniques"
  ],

  prerequisites: [
    { name: "Days 1-7", details: "Completed previous course content" },
    { name: "ML Basics", details: "Understanding of ML fundamentals" },
    { name: "System Design", details: "Basic distributed systems concepts" },
    { name: "Python", details: "Proficient Python programming" }
  ],

  technologies: [
    { name: "vLLM / TGI", purpose: "High-performance LLM serving" },
    { name: "Ray Serve", purpose: "Scalable model deployment" },
    { name: "Docker/K8s", purpose: "Containerization and orchestration" },
    { name: "Redis", purpose: "Caching and feature stores" },
    { name: "Prometheus/Grafana", purpose: "Monitoring and observability" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — ML System Design",
      estimatedTime: "3-4 hours",
      modules: [
        {
          id: "search-ranking",
          title: "Search & Ranking Systems",
          content: `
### How Search Works

Modern search combines multiple stages:

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH RANKING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query: "best python tutorial"                                 │
│      │                                                          │
│      ▼                                                          │
│   ┌──────────────────┐                                         │
│   │ Query Processing │ Tokenize, expand, spell-check            │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │  Retrieval (L1)  │ Fast candidate selection                │
│   │  1M → 1000 docs  │ BM25, embedding similarity              │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │   Ranking (L2)   │ ML-based scoring                        │
│   │  1000 → 100 docs │ Features + learned weights              │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │  Reranking (L3)  │ Fine-grained ranking                    │
│   │  100 → 10 docs   │ Cross-encoders, LLMs                    │
│   └────────┬─────────┘                                         │
│            │                                                    │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Business Logic   │ Diversity, freshness, personalization   │
│   └──────────────────┘                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Key Ranking Signals

| Signal Type | Examples | Purpose |
|-------------|----------|---------|
| **Query-Doc Relevance** | BM25, embedding similarity | Text matching |
| **Document Quality** | PageRank, authority score | Trustworthiness |
| **User Signals** | Click rates, dwell time | Engagement |
| **Freshness** | Publication date, updates | Recency |
| **Personalization** | User history, preferences | Relevance to user |

### Learning to Rank (LTR)

\`\`\`python
# Pointwise: Predict relevance score directly
def pointwise_model(query_features, doc_features):
    combined = concat(query_features, doc_features)
    return model.predict(combined)  # 0-1 relevance

# Pairwise: Learn which doc is better
def pairwise_loss(score_a, score_b, label):
    # label = 1 if A > B, -1 if B > A
    return max(0, 1 - label * (score_a - score_b))

# Listwise: Optimize entire ranking
def listwise_loss(predicted_ranking, ideal_ranking):
    return ndcg_loss(predicted_ranking, ideal_ranking)
\`\`\`

### Metrics for Search

| Metric | Formula | Measures |
|--------|---------|----------|
| **Precision@K** | Relevant in top K / K | Quality of top results |
| **Recall@K** | Relevant found / Total relevant | Coverage |
| **MRR** | Mean 1/rank of first relevant | Position of best result |
| **NDCG** | DCG / Ideal DCG | Ranking quality |
          `
        },
        {
          id: "recommendation-systems",
          title: "Recommendation Systems",
          content: `
### Recommendation Architectures

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                RECOMMENDATION APPROACHES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   COLLABORATIVE FILTERING          CONTENT-BASED               │
│   ──────────────────────           ─────────────                │
│   "Users like you also liked..."   "Similar to what you liked" │
│                                                                 │
│   ┌─────┐   ┌─────┐               ┌─────┐   ┌─────┐            │
│   │User │───│Item │               │Item │───│Item │            │
│   │Matrix│   │Matrix│              │Features│ │Features│        │
│   └─────┘   └─────┘               └─────┘   └─────┘            │
│        ╲   ╱                            ╲   ╱                   │
│         ╲ ╱                              ╲ ╱                    │
│       Similarity                      Similarity                │
│                                                                 │
│   HYBRID APPROACH                                               │
│   ──────────────                                                │
│   Combines both + contextual signals                            │
│   (time, location, device, etc.)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Two-Tower Architecture

Modern recommendation systems use embedding-based retrieval:

\`\`\`python
class TwoTowerModel:
    """
    Separate towers for users and items.
    Enables precomputation and fast retrieval.
    """

    def __init__(self, embedding_dim=128):
        self.user_tower = UserEncoder(embedding_dim)
        self.item_tower = ItemEncoder(embedding_dim)

    def encode_user(self, user_features):
        return self.user_tower(user_features)  # [128]

    def encode_item(self, item_features):
        return self.item_tower(item_features)  # [128]

    def predict(self, user_emb, item_emb):
        return dot_product(user_emb, item_emb)

# At serving time:
# 1. Pre-compute all item embeddings
# 2. Compute user embedding online
# 3. Use ANN search for fast retrieval
\`\`\`

### Feed Ranking

For social feeds, news, content platforms:

\`\`\`python
def rank_feed_items(user, candidate_items):
    """Multi-objective feed ranking."""

    scores = []
    for item in candidate_items:
        # Engagement prediction
        p_click = click_model.predict(user, item)
        p_like = like_model.predict(user, item)
        p_share = share_model.predict(user, item)

        # Combined score with business weights
        score = (
            0.3 * p_click +
            0.4 * p_like +
            0.2 * p_share +
            0.1 * freshness_score(item)
        )

        # Diversity penalty (avoid too similar content)
        score -= diversity_penalty(item, already_shown)

        scores.append(score)

    return sorted(zip(candidate_items, scores),
                  key=lambda x: x[1], reverse=True)
\`\`\`

### Cold Start Problem

New users/items have no history:

| Problem | Solutions |
|---------|-----------|
| **New User** | Content-based, popularity, onboarding questions |
| **New Item** | Content features, creator history, early boosting |
| **Both** | Explore/exploit balance, contextual bandits |

### Recommendation Metrics

\`\`\`python
# Offline metrics
precision_at_k = relevant_recommended / k
recall_at_k = relevant_recommended / total_relevant
ndcg = normalized_discounted_cumulative_gain

# Online metrics (A/B testing)
ctr = clicks / impressions  # Click-through rate
conversion = purchases / clicks
dwell_time = average_time_on_item
\`\`\`
          `
        },
        {
          id: "model-serving",
          title: "Model Serving & Inference",
          content: `
### LLM Serving Challenges

LLMs are expensive to serve:

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    LLM SERVING BOTTLENECKS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Memory                    Compute                             │
│   ──────                    ───────                             │
│   7B model = 14GB (FP16)    Autoregressive = sequential         │
│   70B = 140GB!              KV cache grows with context         │
│                                                                 │
│   Latency                   Throughput                          │
│   ───────                   ──────────                          │
│   First token delay         Tokens/second                       │
│   Time per token            Concurrent requests                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Key Optimization Techniques

#### 1. Continuous Batching

\`\`\`python
# Traditional: Wait for all requests to finish
batch = collect_requests(timeout=100ms)
results = model.generate(batch)  # All finish together

# Continuous: Insert new requests as others complete
while True:
    # Add new requests to running batch
    if new_request and batch_has_capacity:
        batch.add(new_request)

    # Generate one step for all
    batch = model.step(batch)

    # Remove completed requests
    for req in batch.completed():
        respond(req)
        batch.remove(req)
\`\`\`

#### 2. PagedAttention (vLLM)

\`\`\`python
# Traditional: Contiguous KV cache per request
# Wastes memory on padding

# PagedAttention: Memory pages like OS
# - Allocate pages as needed
# - Share pages for common prefixes
# - 2-4x more throughput
\`\`\`

#### 3. Speculative Decoding

\`\`\`python
def speculative_decode(prompt, draft_model, target_model, k=4):
    """
    Use small model to draft, large model to verify.
    """
    # Draft k tokens with small model (fast)
    draft_tokens = draft_model.generate(prompt, k)

    # Verify with large model (parallel)
    probs = target_model.get_probs(prompt + draft_tokens)

    # Accept tokens that match
    accepted = []
    for i, token in enumerate(draft_tokens):
        if probs[i][token] > threshold:
            accepted.append(token)
        else:
            # Sample from adjusted distribution
            new_token = sample(probs[i])
            accepted.append(new_token)
            break

    return accepted
\`\`\`

### Serving Frameworks

| Framework | Best For | Key Features |
|-----------|----------|--------------|
| **vLLM** | High throughput | PagedAttention, continuous batching |
| **TGI** | Production HF | Tensor parallelism, quantization |
| **Ray Serve** | Custom logic | Python-native, autoscaling |
| **Triton** | Multi-model | NVIDIA optimized, ensemble |
| **Ollama** | Local dev | Simple setup, quantized models |

### Example: vLLM Deployment

\`\`\`python
# Install
pip install vllm

# Serve
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-hf \\
    --tensor-parallel-size 2 \\
    --max-num-seqs 256

# Client (OpenAI compatible)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
\`\`\`

### Caching Strategies

\`\`\`python
# Semantic cache for repeated queries
class SemanticCache:
    def __init__(self):
        self.cache = VectorStore()

    def get(self, query):
        # Find similar cached queries
        similar = self.cache.search(query, threshold=0.95)
        if similar:
            return similar[0].response
        return None

    def set(self, query, response):
        self.cache.add(query, response)

# Prefix caching for shared prompts
# System prompts, few-shot examples cached once
\`\`\`
          `
        },
        {
          id: "ml-system-design",
          title: "ML System Design Patterns",
          content: `
### Production ML Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                 PRODUCTION ML SYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │ Feature │    │  Model  │    │ Serving │                    │
│   │  Store  │───▶│ Registry│───▶│  Layer  │                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│        │              │              │                          │
│        │              │              │                          │
│   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐                    │
│   │Training │    │Version  │    │ Caching │                    │
│   │Pipeline │    │Control  │    │  Layer  │                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│        │              │              │                          │
│        └──────────────┼──────────────┘                          │
│                       │                                         │
│               ┌───────▼───────┐                                 │
│               │  Monitoring   │                                 │
│               │  & Alerting   │                                 │
│               └───────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Feature Store Pattern

\`\`\`python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Define features
user_features = store.get_feature_view("user_features")

# Training: Get historical features
training_df = store.get_historical_features(
    entity_df=training_entities,
    features=[
        "user_features:age",
        "user_features:purchase_count",
        "item_features:category_embedding"
    ]
)

# Serving: Get online features (low latency)
online_features = store.get_online_features(
    entity_rows=[{"user_id": 123}],
    features=["user_features:age"]
)
\`\`\`

### A/B Testing & Experimentation

\`\`\`python
class ABExperiment:
    def __init__(self, name, variants, traffic_split):
        self.name = name
        self.variants = variants
        self.split = traffic_split

    def get_variant(self, user_id):
        # Deterministic assignment
        bucket = hash(f"{self.name}:{user_id}") % 100
        cumulative = 0
        for variant, percentage in self.split.items():
            cumulative += percentage
            if bucket < cumulative:
                return variant

    def log_exposure(self, user_id, variant):
        # Track who saw what
        log_event("experiment_exposure", {
            "experiment": self.name,
            "variant": variant,
            "user_id": user_id
        })

# Usage
exp = ABExperiment(
    "new_model_v2",
    variants=["control", "treatment"],
    traffic_split={"control": 50, "treatment": 50}
)
\`\`\`

### Monitoring & Observability

\`\`\`python
# Key metrics to track
metrics = {
    # Latency
    "p50_latency_ms": percentile(latencies, 50),
    "p99_latency_ms": percentile(latencies, 99),

    # Throughput
    "requests_per_second": count / time_window,
    "tokens_per_second": total_tokens / time_window,

    # Errors
    "error_rate": errors / requests,
    "timeout_rate": timeouts / requests,

    # Model quality
    "prediction_distribution": histogram(predictions),
    "confidence_scores": mean(confidences),

    # Drift detection
    "feature_drift": ks_statistic(current, baseline),
    "prediction_drift": psi(current_preds, baseline_preds)
}

# Alert thresholds
if metrics["p99_latency_ms"] > 1000:
    alert("High latency!")
if metrics["error_rate"] > 0.01:
    alert("Error spike!")
\`\`\`

### Scaling Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Horizontal** | More requests | Add replicas |
| **Vertical** | Larger models | Bigger GPUs |
| **Model Parallelism** | Huge models | Split across GPUs |
| **Pipeline Parallelism** | High throughput | Async stages |
| **Caching** | Repeated queries | Redis, semantic cache |
          `
        },
        {
          id: "ethical-ai",
          title: "Ethical AI & Bias Mitigation",
          content: `
### Sources of Bias in ML

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                    BIAS IN ML PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Data Collection        Labeling           Model Training      │
│   ───────────────        ────────           ──────────────      │
│   Sampling bias          Annotator bias     Objective bias      │
│   Historical bias        Instruction bias   Optimization bias   │
│   Selection bias         Label noise        Representation      │
│                                                                 │
│   Deployment             Feedback Loop                          │
│   ──────────             ─────────────                          │
│   Context mismatch       Amplification                          │
│   User interaction       Confirmation                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Measuring Fairness

\`\`\`python
def evaluate_fairness(model, test_data, protected_attribute):
    """
    Calculate fairness metrics across groups.
    """
    groups = test_data.groupby(protected_attribute)

    metrics = {}
    for group_name, group_data in groups:
        preds = model.predict(group_data)
        labels = group_data['label']

        metrics[group_name] = {
            'accuracy': accuracy_score(labels, preds),
            'positive_rate': preds.mean(),
            'tpr': recall_score(labels, preds),  # True positive rate
            'fpr': false_positive_rate(labels, preds)
        }

    # Fairness checks
    results = {}

    # Demographic parity: Equal positive rates
    rates = [m['positive_rate'] for m in metrics.values()]
    results['demographic_parity_ratio'] = min(rates) / max(rates)

    # Equalized odds: Equal TPR and FPR
    tprs = [m['tpr'] for m in metrics.values()]
    results['equalized_odds_tpr'] = min(tprs) / max(tprs)

    return results
\`\`\`

### Mitigation Strategies

#### Pre-processing

\`\`\`python
# Rebalance training data
from imblearn.over_sampling import SMOTE

def balance_dataset(X, y, protected_attr):
    # Oversample underrepresented groups
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced
\`\`\`

#### In-processing

\`\`\`python
# Add fairness constraint to loss
def fairness_regularized_loss(predictions, labels, groups, lambda_fair=0.1):
    base_loss = cross_entropy(predictions, labels)

    # Demographic parity penalty
    group_rates = []
    for g in groups.unique():
        mask = groups == g
        group_rates.append(predictions[mask].mean())

    fairness_penalty = variance(group_rates)

    return base_loss + lambda_fair * fairness_penalty
\`\`\`

#### Post-processing

\`\`\`python
# Adjust thresholds per group
def calibrate_thresholds(model, val_data, target_rate):
    thresholds = {}
    for group in val_data['group'].unique():
        group_data = val_data[val_data['group'] == group]
        probs = model.predict_proba(group_data)[:, 1]

        # Find threshold that achieves target rate
        threshold = find_threshold(probs, target_rate)
        thresholds[group] = threshold

    return thresholds
\`\`\`

### LLM-Specific Considerations

\`\`\`python
# Safety evaluation
safety_checks = [
    "Generate harmful content",
    "Reveal private information",
    "Show demographic bias",
    "Produce misinformation"
]

for check in safety_checks:
    responses = model.generate_many(check, n=100)
    violations = count_violations(responses)
    print(f"{check}: {violations/100:.1%} violation rate")

# Red teaming
def red_team_evaluation(model, attack_prompts):
    """
    Test model against adversarial prompts.
    """
    results = []
    for prompt in attack_prompts:
        response = model.generate(prompt)
        is_safe = safety_classifier(response)
        results.append({
            'prompt': prompt,
            'response': response,
            'safe': is_safe
        })
    return results
\`\`\`

### Responsible AI Checklist

- [ ] **Data**: Audit for representation and historical bias
- [ ] **Metrics**: Measure performance across demographic groups
- [ ] **Testing**: Red team with adversarial prompts
- [ ] **Transparency**: Document model limitations
- [ ] **Monitoring**: Track fairness metrics in production
- [ ] **Feedback**: Mechanism for users to report issues
- [ ] **Updates**: Plan for model updates and corrections
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building Production Systems",
      estimatedTime: "3-4 hours",
      tasks: [
        {
          id: "task-1",
          title: "Deploy with vLLM",
          description: "Set up high-performance LLM serving",
          content: `
### Install vLLM

\`\`\`bash
pip install vllm
\`\`\`

### Start Server

\`\`\`bash
python -m vllm.entrypoints.openai.api_server \\
    --model microsoft/phi-2 \\
    --host 0.0.0.0 \\
    --port 8000
\`\`\`

### Test Endpoint

\`\`\`python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="microsoft/phi-2",
    messages=[
        {"role": "user", "content": "What is Python?"}
    ]
)

print(response.choices[0].message.content)
\`\`\`

### Benchmark Performance

\`\`\`python
import time
import asyncio
from openai import AsyncOpenAI

async def benchmark(n_requests=100):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1")

    start = time.time()

    tasks = [
        client.chat.completions.create(
            model="microsoft/phi-2",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=50
        )
        for _ in range(n_requests)
    ]

    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"Completed {n_requests} requests in {elapsed:.2f}s")
    print(f"Throughput: {n_requests/elapsed:.1f} req/s")

asyncio.run(benchmark())
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "Build Semantic Cache",
          description: "Implement caching for repeated queries",
          content: `
### Create cache.py

\`\`\`python
"""Semantic cache for LLM responses."""
import hashlib
from typing import Optional
import chromadb
from openai import OpenAI

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("cache")
        self.threshold = similarity_threshold
        self.openai = OpenAI()

    def _get_embedding(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def get(self, query: str) -> Optional[str]:
        """Check cache for similar query."""
        results = self.collection.query(
            query_embeddings=[self._get_embedding(query)],
            n_results=1
        )

        if not results['distances'][0]:
            return None

        # Lower distance = more similar
        # Convert to similarity score
        distance = results['distances'][0][0]
        similarity = 1 / (1 + distance)

        if similarity >= self.threshold:
            return results['metadatas'][0][0]['response']

        return None

    def set(self, query: str, response: str):
        """Store query-response pair."""
        query_id = hashlib.md5(query.encode()).hexdigest()

        self.collection.add(
            ids=[query_id],
            embeddings=[self._get_embedding(query)],
            metadatas=[{"query": query, "response": response}]
        )

# Usage
cache = SemanticCache(similarity_threshold=0.92)

def get_completion(prompt: str) -> str:
    # Check cache first
    cached = cache.get(prompt)
    if cached:
        print("Cache hit!")
        return cached

    # Generate new response
    response = llm.generate(prompt)

    # Store in cache
    cache.set(prompt, response)

    return response
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "Add Monitoring",
          description: "Implement observability for ML systems",
          content: `
### Create monitoring.py

\`\`\`python
"""ML system monitoring."""
import time
from dataclasses import dataclass, field
from typing import List
from collections import deque
import statistics

@dataclass
class MetricsCollector:
    window_size: int = 1000
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    errors: int = 0
    requests: int = 0
    tokens_generated: int = 0

    def record_request(self, latency_ms: float, tokens: int, error: bool = False):
        self.latencies.append(latency_ms)
        self.requests += 1
        self.tokens_generated += tokens
        if error:
            self.errors += 1

    def get_metrics(self) -> dict:
        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)
        return {
            "p50_latency_ms": statistics.median(sorted_latencies),
            "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.99)],
            "error_rate": self.errors / self.requests if self.requests else 0,
            "total_requests": self.requests,
            "tokens_generated": self.tokens_generated
        }

# Usage
metrics = MetricsCollector()

def monitored_generate(prompt: str) -> str:
    start = time.time()
    error = False

    try:
        response = model.generate(prompt)
        tokens = count_tokens(response)
    except Exception as e:
        error = True
        tokens = 0
        raise

    finally:
        latency_ms = (time.time() - start) * 1000
        metrics.record_request(latency_ms, tokens, error)

    return response

# Periodic reporting
def report_metrics():
    m = metrics.get_metrics()
    print(f"P99 Latency: {m['p99_latency_ms']:.0f}ms")
    print(f"Error Rate: {m['error_rate']:.2%}")
    print(f"Throughput: {m['tokens_generated']} tokens")
\`\`\`

### Docker Deployment

\`\`\`dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "microsoft/phi-2", \\
     "--host", "0.0.0.0"]
\`\`\`

\`\`\`yaml
# docker-compose.yml
version: '3.8'
services:
  llm-server:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
\`\`\`

🎉 **Congratulations!** You've learned production ML system design!
          `
        }
      ]
    }
  ]
};

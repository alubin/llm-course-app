export const day7Content = {
  id: 7,
  title: "Data Pipeline with AI Enrichment",
  subtitle: "Build production data pipelines with AI classification and transformation",
  duration: "6-8 hours",
  difficulty: "Advanced",

  objectives: [
    "Design batch and streaming data pipelines",
    "Implement AI-powered data enrichment and classification",
    "Build robust error handling and retry logic",
    "Create data validation and quality checks",
    "Set up monitoring and observability",
    "Deploy production-ready pipelines"
  ],

  prerequisites: [
    { name: "Python", details: "Advanced Python including async, generators" },
    { name: "Data Processing", details: "Understanding of ETL concepts" },
    { name: "Previous Days", details: "Completed Days 1-3" }
  ],

  technologies: [
    { name: "Pandas", purpose: "Data manipulation and analysis" },
    { name: "Apache Arrow/Parquet", purpose: "Efficient data storage" },
    { name: "OpenAI/Anthropic", purpose: "AI enrichment" },
    { name: "Prefect/Airflow", purpose: "Pipeline orchestration (optional)" },
    { name: "Prometheus", purpose: "Metrics collection" },
    { name: "SQLAlchemy", purpose: "Database operations" }
  ],

  sections: [
    {
      id: "theory",
      title: "Part 1: Theory — Data Pipelines",
      estimatedTime: "2-2.5 hours",
      modules: [
        {
          id: "pipeline-fundamentals",
          title: "Data Pipeline Fundamentals",
          content: `
### What is a Data Pipeline?

A data pipeline moves and transforms data from sources to destinations:

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐            │
│   │  INGEST  │ ───► │ TRANSFORM│ ───► │  LOAD    │            │
│   └──────────┘      └──────────┘      └──────────┘            │
│        │                  │                  │                  │
│        │                  │                  │                  │
│   • CSV files        • Clean data       • Database             │
│   • APIs             • Enrich with AI   • Data warehouse       │
│   • Databases        • Validate         • Object storage       │
│   • Streams          • Transform        • Analytics            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Batch vs Streaming

**Batch Processing:**
- Process data in chunks at intervals
- Example: Process all customer reviews once per day
- Tools: Pandas, Spark

**Streaming Processing:**
- Process data as it arrives in real-time
- Example: Classify support tickets as they're created
- Tools: Kafka, Flink

### ETL vs ELT

**ETL (Extract, Transform, Load):**
\`\`\`
Source → Transform (clean, enrich) → Load to warehouse
\`\`\`

**ELT (Extract, Load, Transform):**
\`\`\`
Source → Load raw → Transform in warehouse
\`\`\`

### AI in Data Pipelines

Common use cases:

1. **Classification** - Categorize data (e.g., email priority, ticket urgency)
2. **Extraction** - Pull structured data from unstructured text
3. **Enrichment** - Add context (e.g., sentiment, topics)
4. **Validation** - Detect anomalies and quality issues
5. **Summarization** - Condense long documents

### Example: Customer Feedback Pipeline

\`\`\`python
# Input
feedback = {
    "id": 123,
    "text": "The product arrived damaged and support was unhelpful",
    "timestamp": "2024-01-15T10:30:00Z"
}

# AI Enrichment
enriched = {
    **feedback,
    "sentiment": "negative",       # AI classification
    "category": "product_quality", # AI classification
    "priority": "high",            # AI-derived
    "summary": "Damaged product, poor support",  # AI-generated
    "action_items": ["Replacement", "Support follow-up"]  # AI-extracted
}
\`\`\`
          `
        },
        {
          id: "error-handling",
          title: "Error Handling & Retry Logic",
          content: `
### Types of Errors

1. **Transient** - Temporary (network issues, rate limits)
   - **Solution**: Retry with exponential backoff

2. **Permanent** - Will always fail (invalid data, auth error)
   - **Solution**: Skip, log, move to dead letter queue

3. **Partial** - Some records fail, others succeed
   - **Solution**: Process in batches, track failures

### Retry Strategy

\`\`\`python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1):
    """Decorator for exponential backoff retry."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise  # Last attempt, re-raise

                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    print(f"Retry {attempt + 1}/{max_retries} after {delay}s")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, base_delay=2)
def call_llm_api(text):
    """Call LLM with automatic retries."""
    return client.chat(text)
\`\`\`

### Circuit Breaker Pattern

Stop calling a failing service:

\`\`\`python
class CircuitBreaker:
    """Prevents calling failing services."""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
\`\`\`

### Dead Letter Queue

Store failed records for later review:

\`\`\`python
import json
from datetime import datetime

class DeadLetterQueue:
    """Store failed processing attempts."""

    def __init__(self, file_path="dlq.jsonl"):
        self.file_path = file_path

    def add(self, record, error, context=None):
        """Add failed record to DLQ."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "record": record,
            "error": str(error),
            "context": context or {}
        }

        with open(self.file_path, 'a') as f:
            f.write(json.dumps(entry) + '\\n')

# Usage
dlq = DeadLetterQueue()

try:
    process_record(record)
except Exception as e:
    dlq.add(record, e, context={"step": "ai_enrichment"})
\`\`\`
          `
        },
        {
          id: "monitoring",
          title: "Monitoring & Observability",
          content: `
### Key Metrics to Track

1. **Throughput** - Records processed per second
2. **Latency** - Time per record
3. **Error Rate** - Failed records percentage
4. **Cost** - API calls, compute costs
5. **Data Quality** - Validation pass rate

### Metrics Collection

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    total_records: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = None
    end_time: float = None
    api_calls: int = 0
    total_tokens: int = 0

    def record_success(self):
        self.successful += 1
        self.total_records += 1

    def record_failure(self):
        self.failed += 1
        self.total_records += 1

    def record_api_call(self, tokens):
        self.api_calls += 1
        self.total_tokens += tokens

    @property
    def duration_seconds(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    @property
    def throughput(self):
        """Records per second."""
        if self.duration_seconds > 0:
            return self.total_records / self.duration_seconds
        return 0

    @property
    def error_rate(self):
        """Percentage of failed records."""
        if self.total_records > 0:
            return (self.failed / self.total_records) * 100
        return 0

    def summary(self):
        return {
            "total_records": self.total_records,
            "successful": self.successful,
            "failed": self.failed,
            "duration_seconds": round(self.duration_seconds, 2),
            "throughput": round(self.throughput, 2),
            "error_rate": round(self.error_rate, 2),
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens
        }
\`\`\`

### Logging Best Practices

\`\`\`python
import logging
import json

# Structured logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log_pipeline_event(event_type, record_id, **kwargs):
    """Log structured event."""
    log_entry = {
        "event": event_type,
        "record_id": record_id,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    logger.info(json.dumps(log_entry))

# Usage
log_pipeline_event("processing_start", record_id=123)
log_pipeline_event("ai_enrichment", record_id=123, tokens=150, latency_ms=320)
log_pipeline_event("processing_complete", record_id=123, status="success")
\`\`\`
          `
        }
      ]
    },
    {
      id: "hands-on",
      title: "Part 2: Hands-On — Building the Pipeline",
      estimatedTime: "3.5-5.5 hours",
      tasks: [
        {
          id: "task-1",
          title: "Project Setup",
          description: "Initialize data pipeline project",
          content: `
### Create Project Structure

\`\`\`bash
mkdir ai-data-pipeline && cd ai-data-pipeline
mkdir -p {pipeline/{ingestion,enrichment,validation,output},data/{raw,processed,failed},config,tests}
touch pipeline/__init__.py
touch pipeline/{models.py,enricher.py,validator.py,orchestrator.py}
touch main.py config.py
\`\`\`

### Install Dependencies

\`\`\`bash
pip install pandas openai anthropic pydantic sqlalchemy python-dotenv
pip install pyarrow  # For parquet support
\`\`\`

### Create requirements.txt

\`\`\`txt
pandas>=2.0.0
openai>=1.0.0
anthropic>=0.18.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
pyarrow>=14.0.0
\`\`\`

### Create .env

\`\`\`bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
BATCH_SIZE=10
MAX_WORKERS=5
\`\`\`
          `
        },
        {
          id: "task-2",
          title: "Data Models",
          description: "Define data structures with Pydantic",
          content: `
### Create pipeline/models.py

\`\`\`python
"""Data models for the pipeline."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator

class InputRecord(BaseModel):
    """Raw input record."""
    id: str
    text: str
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v

class EnrichedRecord(BaseModel):
    """Record after AI enrichment."""
    id: str
    text: str
    source: str
    timestamp: datetime

    # AI-generated fields
    sentiment: Optional[str] = None  # positive, negative, neutral
    category: Optional[str] = None
    priority: Optional[str] = None   # low, medium, high
    summary: Optional[str] = None
    tags: List[str] = []
    confidence_scores: dict = {}

    # Metadata
    enriched_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None

class ProcessingResult(BaseModel):
    """Result of processing a batch."""
    successful: List[EnrichedRecord] = []
    failed: List[dict] = []  # {record, error}
    metrics: dict = {}
\`\`\`
          `
        },
        {
          id: "task-3",
          title: "AI Enricher",
          description: "Build the AI enrichment component",
          content: `
### Create pipeline/enricher.py

\`\`\`python
"""AI enrichment service."""
import json
import time
from typing import Optional
from openai import OpenAI
from pipeline.models import InputRecord, EnrichedRecord

class AIEnricher:
    """Enriches records with AI-generated metadata."""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI()
        self.model = model

    def enrich(self, record: InputRecord) -> EnrichedRecord:
        """
        Enrich a single record with AI analysis.

        Returns:
            EnrichedRecord with AI-generated fields
        """
        start_time = time.time()

        # Build enrichment prompt
        prompt = f"""Analyze the following text and provide:
1. Sentiment (positive, negative, or neutral)
2. Category (e.g., product_feedback, support_request, feature_request, bug_report)
3. Priority (low, medium, or high)
4. Summary (one sentence)
5. Tags (2-4 relevant keywords)

Text: {record.text}

Respond in JSON format:
{{
  "sentiment": "...",
  "category": "...",
  "priority": "...",
  "summary": "...",
  "tags": ["tag1", "tag2"],
  "confidence_scores": {{
    "sentiment": 0.95,
    "category": 0.87
  }}
}}"""

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower for consistency
        )

        # Parse response
        ai_result = json.loads(response.choices[0].message.content)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        # Build enriched record
        enriched = EnrichedRecord(
            id=record.id,
            text=record.text,
            source=record.source,
            timestamp=record.timestamp,
            sentiment=ai_result.get("sentiment"),
            category=ai_result.get("category"),
            priority=ai_result.get("priority"),
            summary=ai_result.get("summary"),
            tags=ai_result.get("tags", []),
            confidence_scores=ai_result.get("confidence_scores", {}),
            processing_time_ms=processing_time,
            tokens_used=response.usage.total_tokens
        )

        return enriched

    def enrich_batch(self, records: list[InputRecord]) -> list[EnrichedRecord]:
        """Enrich a batch of records."""
        enriched_records = []

        for record in records:
            try:
                enriched = self.enrich(record)
                enriched_records.append(enriched)
            except Exception as e:
                print(f"Error enriching record {record.id}: {e}")
                # Could add to DLQ here
                continue

        return enriched_records
\`\`\`
          `
        },
        {
          id: "task-4",
          title: "Data Validator",
          description: "Implement validation and quality checks",
          content: `
### Create pipeline/validator.py

\`\`\`python
"""Data validation service."""
from typing import List, Tuple
from pipeline.models import EnrichedRecord

class DataValidator:
    """Validates enriched data quality."""

    def __init__(self):
        self.valid_sentiments = ["positive", "negative", "neutral"]
        self.valid_priorities = ["low", "medium", "high"]
        self.min_confidence = 0.5

    def validate(self, record: EnrichedRecord) -> Tuple[bool, List[str]]:
        """
        Validate an enriched record.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check sentiment
        if record.sentiment not in self.valid_sentiments:
            errors.append(f"Invalid sentiment: {record.sentiment}")

        # Check priority
        if record.priority not in self.valid_priorities:
            errors.append(f"Invalid priority: {record.priority}")

        # Check confidence scores
        for field, score in record.confidence_scores.items():
            if score < self.min_confidence:
                errors.append(f"Low confidence for {field}: {score}")

        # Check summary exists and is reasonable length
        if not record.summary or len(record.summary) < 10:
            errors.append("Summary too short or missing")

        # Check tags
        if not record.tags or len(record.tags) == 0:
            errors.append("No tags provided")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_batch(self, records: List[EnrichedRecord]) -> dict:
        """
        Validate a batch of records.

        Returns:
            {
                "valid": [records],
                "invalid": [(record, errors)],
                "validation_rate": 0.95
            }
        """
        valid = []
        invalid = []

        for record in records:
            is_valid, errors = self.validate(record)
            if is_valid:
                valid.append(record)
            else:
                invalid.append((record, errors))

        validation_rate = len(valid) / len(records) if records else 0

        return {
            "valid": valid,
            "invalid": invalid,
            "validation_rate": validation_rate
        }
\`\`\`
          `
        },
        {
          id: "task-5",
          title: "Pipeline Orchestrator",
          description: "Build the main pipeline orchestrator",
          content: `
### Create pipeline/orchestrator.py

\`\`\`python
"""Pipeline orchestration."""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List
from pipeline.models import InputRecord, EnrichedRecord, ProcessingResult
from pipeline.enricher import AIEnricher
from pipeline.validator import DataValidator

class DataPipeline:
    """Main data pipeline orchestrator."""

    def __init__(self):
        self.enricher = AIEnricher()
        self.validator = DataValidator()
        self.dlq_path = Path("data/failed")
        self.dlq_path.mkdir(parents=True, exist_ok=True)

    def load_input_data(self, file_path: str) -> List[InputRecord]:
        """Load data from CSV/JSON."""
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_json(file_path)

        records = [
            InputRecord(
                id=str(row.get('id', i)),
                text=row['text'],
                source=row.get('source', 'unknown'),
                timestamp=pd.to_datetime(row.get('timestamp', datetime.utcnow()))
            )
            for i, row in df.iterrows()
        ]

        return records

    def process_batch(self, records: List[InputRecord]) -> ProcessingResult:
        """Process a batch of records through the pipeline."""
        result = ProcessingResult()

        print(f"Processing {len(records)} records...")

        # Step 1: Enrich with AI
        print("Step 1: AI Enrichment...")
        enriched = []
        for record in records:
            try:
                enriched_record = self.enricher.enrich(record)
                enriched.append(enriched_record)
            except Exception as e:
                result.failed.append({
                    "record": record.dict(),
                    "error": str(e),
                    "step": "enrichment"
                })

        # Step 2: Validate
        print("Step 2: Validation...")
        validation_result = self.validator.validate_batch(enriched)

        result.successful = validation_result["valid"]

        # Add validation failures to failed
        for record, errors in validation_result["invalid"]:
            result.failed.append({
                "record": record.dict(),
                "error": f"Validation failed: {errors}",
                "step": "validation"
            })

        # Step 3: Calculate metrics
        total_tokens = sum(r.tokens_used for r in result.successful if r.tokens_used)
        avg_processing_time = sum(r.processing_time_ms for r in result.successful if r.processing_time_ms) / len(result.successful) if result.successful else 0

        result.metrics = {
            "total_records": len(records),
            "successful": len(result.successful),
            "failed": len(result.failed),
            "validation_rate": validation_result["validation_rate"],
            "total_tokens": total_tokens,
            "avg_processing_time_ms": round(avg_processing_time, 2)
        }

        return result

    def save_results(self, result: ProcessingResult, output_path: str):
        """Save successful records and failed records."""
        # Save successful records
        if result.successful:
            df = pd.DataFrame([r.dict() for r in result.successful])
            df.to_parquet(output_path, index=False)
            print(f"✓ Saved {len(result.successful)} records to {output_path}")

        # Save failed records to DLQ
        if result.failed:
            dlq_file = self.dlq_path / f"failed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            pd.DataFrame(result.failed).to_json(dlq_file, orient='records', lines=True)
            print(f"✗ Saved {len(result.failed)} failed records to {dlq_file}")

    def run(self, input_path: str, output_path: str):
        """Run the complete pipeline."""
        print("=" * 60)
        print("DATA PIPELINE STARTED")
        print("=" * 60)

        # Load data
        records = self.load_input_data(input_path)
        print(f"Loaded {len(records)} records from {input_path}")

        # Process
        result = self.process_batch(records)

        # Save
        self.save_results(result, output_path)

        # Print metrics
        print("\\n" + "=" * 60)
        print("PIPELINE METRICS")
        print("=" * 60)
        for key, value in result.metrics.items():
            print(f"{key}: {value}")

        print("\\nPIPELINE COMPLETE")
\`\`\`
          `
        },
        {
          id: "task-6",
          title: "CLI Interface & Testing",
          description: "Create CLI and test the pipeline",
          content: `
### Create main.py

\`\`\`python
"""CLI for data pipeline."""
import argparse
from pipeline.orchestrator import DataPipeline

def main():
    parser = argparse.ArgumentParser(description="AI Data Pipeline")
    parser.add_argument("input", help="Input file (CSV or JSON)")
    parser.add_argument("output", help="Output parquet file")

    args = parser.parse_args()

    # Run pipeline
    pipeline = DataPipeline()
    pipeline.run(args.input, args.output)

if __name__ == "__main__":
    main()
\`\`\`

### Create Sample Data

\`\`\`bash
cat > data/raw/sample.csv << 'EOF'
id,text,source
1,"The product quality is amazing! Very happy with my purchase.",customer_review
2,"Delivery was delayed by 2 weeks. Very frustrated.",customer_feedback
3,"Feature request: Add dark mode to the mobile app",feature_request
4,"Critical bug: App crashes on startup for Android 14",bug_report
5,"Great customer service. Agent was very helpful.",support_feedback
EOF
\`\`\`

### Run the Pipeline

\`\`\`bash
python main.py data/raw/sample.csv data/processed/enriched.parquet
\`\`\`

### Expected Output

\`\`\`
============================================================
DATA PIPELINE STARTED
============================================================
Loaded 5 records from data/raw/sample.csv
Processing 5 records...
Step 1: AI Enrichment...
Step 2: Validation...
✓ Saved 5 records to data/processed/enriched.parquet

============================================================
PIPELINE METRICS
============================================================
total_records: 5
successful: 5
failed: 0
validation_rate: 1.0
total_tokens: 1250
avg_processing_time_ms: 423.5

PIPELINE COMPLETE
\`\`\`

### Analyze Results

\`\`\`python
import pandas as pd

# Load results
df = pd.read_parquet('data/processed/enriched.parquet')

print(df[['text', 'sentiment', 'category', 'priority', 'summary']])

# Aggregate statistics
print("\\nSentiment Distribution:")
print(df['sentiment'].value_counts())

print("\\nCategory Distribution:")
print(df['category'].value_counts())
\`\`\`

🎉 **Congratulations!** You've built a production data pipeline with AI!
          `
        }
      ]
    }
  ]
};

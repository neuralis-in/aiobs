# Benchmark: HallucinationDetectionEval vs HaluEvalQA

Benchmark script to evaluate `HallucinationDetectionEval` against the HaluEvalQA dataset (10,000 QA samples).

## Quick Start

```bash
# Install dependencies
pip install openai google-genai

# Set API key
export OPENAI_API_KEY="your-key-here"
# or
export GEMINI_API_KEY="your-key-here"

# Run benchmark
python benchmark/halueval_qa.py --sample-size 100
```

## Usage

```bash
# Quick test with 10 samples
python benchmark/halueval_qa.py --sample-size 10

# Full benchmark with GPT-4o
python benchmark/halueval_qa.py --judge-model gpt-4o --provider openai

# With Gemini
python benchmark/halueval_qa.py --provider gemini --judge-model gemini-2.0-flash
```

## Command-Line Options

```
--judge-model MODEL     Judge model name (default: gpt-4o-mini)
--provider PROVIDER     LLM provider: openai, gemini (default: openai)
--sample-size N         Number of samples to evaluate (default: all)
--temperature T         Temperature for judge LLM (default: 0.0)
--output PATH           Output JSON file path (default: auto-generated)
```

## Metrics

The benchmark calculates:
- Accuracy, Precision, Recall, F1 Score
- True Positive Rate (TPR), False Positive Rate (FPR)
- Score distributions for hallucinated vs correct answers

Results are saved to `benchmark/data/results_*.json` and displayed in the console.

## Gemini Service Account

For Gemini with service account JSON:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_REGION="us-central1"
pip install google-genai google-auth
```

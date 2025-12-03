# aiobs.evals Examples

Minimal examples demonstrating each evaluator in the `aiobs.evals` module.

## Quick Start

```bash
cd example/evals
python regex_assertion_example.py
```

## Examples

| File | Evaluator | Purpose |
|------|-----------|---------|
| `regex_assertion_example.py` | `RegexAssertion` | Check output matches regex patterns |
| `schema_assertion_example.py` | `SchemaAssertion` | Validate JSON output against schema |
| `ground_truth_example.py` | `GroundTruthEval` | Compare output to expected answer |
| `latency_consistency_example.py` | `LatencyConsistencyEval` | Check latency statistics |
| `pii_detection_example.py` | `PIIDetectionEval` | Detect PII in outputs |

## Common Pattern

All evaluators follow the same pattern:

```python
from aiobs.evals import EvalInput, SomeEvaluator

# 1. Create evaluator with config
evaluator = SomeEvaluator.from_config(...)

# 2. Create input
eval_input = EvalInput(
    user_input="User's question",
    model_output="Model's response",
    system_prompt="Optional system prompt",
)

# 3. Run evaluation
result = evaluator(eval_input)

# 4. Check result
print(result.status)  # EvalStatus.PASSED or EvalStatus.FAILED
print(result.score)   # 0.0 to 1.0
print(result.message) # Human-readable explanation
```

## Batch Evaluation

```python
inputs = [EvalInput(...), EvalInput(...), EvalInput(...)]
results = evaluator.evaluate_batch(inputs)
```


"""Minimal example: LatencyConsistencyEval evaluator.

Checks latency statistics across multiple runs.
"""

from aiobs.evals import EvalInput, LatencyConsistencyEval

# Simulated latency data from multiple runs (in milliseconds)
latencies = [120, 135, 128, 142, 115, 138, 125, 180, 122, 130]

# Create evaluator with thresholds
evaluator = LatencyConsistencyEval.with_thresholds(
    max_latency_ms=200,       # Max single latency
    max_p95_ms=180,           # 95th percentile threshold
    cv_threshold=0.3,         # Coefficient of variation threshold
)

# Create input with latency data in metadata
eval_input = EvalInput(
    user_input="Test query",
    model_output="Test response",
    metadata={"latencies": latencies},
)

# Run evaluation
result = evaluator(eval_input)

print(f"Status: {result.status.value}")
print(f"Score: {result.score}")
print(f"Message: {result.message}")

# Print statistics
if result.details:
    print("\nLatency Statistics:")
    print(f"  Count: {result.details['count']}")
    print(f"  Mean: {result.details['mean']:.2f} ms")
    print(f"  Std Dev: {result.details['std_dev']:.2f} ms")
    print(f"  Min: {result.details['min']:.2f} ms")
    print(f"  Max: {result.details['max']:.2f} ms")
    print(f"  P95: {result.details['p95']:.2f} ms")
    print(f"  P99: {result.details['p99']:.2f} ms")
    print(f"  CV: {result.details['cv']:.4f}")

# --- Example: High variance latencies (should fail) ---
print("\n--- High Variance Example ---")

high_variance_latencies = [50, 500, 80, 450, 100, 400, 75, 480]

result = evaluator(
    EvalInput(user_input="q", model_output="r"),
    latencies=high_variance_latencies,  # Pass via kwargs
)

print(f"High variance - Status: {result.status.value}")
print(f"CV: {result.details['cv']:.4f} (threshold: 0.3)")


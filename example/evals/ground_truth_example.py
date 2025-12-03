"""Minimal example: GroundTruthEval evaluator.

Compares model output against expected ground truth.
"""

from aiobs.evals import EvalInput, GroundTruthEval

# --- Exact Match ---
print("--- Exact Match ---")

exact_eval = GroundTruthEval.exact(case_sensitive=False)

result = exact_eval(EvalInput(
    user_input="What is 2+2?",
    model_output="4",
    expected_output="4",
))
print(f"Exact match '4' == '4': {result.status.value}")

result = exact_eval(EvalInput(
    user_input="What is 2+2?",
    model_output="The answer is 4",
    expected_output="4",
))
print(f"Exact match 'The answer is 4' == '4': {result.status.value}")

# --- Contains Match ---
print("\n--- Contains Match ---")

contains_eval = GroundTruthEval.contains(case_sensitive=False)

result = contains_eval(EvalInput(
    user_input="Capital of France?",
    model_output="The capital of France is Paris, a beautiful city.",
    expected_output="Paris",
))
print(f"Contains 'Paris': {result.status.value}")

# --- Normalized Match ---
print("\n--- Normalized Match ---")

normalized_eval = GroundTruthEval.normalized(
    case_sensitive=False,
    strip_punctuation=True,
)

result = normalized_eval(EvalInput(
    user_input="Who wrote Hamlet?",
    model_output="WILLIAM   SHAKESPEARE!",
    expected_output="William Shakespeare",
))
print(f"Normalized match: {result.status.value}")
print(f"Score: {result.score}")

# --- Using kwargs for expected ---
print("\n--- Expected from kwargs ---")

result = contains_eval(
    EvalInput(
        user_input="What color is the sky?",
        model_output="The sky is blue during the day.",
    ),
    expected="blue",  # Pass expected via kwargs
)
print(f"Contains 'blue': {result.status.value}")


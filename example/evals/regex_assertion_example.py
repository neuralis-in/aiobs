"""Minimal example: RegexAssertion evaluator.

Checks if model output matches (or doesn't match) regex patterns.
"""

from aiobs.evals import EvalInput, RegexAssertion

# Create evaluator with patterns that MUST match
evaluator = RegexAssertion.from_patterns(
    patterns=[r"Paris", r"\d+"],  # Must contain "Paris" and a number
    match_mode="all",  # All patterns must match
    case_sensitive=False,
)

# Test input
eval_input = EvalInput(
    user_input="What is the population of Paris?",
    model_output="Paris has a population of about 2.1 million people.",
)

# Run evaluation
result = evaluator(eval_input)

print(f"Status: {result.status.value}")
print(f"Score: {result.score}")
print(f"Message: {result.message}")

# --- Example with negative patterns ---

print("\n--- Negative Pattern Example ---")

# Evaluator that fails if output contains apology words
no_apology_eval = RegexAssertion.from_patterns(
    negative_patterns=[r"\b(sorry|cannot|unable|apologize)\b"],
    case_sensitive=False,
)

good_response = EvalInput(
    user_input="Help me with X",
    model_output="Here's how to do X: step 1, step 2, step 3.",
)

bad_response = EvalInput(
    user_input="Help me with X",
    model_output="Sorry, I cannot help with that request.",
)

print(f"Good response: {no_apology_eval(good_response).status.value}")
print(f"Bad response: {no_apology_eval(bad_response).status.value}")


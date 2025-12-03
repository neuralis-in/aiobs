"""Minimal example: PIIDetectionEval evaluator.

Detects personally identifiable information in model outputs.
"""

from aiobs.evals import EvalInput, PIIDetectionEval, PIIType, PIIDetectionConfig

# Create default PII detector (email, phone, SSN, credit card)
evaluator = PIIDetectionEval.default()

# --- Test clean output ---
print("--- Clean Output ---")

clean_input = EvalInput(
    user_input="Tell me about Python",
    model_output="Python is a programming language created by Guido van Rossum.",
)

result = evaluator(clean_input)
print(f"Status: {result.status.value}")
print(f"PII found: {result.details['pii_count']}")

# --- Test output with PII ---
print("\n--- Output with PII ---")

pii_input = EvalInput(
    user_input="What's your contact info?",
    model_output="You can reach me at john.doe@example.com or call 555-123-4567.",
)

result = evaluator(pii_input)
print(f"Status: {result.status.value}")
print(f"PII found: {result.details['pii_count']}")
print(f"Types: {result.details['pii_types_found']}")

# --- Scan and redact ---
print("\n--- Scan & Redact ---")

text = "My SSN is 123-45-6789 and my card is 4111111111111111"

# Scan for PII
matches = evaluator.scan(text)
for match in matches:
    print(f"Found {match.pii_type}: {match.value}")

# Redact PII
redacted = evaluator.redact(text)
print(f"\nRedacted: {redacted}")

# --- Custom patterns ---
print("\n--- Custom Patterns ---")

custom_eval = PIIDetectionEval.with_custom_patterns({
    "employee_id": r"EMP-\d{6}",
    "internal_code": r"INT-[A-Z]{3}-\d{4}",
})

result = custom_eval(EvalInput(
    user_input="Get employee info",
    model_output="Employee EMP-123456 has code INT-ABC-1234",
))

print(f"Status: {result.status.value}")
print(f"Custom PII found: {result.details['pii_types_found']}")

# --- Strict mode (checks input and system prompt too) ---
print("\n--- Strict Mode ---")

strict_eval = PIIDetectionEval.strict()

result = strict_eval(EvalInput(
    user_input="My email is user@example.com, help me",
    model_output="Sure, I can help!",
    system_prompt="You are an assistant.",
))

print(f"Strict check - Status: {result.status.value}")
print(f"Checked fields: {result.details['checked_fields']}")


"""Minimal example: SchemaAssertion evaluator.

Validates that model output is valid JSON matching a schema.
"""

from aiobs.evals import EvalInput, SchemaAssertion

# Define JSON schema for expected output
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age"],
}

# Create evaluator
evaluator = SchemaAssertion.from_schema(schema)

# Test with valid JSON output
valid_input = EvalInput(
    user_input="Extract person info from: John is 30 years old",
    model_output='{"name": "John", "age": 30}',
)

result = evaluator(valid_input)
print(f"Valid JSON - Status: {result.status.value}, Score: {result.score}")

# Test with invalid JSON
invalid_input = EvalInput(
    user_input="Extract person info",
    model_output="name: John, age: thirty",  # Not valid JSON
)

result = evaluator(invalid_input)
print(f"Invalid JSON - Status: {result.status.value}")
print(f"Message: {result.message}")

# --- Example: Extracting JSON from markdown ---

print("\n--- Markdown Extraction Example ---")

markdown_input = EvalInput(
    user_input="Give me the data as JSON",
    model_output='''Here's the extracted data:

```json
{"name": "Alice", "age": 25}
```

Let me know if you need anything else!''',
)

result = evaluator(markdown_input)
print(f"Markdown JSON - Status: {result.status.value}")


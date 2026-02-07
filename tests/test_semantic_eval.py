import pytest
import math

from aiobs.evals import EvalInput, EvalStatus
from aiobs.evals.correctness.semantic_similarity import SemanticSimilarityEval, SemanticSimilarityConfig


class FakeLLM:
    """Deterministic fake embeddings for unit testing."""

    def embed(self, text: str, **kwargs):
        # Very simple deterministic embedding based on character codes
        # Return a high-similarity embedding for similar texts
        embedding = [float(ord(c) % 50) for c in text[:50]]
        # Normalize to prevent floating point precision issues
        return [min(max(e, 0.0), 1.0) for e in embedding]


class HighSimilarityLLM:
    """LLM that returns identical embeddings for test setup."""
    
    def embed(self, text: str, **kwargs):
        # Returns embedding that will produce ~0.8 similarity with different text
        return [1.0, 1.0, 1.0, 1.0, 1.0]


class LowSimilarityLLM:
    """LLM that returns opposite embeddings to produce low similarity."""
    
    def embed(self, text: str, **kwargs):
        # Returns embeddings that will produce ~0.2 similarity
        normalized = text.lower().strip()
        if "paris" in normalized or "france" in normalized:
            return [1.0, 0.0, 1.0, 0.0, 1.0]
        elif "text a" in normalized:
            return [1.0, 1.0, 1.0, 1.0, 1.0]
        elif "text b" in normalized:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            return [0.0, 1.0, 0.0, 1.0, 0.0]


class ConsistentLLM:
    """LLM that returns consistent embeddings for similar texts."""
    
    def embed(self, text: str, **kwargs):
        # Return same embedding for semantically similar texts
        normalized = text.lower().strip()
        if "paris" in normalized and "france" in normalized:
            # Similar semantic meaning
            return [0.9, 0.8, 0.7, 0.6, 0.5]
        elif "berlin" in normalized and "germany" in normalized:
            # Different semantic meaning
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            # Generic embedding
            return [0.5, 0.5, 0.5, 0.5, 0.5]


class ZeroVectorLLM:
    """LLM that returns zero vectors (edge case)."""
    
    def embed(self, text: str, **kwargs):
        return [0.0, 0.0, 0.0]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def high_similarity_llm():
    return HighSimilarityLLM()


@pytest.fixture
def low_similarity_llm():
    return LowSimilarityLLM()


@pytest.fixture
def consistent_llm():
    return ConsistentLLM()


@pytest.fixture
def zero_vector_llm():
    return ZeroVectorLLM()


# =============================================================================
# Basic Evaluation Tests
# =============================================================================


def test_semantic_similarity_pass(high_similarity_llm):
    """Test semantic similarity evaluation passes when above threshold."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Paris is the capital city of France.",
        expected_output="The capital of France is Paris.",
    )

    result = evaluator.evaluate(inp, llm=high_similarity_llm)

    assert result.status == EvalStatus.PASSED
    assert result.passed is True
    assert result.score > evaluator.config.threshold


def test_semantic_similarity_fail(low_similarity_llm):
    """Test semantic similarity evaluation fails when below threshold."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Berlin is the capital of Germany.",
        expected_output="The capital of France is Paris.",
    )

    result = evaluator.evaluate(inp, llm=low_similarity_llm)

    assert result.status == EvalStatus.FAILED
    assert result.passed is False
    assert result.score < evaluator.config.threshold


def test_semantic_similarity_identical_text(high_similarity_llm):
    """Test that identical text produces perfect similarity."""
    evaluator = SemanticSimilarityEval()

    text = "This is exactly the same text."

    inp = EvalInput(
        user_input="Q",
        model_output=text,
        expected_output=text,
    )

    result = evaluator.evaluate(inp, llm=high_similarity_llm)

    assert result.passed
    assert result.score == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_semantic_similarity_missing_expected(fake_llm):
    """Test error when expected_output is not provided."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="A",
    )

    result = evaluator.evaluate(inp, llm=fake_llm)

    assert result.status == EvalStatus.ERROR
    assert "expected_output" in result.message.lower()


def test_semantic_similarity_missing_llm():
    """Test error when LLM is not provided."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="A",
        expected_output="Expected",
    )

    result = evaluator.evaluate(inp)  # No llm provided

    assert result.status == EvalStatus.ERROR
    assert "llm" in result.message.lower()


def test_semantic_similarity_expected_from_kwargs(fake_llm):
    """Test that expected_output can be provided via kwargs."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="A",
    )

    result = evaluator.evaluate(inp, llm=fake_llm, expected="Expected")

    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert "expected" not in result.message.lower()


# =============================================================================
# Configuration Tests
# =============================================================================


def test_semantic_similarity_custom_threshold_pass(consistent_llm):
    """Test evaluation with custom threshold that passes."""
    # Use a low threshold so it passes
    config = SemanticSimilarityConfig(threshold=0.1)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Paris is the capital of France.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.passed is True
    assert result.score >= evaluator.config.threshold


def test_semantic_similarity_custom_threshold_fail(consistent_llm):
    """Test evaluation with custom threshold that fails."""
    # Use a high threshold so it fails
    config = SemanticSimilarityConfig(threshold=0.99)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Berlin is the capital of Germany.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.passed is False
    assert result.score < evaluator.config.threshold


def test_semantic_similarity_threshold_boundary(consistent_llm):
    """Test evaluation at the exact threshold boundary."""
    # Create embeddings that will produce exactly 0.7 similarity
    evaluator = SemanticSimilarityEval(
        config=SemanticSimilarityConfig(threshold=0.7)
    )

    inp = EvalInput(
        user_input="Q",
        model_output="Paris is the capital of France.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    # At or above threshold should pass
    assert result.passed == (result.score >= 0.7)


def test_semantic_similarity_include_details_true(consistent_llm):
    """Test that details are included when config enables it."""
    config = SemanticSimilarityConfig(include_details=True)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Paris is capital of France.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.details is not None
    assert "similarity" in result.details
    assert "threshold" in result.details


def test_semantic_similarity_include_details_false(consistent_llm):
    """Test that details are excluded when config disables it."""
    config = SemanticSimilarityConfig(include_details=False)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Paris is capital of France.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.details is None or result.details == {}


# =============================================================================
# Assertions Tests
# =============================================================================


def test_semantic_similarity_includes_assertions(consistent_llm):
    """Test that assertion details are populated."""
    config = SemanticSimilarityConfig(include_details=True)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Capital?",
        model_output="Paris is capital of France.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.assertions is not None
    assert len(result.assertions) > 0
    
    assertion = result.assertions[0]
    assert assertion.name == "semantic_similarity"
    assert assertion.passed == result.passed
    assert "similarity" in assertion.message


def test_semantic_similarity_assertion_contains_values(consistent_llm):
    """Test that assertions contain expected and actual values."""
    config = SemanticSimilarityConfig(include_details=True)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Model says this",
        expected_output="Expected says that",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.assertions is not None
    assertion = result.assertions[0]
    assert assertion.expected is not None
    assert assertion.actual is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_semantic_similarity_empty_strings(fake_llm):
    """Test with empty strings."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="",
        expected_output="",
    )

    result = evaluator.evaluate(inp, llm=fake_llm)

    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_very_long_strings(fake_llm):
    """Test with very long strings."""
    evaluator = SemanticSimilarityEval()
    
    long_text = "This is a very long text. " * 100

    inp = EvalInput(
        user_input="Q",
        model_output=long_text,
        expected_output=long_text,
    )

    result = evaluator.evaluate(inp, llm=fake_llm)

    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_single_character(fake_llm):
    """Test with single character strings."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="a",
        expected_output="a",
    )

    result = evaluator.evaluate(inp, llm=fake_llm)

    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_special_characters(fake_llm):
    """Test with special characters."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="!@#$%^&*()",
        expected_output="!@#$%^&*()",
    )

    result = evaluator.evaluate(inp, llm=fake_llm)

    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_unicode_characters(fake_llm):
    """Test with unicode characters."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="こんにちは世界",
        expected_output="こんにちは世界",
    )

    result = evaluator.evaluate(inp, llm=fake_llm)

    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_zero_vectors(zero_vector_llm):
    """Test with zero vectors (edge case for cosine similarity)."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="Some text",
        expected_output="Some other text",
    )

    result = evaluator.evaluate(inp, llm=zero_vector_llm)

    # Should handle gracefully and return 0.0
    assert result.status in [EvalStatus.PASSED, EvalStatus.FAILED]
    assert result.score == 0.0


# =============================================================================
# Cosine Similarity Calculation Tests
# =============================================================================


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity of identical vectors."""
    similarity = SemanticSimilarityEval._cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    )
    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity of orthogonal vectors."""
    similarity = SemanticSimilarityEval._cosine_similarity(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    )
    assert similarity == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity of opposite vectors."""
    similarity = SemanticSimilarityEval._cosine_similarity(
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0]
    )
    assert similarity == pytest.approx(-1.0)


def test_cosine_similarity_scaled_vectors():
    """Test cosine similarity of scaled vectors (should be same)."""
    v1 = [1.0, 2.0, 3.0]
    v2 = [2.0, 4.0, 6.0]  # Scaled by 2
    
    similarity = SemanticSimilarityEval._cosine_similarity(v1, v2)
    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_empty_vectors():
    """Test cosine similarity with empty vectors."""
    similarity = SemanticSimilarityEval._cosine_similarity([], [])
    assert similarity == 0.0


def test_cosine_similarity_single_element():
    """Test cosine similarity with single element vectors."""
    similarity = SemanticSimilarityEval._cosine_similarity([5.0], [5.0])
    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_different_lengths():
    """Test cosine similarity with different length vectors."""
    # zip will stop at shortest length
    similarity = SemanticSimilarityEval._cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0]
    )
    # Should compute similarity on common elements
    assert 0.0 <= similarity <= 1.0


# =============================================================================
# Message and Score Tests
# =============================================================================


def test_semantic_similarity_message_format(consistent_llm):
    """Test that result message is properly formatted."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="Model output",
        expected_output="Expected output",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert "similarity" in result.message.lower()
    assert "threshold" in result.message.lower()


def test_semantic_similarity_score_in_range(consistent_llm):
    """Test that score is always in valid range [0, 1]."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="Very different text",
        expected_output="Completely opposite text",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_eval_name(consistent_llm):
    """Test that eval name is set correctly."""
    evaluator = SemanticSimilarityEval()

    inp = EvalInput(
        user_input="Q",
        model_output="A",
        expected_output="B",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.eval_name == "semantic_similarity"


# =============================================================================
# Status and Pass/Fail Tests
# =============================================================================


def test_semantic_similarity_status_passed_above_threshold(consistent_llm):
    """Test that status is PASSED when score is above threshold."""
    config = SemanticSimilarityConfig(threshold=0.1)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Paris is the capital of France.",
        expected_output="Paris is the capital of France.",
    )

    result = evaluator.evaluate(inp, llm=consistent_llm)

    assert result.status == EvalStatus.PASSED
    assert result.passed is True
    assert result.failed is False


def test_semantic_similarity_status_failed_below_threshold(low_similarity_llm):
    """Test that status is FAILED when score is below threshold."""
    # Use a moderate threshold so that the low similarity embeddings fail
    config = SemanticSimilarityConfig(threshold=0.7)
    evaluator = SemanticSimilarityEval(config=config)

    inp = EvalInput(
        user_input="Q",
        model_output="Text A",
        expected_output="Text B",
    )

    result = evaluator.evaluate(inp, llm=low_similarity_llm)

    assert result.status == EvalStatus.FAILED
    assert result.passed is False
    assert result.failed is True

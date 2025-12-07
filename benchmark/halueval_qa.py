"""Benchmark HallucinationDetectionEval against HaluEvalQA Dataset.

This script benchmarks the HallucinationDetectionEval evaluator against
the HaluEvalQA dataset, which contains 10,000 QA samples with both
hallucinated and correct answers.

Usage:
    python benchmark/halueval_qa.py --judge-model gpt-4o --sample-size 100
    python benchmark/halueval_qa.py --judge-model gpt-4o-mini --provider openai
    python benchmark/halueval_qa.py --judge-model gemini-2.0-flash --provider gemini
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

from aiobs.evals import HallucinationDetectionEval, EvalInput


# Dataset URL
HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str, output_path: Path) -> None:
    """Download the HaluEvalQA dataset.
    
    Args:
        url: URL to download from.
        output_path: Path to save the dataset.
    """
    if output_path.exists():
        print(f"Dataset already exists at {output_path}")
        return
    
    print(f"Downloading dataset from {url}...")
    try:
        with urlopen(url) as response:
            data = response.read()
            with open(output_path, "wb") as f:
                f.write(data)
        print(f"Dataset downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def load_halueval_qa(path: Path) -> List[Dict[str, Any]]:
    """Load HaluEvalQA dataset.
    
    The dataset is in JSONL format (JSON Lines) where each line is a JSON object.
    
    Args:
        path: Path to the dataset JSON/JSONL file.
        
    Returns:
        List of dataset samples.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        # Try to load as JSON array first
        try:
            content = f.read()
            f.seek(0)
            parsed = json.loads(content)
            if isinstance(parsed, list):
                data = parsed
            else:
                # Single object, wrap in list
                data = [parsed]
        except json.JSONDecodeError:
            # If that fails, try JSONL format (one JSON object per line)
            f.seek(0)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    if not data:
        raise ValueError(f"No valid data found in {path}")
    
    print(f"Loaded {len(data)} samples from {path}")
    return data


def create_evaluator(
    provider: str,
    model: str,
    temperature: float = 0.0,
) -> HallucinationDetectionEval:
    """Create a HallucinationDetectionEval evaluator.
    
    Args:
        provider: LLM provider ("openai", "gemini").
        model: Model name.
        temperature: Temperature for the judge LLM.
        
    Returns:
        Configured evaluator instance.
    """
    if provider == "openai":
        if OpenAI is None:
            raise ImportError("openai package not installed. Install with: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        return HallucinationDetectionEval.with_openai(
            client=client,
            model=model,
            temperature=temperature,
        )
    
    elif provider == "gemini":
        if genai is None:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        
        # Support both API key and service account JSON
        api_key = os.getenv("GEMINI_API_KEY")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if api_key:
            # Use API key authentication
            client = genai.Client(api_key=api_key)
        elif credentials_path:
            # Use service account JSON file
            try:
                from google.oauth2 import service_account
                import google.auth
            except ImportError:
                raise ImportError(
                    "google-auth package required for service account authentication. "
                    "Install with: pip install google-auth"
                )
            
            if not os.path.exists(credentials_path):
                raise ValueError(f"Service account file not found: {credentials_path}")
            
            # Load credentials from service account file
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Get project ID from credentials file
            with open(credentials_path, 'r') as f:
                creds_info = json.load(f)
                project_id = creds_info.get('project_id')
            
            if not project_id:
                raise ValueError("project_id not found in service account JSON")
            
            # Create client with Vertex AI and service account credentials
            location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials
            )
        else:
            # Try default credentials
            try:
                import google.auth
                credentials, project_id = google.auth.default(
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                if not project_id:
                    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                if not project_id:
                    raise ValueError("No project ID found")
                
                location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                    credentials=credentials
                )
            except Exception as e:
                raise ValueError(
                    "No Gemini authentication found. Set either:\n"
                    "  - GEMINI_API_KEY environment variable (for API key)\n"
                    "  - GOOGLE_APPLICATION_CREDENTIALS environment variable (for service account JSON)\n"
                    f"Error: {e}"
                )
        
        return HallucinationDetectionEval.with_gemini(
            client=client,
            model=model,
            temperature=temperature,
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, gemini")


async def evaluate_sample_async(
    evaluator: HallucinationDetectionEval,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single sample (both hallucinated and correct answers).
    
    Args:
        evaluator: The evaluator instance.
        sample: Dataset sample with knowledge, question, right_answer, hallucinated_answer.
        
    Returns:
        Dictionary with evaluation results for both answers.
    """
    knowledge = sample["knowledge"]
    question = sample["question"]
    
    results = {}
    
    # Test hallucinated answer (should fail)
    try:
        result_hallucinated = await evaluator.evaluate_async(
            EvalInput(
                user_input=question,
                model_output=sample["hallucinated_answer"],
                context={"documents": [knowledge]},
            )
        )
        results["hallucinated"] = {
            "status": result_hallucinated.status.value,
            "score": result_hallucinated.score,
            "failed": result_hallucinated.failed,
            "passed": result_hallucinated.passed,
            "has_hallucinations": result_hallucinated.details.get("has_hallucinations", False) if result_hallucinated.details else False,
        }
    except Exception as e:
        results["hallucinated"] = {
            "status": "error",
            "score": 0.0,
            "failed": True,
            "passed": False,
            "error": str(e),
        }
    
    # Test correct answer (should pass)
    try:
        result_correct = await evaluator.evaluate_async(
            EvalInput(
                user_input=question,
                model_output=sample["right_answer"],
                context={"documents": [knowledge]},
            )
        )
        results["correct"] = {
            "status": result_correct.status.value,
            "score": result_correct.score,
            "failed": result_correct.failed,
            "passed": result_correct.passed,
            "has_hallucinations": result_correct.details.get("has_hallucinations", False) if result_correct.details else False,
        }
    except Exception as e:
        results["correct"] = {
            "status": "error",
            "score": 0.0,
            "failed": True,
            "passed": False,
            "error": str(e),
        }
    
    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate evaluation metrics.
    
    Args:
        results: List of evaluation results for each sample.
        
    Returns:
        Dictionary with calculated metrics.
    """
    counts = {
        "hallucinated_detected": 0,  # True positives
        "hallucinated_missed": 0,    # False negatives
        "correct_passed": 0,         # True negatives
        "correct_flagged": 0,        # False positives
    }
    
    hallucinated_scores = []
    correct_scores = []
    
    for result in results:
        # Check hallucinated answer results
        if "hallucinated" in result:
            h_result = result["hallucinated"]
            if h_result.get("failed") or h_result.get("has_hallucinations"):
                counts["hallucinated_detected"] += 1
            else:
                counts["hallucinated_missed"] += 1
            hallucinated_scores.append(h_result.get("score", 0.0))
        
        # Check correct answer results
        if "correct" in result:
            c_result = result["correct"]
            if c_result.get("passed") and not c_result.get("has_hallucinations"):
                counts["correct_passed"] += 1
            else:
                counts["correct_flagged"] += 1
            correct_scores.append(c_result.get("score", 0.0))
    
    total_samples = len(results)
    total_evaluations = total_samples * 2
    
    # Calculate metrics
    accuracy = (counts["hallucinated_detected"] + counts["correct_passed"]) / total_evaluations if total_evaluations > 0 else 0.0
    
    tp = counts["hallucinated_detected"]
    fp = counts["correct_flagged"]
    fn = counts["hallucinated_missed"]
    tn = counts["correct_passed"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # True Positive Rate (TPR) = Recall
    tpr = recall
    
    # False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Score statistics
    hallucinated_mean = sum(hallucinated_scores) / len(hallucinated_scores) if hallucinated_scores else 0.0
    correct_mean = sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
    
    return {
        "raw_counts": counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "total_samples": total_samples,
        "total_evaluations": total_evaluations,
        "score_statistics": {
            "hallucinated_mean": hallucinated_mean,
            "correct_mean": correct_mean,
            "hallucinated_scores": hallucinated_scores,
            "correct_scores": correct_scores,
        },
    }


def print_metrics_summary(metrics: Dict[str, Any], judge_model: str, provider: str) -> None:
    """Print metrics summary to console.
    
    Args:
        metrics: Calculated metrics dictionary.
        judge_model: Judge model name.
        provider: Provider name.
    """
    print("\n" + "=" * 80)
    print("HaluEvalQA Benchmark Results")
    print("=" * 80)
    print(f"Judge Model: {judge_model} ({provider})")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Total Evaluations: {metrics['total_evaluations']}")
    print("\n" + "-" * 80)
    print("Confusion Matrix:")
    print(f"  True Positives (Hallucinated Detected):  {metrics['raw_counts']['hallucinated_detected']}")
    print(f"  False Negatives (Hallucinated Missed):   {metrics['raw_counts']['hallucinated_missed']}")
    print(f"  True Negatives (Correct Passed):          {metrics['raw_counts']['correct_passed']}")
    print(f"  False Positives (Correct Flagged):        {metrics['raw_counts']['correct_flagged']}")
    print("\n" + "-" * 80)
    print("Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  TPR:       {metrics['true_positive_rate']:.4f}")
    print(f"  FPR:       {metrics['false_positive_rate']:.4f}")
    print("\n" + "-" * 80)
    print("Score Statistics:")
    print(f"  Hallucinated Answers - Mean Score: {metrics['score_statistics']['hallucinated_mean']:.4f}")
    print(f"  Correct Answers - Mean Score:     {metrics['score_statistics']['correct_mean']:.4f}")
    print("=" * 80 + "\n")


async def run_benchmark(
    evaluator: HallucinationDetectionEval,
    data: List[Dict[str, Any]],
    sample_size: Optional[int] = None,
    progress_interval: int = 10,
) -> List[Dict[str, Any]]:
    """Run benchmark evaluation loop.
    
    Args:
        evaluator: The evaluator instance.
        data: Dataset samples.
        sample_size: Number of samples to evaluate (None for all).
        progress_interval: Print progress every N samples.
        
    Returns:
        List of evaluation results.
    """
    samples = data[:sample_size] if sample_size else data
    total = len(samples)
    
    print(f"\nEvaluating {total} samples...")
    start_time = time.time()
    
    results = []
    for i, sample in enumerate(samples, 1):
        result = await evaluate_sample_async(evaluator, sample)
        results.append(result)
        
        if i % progress_interval == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"Progress: {i}/{total} ({i/total*100:.1f}%) | "
                  f"Rate: {rate:.2f} samples/s | ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f} seconds ({total/total_time:.2f} samples/s)")
    
    return results


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_path: Path,
    judge_model: str,
    provider: str,
    sample_size: Optional[int],
) -> None:
    """Save results to JSON file.
    
    Args:
        results: Evaluation results.
        metrics: Calculated metrics.
        output_path: Path to save results.
        judge_model: Judge model name.
        provider: Provider name.
        sample_size: Sample size used.
    """
    output_data = {
        "benchmark": "HaluEvalQA",
        "judge_model": judge_model,
        "provider": provider,
        "sample_size": sample_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark HallucinationDetectionEval against HaluEvalQA Dataset"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for judge LLM (default: 0.0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: auto-generated)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip dataset download if file exists",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "qa_data.json"
    
    # Download dataset if needed
    if not args.no_download or not dataset_path.exists():
        download_dataset(HALUEVAL_QA_URL, dataset_path)
    
    # Load dataset
    data = load_halueval_qa(dataset_path)
    
    # Create evaluator
    try:
        evaluator = create_evaluator(
            provider=args.provider,
            model=args.judge_model,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"Error creating evaluator: {e}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_safe = args.judge_model.replace("/", "_").replace(":", "_")
        output_path = data_dir / f"results_{args.provider}_{model_safe}_{args.sample_size or len(data)}.json"
    
    # Run benchmark
    try:
        results = asyncio.run(
            run_benchmark(
                evaluator=evaluator,
                data=data,
                sample_size=args.sample_size,
            )
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print_metrics_summary(metrics, args.judge_model, args.provider)
    
    # Save results
    save_results(
        results=results,
        metrics=metrics,
        output_path=output_path,
        judge_model=args.judge_model,
        provider=args.provider,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()


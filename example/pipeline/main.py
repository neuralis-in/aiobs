import os
import sys
from typing import Optional

# Make repo root importable for local package (aiobs)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from aiobs import observer  # noqa: E402
from aiobs.exporters import GCSExporter  # noqa: E402

# Support running as a module or a script
try:  # package-relative imports (when run with -m example.pipeline.main)
    from .client import make_client, default_model  # type: ignore
    from .tasks.research import research  # type: ignore
    from .tasks.summarize import summarize  # type: ignore
    from .tasks.critique import critique  # type: ignore
except Exception:  # fallback for direct script run: python example/pipeline/main.py
    _example_dir = os.path.dirname(os.path.dirname(__file__))
    if _example_dir not in sys.path:
        sys.path.insert(0, _example_dir)
    from pipeline.client import make_client, default_model  # type: ignore
    from pipeline.tasks.research import research  # type: ignore
    from pipeline.tasks.summarize import summarize  # type: ignore
    from pipeline.tasks.critique import critique  # type: ignore


def main(query: Optional[str] = None, use_gcs: bool = False) -> None:
    client = make_client()
    model = default_model()

    # Start a single observability session for the whole pipeline
    observer.observe(session_name="pipeline-example", api_key=os.getenv("AIOBS_API_KEY"))

    try:
        q = query or "In one sentence, explain what an API is."
        print(f"Query: {q}\n")

        notes = research(q, client, model)
        print("Notes:")
        for n in notes:
            print(f"- {n}")
        print()

        draft = summarize(notes, client, model)
        print("Draft:\n" + draft + "\n")

        improved = critique(draft, client, model)
        print("Improved:\n" + improved + "\n")
    finally:
        observer.end()
        
        if use_gcs:
            # Export to Google Cloud Storage
            # Set env vars: AIOBS_GCS_BUCKET, AIOBS_GCS_PREFIX (optional), AIOBS_GCS_PROJECT (optional)
            exporter = GCSExporter(
                bucket=os.environ["AIOBS_GCS_BUCKET"],
                prefix=os.getenv("AIOBS_GCS_PREFIX", "traces/"),
                project=os.getenv("AIOBS_GCS_PROJECT"),
            )
            result = observer.flush(exporter=exporter)
            print(f"Observability exported to: {result.destination}")
        else:
            out = observer.flush()
            print(f"Observability written to: {out}")


if __name__ == "__main__":
    # Usage: python main.py [query] [--gcs]
    # Export to GCS: AIOBS_GCS_BUCKET=my-bucket python main.py --gcs
    use_gcs = "--gcs" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--gcs"]
    arg_query = args[0] if args else None
    main(arg_query, use_gcs=use_gcs)

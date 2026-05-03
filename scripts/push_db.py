"""Push local data/portfolio.db to the HF Dataset so the Space picks it up.

Usage:
    python scripts/push_db.py                    # uses HF_TOKEN from .env
    HF_TOKEN=hf_xxx python scripts/push_db.py
"""
import os
import sys
from pathlib import Path

# Allow running from repo root or from scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from core.config import DB_PATH  # noqa: E402

DATASET_REPO = os.environ.get("PERSIST_DATASET", "irajkoohi/stocksPortfolio_dataset")
DB_FILENAME  = "portfolio.db"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("ERROR: HF_TOKEN not set (add it to .env or export it).")
        sys.exit(1)

    db = Path(DB_PATH)
    if not db.exists():
        print(f"ERROR: {db} not found — run the app and optimise first.")
        sys.exit(1)

    size_kb = db.stat().st_size / 1024
    print(f"Uploading {db} ({size_kb:.1f} KB) → {DATASET_REPO}/{DB_FILENAME} …")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(db),
            path_in_repo=DB_FILENAME,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message="manual: sync local portfolio.db to Space",
        )
        print("Done — Space will load this DB on next cold-start.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""HF Dataset persistence for portfolio.db on Space cold-starts.

On Space:
- pull_db_from_hub(): download portfolio.db from the dataset before init_db()
- schedule_db_push(): debounced background upload after every commit (~10s)

Locally (no SPACE_ID env var), every function is a no-op.
"""
import logging
import os
import threading

from core.config import DB_PATH, IS_HF_SPACE

log = logging.getLogger(__name__)

DATASET_REPO       = os.environ.get("PERSIST_DATASET", "irajkoohi/stocksPortfolio_state")
PUSH_DEBOUNCE_SEC  = 10.0
DB_FILENAME        = "portfolio.db"

_push_timer: threading.Timer | None = None
_push_lock  = threading.Lock()


def _token() -> str | None:
    return os.environ.get("HF_TOKEN")


def pull_db_from_hub() -> None:
    """Download the persisted DB from the dataset on Space cold-start."""
    if not IS_HF_SPACE:
        return
    token = _token()
    if not token:
        log.warning("Persistence: HF_TOKEN missing; skipping pull")
        return
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id    = DATASET_REPO,
            repo_type  = "dataset",
            filename   = DB_FILENAME,
            token      = token,
            local_dir  = str(DB_PATH.parent),
        )
        log.info("Persistence: pulled %s -> %s", DATASET_REPO, path)
    except Exception as exc:
        log.info("Persistence: nothing to pull (%s)", exc)


def _push_now() -> None:
    token = _token()
    if not token or not DB_PATH.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj = str(DB_PATH),
            path_in_repo    = DB_FILENAME,
            repo_id         = DATASET_REPO,
            repo_type       = "dataset",
            commit_message  = "auto: persist portfolio.db",
        )
        log.info("Persistence: pushed DB to %s", DATASET_REPO)
    except Exception as exc:
        log.warning("Persistence: push failed: %s", exc)


def schedule_db_push() -> None:
    """Debounced push: rapid commits coalesce into one upload PUSH_DEBOUNCE_SEC later."""
    if not IS_HF_SPACE:
        return
    global _push_timer
    with _push_lock:
        if _push_timer is not None:
            _push_timer.cancel()
        _push_timer = threading.Timer(PUSH_DEBOUNCE_SEC, _push_now)
        _push_timer.daemon = True
        _push_timer.start()

"""
Entry point for both local development and HF Space (app_file: main.py).
Kills any process bound to the app ports, then starts FastAPI in a
background thread and launches Gradio on port 7860.
"""
import logging
import subprocess
import sys
import threading
import time

import uvicorn

from core.config import API_PORT, GRADIO_PORT, IS_HF_SPACE
from core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _kill_port(port: int) -> None:
    """Kill any process currently bound to *port* so we can reclaim it."""
    logger.info("Freeing port %d …", port)
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and f":{port}" in parts[1] and "LISTENING" in line:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", parts[-1]], check=False
                    )
        else:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                check=False, capture_output=True,
            )
        time.sleep(0.4)
    except Exception as exc:
        logger.debug("Port-kill for %d skipped: %s", port, exc)


def _start_api() -> None:
    from api.server import create_api
    uvicorn.run(create_api(), host="0.0.0.0", port=API_PORT, log_level="warning")


def main() -> None:
    # ── free ports before binding ─────────────────────────────────────────────
    _kill_port(GRADIO_PORT)
    _kill_port(API_PORT)

    logger.info("Initialising database …")
    init_db()

    logger.info("Populating knowledge base (skipped if already done) …")
    try:
        from scripts.populate_kb import populate
        populate()
    except Exception as exc:
        logger.warning("KB population failed: %s", exc)

    logger.info("Starting FastAPI on port %d (background) …", API_PORT)
    threading.Thread(target=_start_api, daemon=True).start()

    logger.info("Launching Gradio on port %d …", GRADIO_PORT)
    from ui.gradio_interface import create_interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

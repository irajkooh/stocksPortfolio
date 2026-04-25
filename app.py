"""
Entry point for both local development and HF Space (app_file: main.py).
Kills any process bound to the app ports, then starts FastAPI in a
background thread and launches Gradio on port 7860.
"""
import logging
import subprocess
import sys
import tempfile
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
            # macOS: lsof; Linux: fuser (fallback)
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, check=False,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                subprocess.run(["kill", "-9", pid], check=False, capture_output=True)
        time.sleep(0.4)
    except Exception as exc:
        logger.debug("Port-kill for %d skipped: %s", port, exc)


def _start_api() -> None:
    from api.backend import create_api
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
    from ui.frontend import create_interface
    from ui.theme import get_theme, CUSTOM_CSS
    demo = create_interface()
    _LAUNCH_JS = r"""
(() => {
  document.querySelector('body').classList.add('dark');

  const slug = s => (s || 'plot').toString().toLowerCase()
    .replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'plot';
  const stamp = () => {
    const d = new Date(), p = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}` +
           `_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  };
  const readTitle = gd => {
    const t = ((gd && (gd._fullLayout || gd.layout)) || {}).title;
    return t ? (typeof t === 'string' ? t : (t.text || 'plot')) : 'plot';
  };

  let _gd = null;

  // Track which plot's camera button was pressed (mousedown fires before Plotly's handler)
  const trackBtn = ev => {
    const btn = ev.target.closest &&
      ev.target.closest('.modebar-btn[data-title*="Download plot"]');
    if (btn) _gd = btn.closest('.js-plotly-plot') || _gd;
  };
  document.addEventListener('mousedown', trackBtn, true);
  document.addEventListener('click',     trackBtn, true);

  const applyName = a => {
    if (!_gd || !a.hasAttribute('download')) return;
    const ext = (a.getAttribute('download').match(/\.\w+$/) || ['.png'])[0];
    a.setAttribute('download', `${slug(readTitle(_gd))}_${stamp()}${ext}`);
    _gd = null;
  };

  // Path 1 — Plotly calls a.click()
  const _origClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function() {
    applyName(this);
    return _origClick.call(this);
  };

  // Path 2 — Plotly calls a.dispatchEvent(new MouseEvent('click', ...))
  const _origDispatch = EventTarget.prototype.dispatchEvent;
  EventTarget.prototype.dispatchEvent = function(ev) {
    if (ev.type === 'click' && this instanceof HTMLAnchorElement) applyName(this);
    return _origDispatch.call(this, ev);
  };

  // Path 3 — fallback: watch for <a download="..."> inserted into the DOM
  new MutationObserver(muts => {
    if (!_gd) return;
    for (const m of muts)
      for (const n of m.addedNodes)
        if (n.nodeType === 1 && n.tagName === 'A' && n.hasAttribute('download'))
          { applyName(n); return; }
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
"""
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=get_theme(),
        js=_LAUNCH_JS,
        css=CUSTOM_CSS,
        allowed_paths=[tempfile.gettempdir()],
    )


if __name__ == "__main__":
    main()

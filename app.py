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


def _patch_gradio_client() -> None:
    """Fix gradio_client 0.9.x crash when Pydantic 2 emits additionalProperties:true (bool).

    gradio_client.utils.get_type does `if "const" in schema` which raises TypeError
    when schema is a boolean. Guard both get_type and _json_schema_to_python_type.
    """
    try:
        import gradio_client.utils as _gcu
        _orig_get_type = _gcu.get_type
        _orig_j2p = _gcu._json_schema_to_python_type

        def _safe_get_type(schema):
            if not isinstance(schema, dict):
                return "any"
            return _orig_get_type(schema)

        def _safe_j2p(schema, defs=None):
            if not isinstance(schema, dict):
                return "any"
            return _orig_j2p(schema, defs)

        _gcu.get_type = _safe_get_type
        _gcu._json_schema_to_python_type = _safe_j2p
        logger.debug("Patched gradio_client.utils for boolean additionalProperties.")
    except Exception as exc:
        logger.debug("gradio_client patch skipped: %s", exc)


def _patch_websockets_asyncio() -> None:
    """Shim websockets.asyncio for yfinance 1.x on websockets < 14."""
    import sys
    try:
        import websockets.asyncio  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types
    import websockets

    asyncio_mod = types.ModuleType("websockets.asyncio")
    asyncio_mod.__path__ = []
    asyncio_mod.__package__ = "websockets.asyncio"
    sys.modules["websockets.asyncio"] = asyncio_mod
    websockets.asyncio = asyncio_mod  # type: ignore[attr-defined]

    client_mod = types.ModuleType("websockets.asyncio.client")
    client_mod.__package__ = "websockets.asyncio"
    try:
        from websockets.legacy.client import connect
        client_mod.connect = connect
    except ImportError:
        client_mod.connect = getattr(websockets, "connect", None)
    sys.modules["websockets.asyncio.client"] = client_mod
    asyncio_mod.client = client_mod  # type: ignore[attr-defined]
    logger.debug("Injected websockets.asyncio shim.")


def _patch_hf_folder() -> None:
    # HfFolder was removed in huggingface_hub 1.0; Gradio 4.x still imports it.
    # Inject a shim before any gradio import runs.
    import huggingface_hub as _hub
    if hasattr(_hub, "HfFolder"):
        return

    class HfFolder:
        @staticmethod
        def get_token():
            try:
                return _hub.get_token()
            except Exception:
                return None

        @staticmethod
        def save_token(token: str) -> None:
            try:
                _hub.login(token=token, add_to_git_credential=False)
            except Exception:
                pass

        @staticmethod
        def delete_token() -> None:
            try:
                _hub.logout()
            except Exception:
                pass

    _hub.HfFolder = HfFolder
    logger.debug("Injected HfFolder shim into huggingface_hub.")


def _start_api() -> None:
    from api.backend import create_api
    uvicorn.run(create_api(), host="0.0.0.0", port=API_PORT, log_level="warning")


def main() -> None:
    _patch_gradio_client()
    _patch_websockets_asyncio()
    _patch_hf_folder()

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
    demo = create_interface(theme=get_theme(), css=CUSTOM_CSS, js=_LAUNCH_JS)
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        inbrowser=True,
        allowed_paths=[tempfile.gettempdir()],
        max_threads=40,
    )


if __name__ == "__main__":
    main()

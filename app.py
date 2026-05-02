"""
Entry point for both local development and HF Space (app_file: main.py).
Kills any process bound to the app ports, then starts FastAPI in a
background thread and launches Gradio on port 7860.
"""
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time

import requests as _requests
import uvicorn

from core.config import API_PORT, GRADIO_PORT, IS_HF_SPACE
from core.database import init_db
from core.persistence import pull_db_from_hub

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


def _ping_self(url: str) -> None:
    """Lightweight GET to keep the HF Space from going to sleep."""
    try:
        _requests.get(url, timeout=10)
        logger.info("[keep-alive] pinged %s", url)
    except Exception as exc:
        logger.warning("[keep-alive] ping failed: %s", exc)


def start_keep_alive_scheduler(space_url: str):
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BackgroundScheduler(timezone="UTC", daemon=True)

    # Every 30 minutes — resets HF inactivity timer
    scheduler.add_job(
        _ping_self,
        trigger=IntervalTrigger(minutes=30),
        args=[space_url],
        id="keep_alive_30m",
        replace_existing=True,
    )

    # Daily at 06:00 UTC — warm-up before market open
    scheduler.add_job(
        _ping_self,
        trigger=CronTrigger(hour=6, minute=0),
        args=[space_url],
        id="keep_alive_daily",
        replace_existing=True,
    )

    scheduler.start()
    return scheduler


def main() -> None:
    _patch_gradio_client()
    _patch_websockets_asyncio()
    _patch_hf_folder()

    # ── free ports before binding ─────────────────────────────────────────────
    _kill_port(GRADIO_PORT)
    _kill_port(API_PORT)

    logger.info("Pulling persisted DB from HF Dataset (Space only) …")
    pull_db_from_hub()

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

    # Keep-alive scheduler — only on HF Space (SPACE_ID is set automatically by HF).
    # Format: SPACE_ID="username/spacename" → "https://username-spacename.hf.space"
    if IS_HF_SPACE:
        space_id = os.environ.get("SPACE_ID", "")
        if "/" in space_id:
            user, name = space_id.split("/", 1)
            hf_space_url = f"https://{user}-{name}.hf.space".lower()
            start_keep_alive_scheduler(hf_space_url)
            logger.info("[keep-alive] scheduler started for %s", hf_space_url)

    logger.info("Launching Gradio on port %d …", GRADIO_PORT)
    from ui.frontend import create_interface
    from ui.theme import get_theme, CUSTOM_CSS

    # Body of the Plotly-download patch, used for both `js=` (arrow-fn body)
    # and `head=` (raw <script> body). The head= path is the reliable one —
    # Gradio 6.9.0's frontend bundle does not appear to evaluate the top-level
    # `js=` config field, but `head=` is injected verbatim into <head>.
    _PATCH_BODY = r"""
  try { document.querySelector('body').classList.add('dark'); } catch (_) {}

  const slug = s => (s || 'plot').toString().toLowerCase()
    .replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'plot';
  const stamp = () => {
    const d = new Date(), p = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}` +
           `_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  };
  const readTitle = gd => {
    const t = ((gd && (gd._fullLayout || gd.layout)) || {}).title;
    const raw = typeof t === 'string' ? t : (t && t.text) || '';
    return raw.replace(/<[^>]*>/g, '').trim();
  };
  const buildName = gd => `${slug(readTitle(gd) || 'plot')}_${stamp()}`;

  // Primary — capture-phase click sets gd._context.toImageButtonOptions.filename
  // BEFORE Plotly's bubble-phase handler reads it.
  document.addEventListener('click', ev => {
    const path = ev.composedPath ? ev.composedPath() : [];
    let isDownload = false, gd = null;
    for (const el of path) {
      if (!el || !el.classList) continue;
      if (el.classList.contains('modebar-btn')) {
        const title = (el.getAttribute('data-title') || '').toLowerCase();
        if (title.includes('download')) isDownload = true;
      }
      if (el.classList.contains('js-plotly-plot')) { gd = el; break; }
    }
    if (isDownload && gd) {
      const ctx = gd._context = gd._context || {};
      const opts = ctx.toImageButtonOptions = ctx.toImageButtonOptions || {};
      opts.filename = buildName(gd);
    }
  }, true);

  // Track last-touched plot for the prototype fallbacks below.
  let _lastGd = null;
  document.addEventListener('mousedown', ev => {
    const path = ev.composedPath ? ev.composedPath() : [];
    for (const el of path) {
      if (el && el.classList && el.classList.contains('js-plotly-plot')) {
        _lastGd = el;
        return;
      }
    }
  }, true);

  // Fallback 1 — rewrite `newplot.*` filenames at HTMLAnchorElement.click().
  const _origClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function() {
    if (typeof this.download === 'string' && /^newplot\./i.test(this.download)) {
      const ext = (/\.[a-z0-9]+$/i.exec(this.download) || [''])[0];
      this.download = `${buildName(_lastGd)}${ext}`;
    }
    return _origClick.call(this);
  };

  // Fallback 2 — rewrite at the moment Plotly inserts its temporary <a>.
  const _origAppend = Node.prototype.appendChild;
  Node.prototype.appendChild = function(node) {
    if (node instanceof HTMLAnchorElement && typeof node.download === 'string'
        && /^newplot\./i.test(node.download)) {
      const ext = (/\.[a-z0-9]+$/i.exec(node.download) || [''])[0];
      node.download = `${buildName(_lastGd)}${ext}`;
    }
    return _origAppend.call(this, node);
  };

  // Visibility marker — lets us confirm the patch ran from the browser.
  window.__plotlyDlPatchInstalled = true;

  // ── Mobile table scroll fix ───────────────────────────────────────────────
  // Gradio sets width:100% on both .table-wrap and <table>, so the table never
  // overflows and scroll does nothing.  We also need touch-action on every
  // ancestor so the browser doesn't swallow pan gestures.
  function _fixWatchlistScroll() {
    document.querySelectorAll('.watchlist-df').forEach(function(wdf) {
      wdf.style.touchAction = 'pan-x pan-y';
      wdf.querySelectorAll('table').forEach(function(tbl) {
        tbl.style.tableLayout = 'auto';
        tbl.style.width = 'max-content';
        tbl.style.minWidth = '100%';
        tbl.style.overflow = 'visible';
        // Direct parent of <table> is the horizontal scroll container.
        var sc = tbl.parentElement;
        if (sc && sc !== wdf) {
          sc.style.overflowX = 'auto';
          sc.style.maxWidth = '100%';
          sc.style.webkitOverflowScrolling = 'touch';
          sc.style.touchAction = 'pan-x pan-y';
          sc.style.overscrollBehavior = 'contain';
        }
        // Propagate touch-action up through intermediate ancestors.
        var el = sc ? sc.parentElement : null;
        while (el && el !== wdf) {
          el.style.touchAction = 'pan-x pan-y';
          el = el.parentElement;
        }
      });
    });
  }
  if (document.readyState !== 'loading') {
    _fixWatchlistScroll();
  } else {
    document.addEventListener('DOMContentLoaded', _fixWatchlistScroll);
  }
  // Re-apply whenever Gradio re-renders table content.
  var _wlScrollObs = new MutationObserver(function(muts) {
    for (var i = 0; i < muts.length; i++) {
      if (muts[i].addedNodes.length) { setTimeout(_fixWatchlistScroll, 200); break; }
    }
  });
  function _startWlObs() {
    _wlScrollObs.observe(document.body, {childList: true, subtree: true});
  }
  document.body ? _startWlObs() : document.addEventListener('DOMContentLoaded', _startWlObs);
"""
    # `js=` form: arrow function the deprecated Blocks(js=) hook awaits.
    _LAUNCH_JS = "() => {\n" + _PATCH_BODY + "\n}"
    # `head=` form: raw <script> tag injected directly into <head> at server
    # render time. Wrapped in an IIFE so internal const/let don't leak.
    _LAUNCH_HEAD = "<script>(function(){\n" + _PATCH_BODY + "\n})();</script>"

    demo = create_interface(theme=get_theme(), css=CUSTOM_CSS, js=_LAUNCH_JS)
    demo.queue(default_concurrency_limit=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        inbrowser=True,
        allowed_paths=[tempfile.gettempdir()],
        max_threads=40,
        head=_LAUNCH_HEAD,
    )


if __name__ == "__main__":
    main()

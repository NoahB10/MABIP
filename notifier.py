"""Push a message to a phone when the rig needs a human.

Why this exists
---------------
A well-plate run takes hours and nobody sits next to it. The GUI already
detects when the metabolite signal stops cycling (blockage_detector.py) and
parks the plate in the buffer — but it told you with a popup on a screen in
another room. This module carries the same events out of the building.

Channel: ntfy (https://ntfy.sh). One plain HTTP POST per message, no account,
no API key, no extra Python dependency (stdlib urllib). Subscribe to the same
topic in the ntfy app (Android / iOS) or at https://ntfy.sh/<topic> in any
browser, and every send() lands there as a push notification within seconds.
A topic is just a string; anyone who knows it can read it, so the default is
a random one ("mabip-" + 8 hex), not a guessable name.

Also mirrors each message to the Pi's desktop (notify-send) when available, so
someone standing at the rig sees it without the GUI having focus.

Design points
-------------
* send() never blocks and never raises. Delivery happens on a daemon thread;
  the GUI's asyncio loop (sensor reader, pump, robot) must not stall on a slow
  network. Failures are logged, not surfaced — a dead notification is a
  nuisance, a stalled run is a ruined plate.
* Per-key rate limit. The detector only emits on state edges, but a chattering
  edge (or a future caller in a loop) must not produce a notification storm:
  same `key` within min_gap_s is dropped.
* The JSON publish endpoint (POST to the server root) rather than per-topic
  headers: bodies are UTF-8, HTTP headers are not, and messages carry
  en-dashes and the odd emoji.
* send_now() is the synchronous variant for the Settings dialog's "Send test"
  button: it returns (ok, error_text) so the user sees why a test failed.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["Notifier", "DEFAULT_SERVER", "PRIORITIES"]

DEFAULT_SERVER = "https://ntfy.sh"

# ntfy priority names → numeric levels (5 = urgent, 1 = min).
PRIORITIES = {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}


class Notifier:
    """Fire-and-forget push notifications over ntfy, plus a desktop mirror."""

    def __init__(self, enabled: bool = False, topic: str = "",
                 server: str = DEFAULT_SERVER, desktop: bool = True,
                 min_gap_s: float = 30.0, timeout_s: float = 10.0):
        self.enabled = bool(enabled)
        self.topic = (topic or "").strip()
        self.server = (server or DEFAULT_SERVER).strip().rstrip("/")
        self.desktop = bool(desktop)
        self.min_gap_s = float(min_gap_s)
        self.timeout_s = float(timeout_s)
        self._last_sent: dict = {}
        self._lock = threading.Lock()
        # Seam for tests: replaced with a fake that records the request.
        self._urlopen = urllib.request.urlopen
        self.sent_count = 0
        self.failed_count = 0
        self.last_error: Optional[str] = None

    # ---- configuration ----------------------------------------------------
    @staticmethod
    def new_topic() -> str:
        """A fresh, unguessable topic name."""
        return "mabip-" + secrets.token_hex(4)

    def configure(self, enabled=None, topic=None, server=None, desktop=None) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if topic is not None:
            self.topic = topic.strip()
        if server is not None:
            self.server = (server or DEFAULT_SERVER).strip().rstrip("/")
        if desktop is not None:
            self.desktop = bool(desktop)

    @property
    def topic_url(self) -> str:
        """Where to subscribe (browser or the ntfy app)."""
        return f"{self.server}/{self.topic}" if self.topic else ""

    @property
    def push_ready(self) -> bool:
        return self.enabled and bool(self.topic)

    # ---- sending ----------------------------------------------------------
    def send(self, title: str, message: str, priority: str = "default",
             tags: Iterable[str] = (), key: Optional[str] = None) -> bool:
        """Queue a notification. Returns True if it was queued (not delivered).

        `key` groups messages for rate limiting: a second message with the same
        key inside min_gap_s is dropped. Pass None to bypass the limit."""
        if not self.enabled:
            return False
        now = time.monotonic()
        with self._lock:
            if key is not None:
                last = self._last_sent.get(key)
                if last is not None and now - last < self.min_gap_s:
                    logger.debug(f"Notification '{key}' suppressed (rate limit)")
                    return False
                self._last_sent[key] = now
        t = threading.Thread(
            target=self._deliver, args=(title, message, priority, tuple(tags)),
            name="notifier", daemon=True)
        t.start()
        return True

    def send_now(self, title: str, message: str, priority: str = "default",
                 tags: Iterable[str] = ()) -> Tuple[bool, str]:
        """Deliver synchronously, ignoring `enabled` and the rate limit.

        For the Settings dialog's test button. Returns (ok, detail)."""
        if not self.topic:
            return False, "No topic set."
        try:
            self._push(title, message, priority, tuple(tags))
        except Exception as e:  # noqa: BLE001 — report any failure to the user
            return False, self._describe(e)
        if self.desktop:
            self._desktop_mirror(title, message)
        return True, f"Sent to {self.topic_url}"

    # ---- delivery ---------------------------------------------------------
    def _deliver(self, title, message, priority, tags) -> None:
        if self.desktop:
            self._desktop_mirror(title, message)
        if not self.topic:
            return
        try:
            self._push(title, message, priority, tags)
            self.sent_count += 1
            self.last_error = None
        except Exception as e:  # noqa: BLE001 — never let a notification raise
            self.failed_count += 1
            self.last_error = self._describe(e)
            logger.warning(f"Push notification failed: {self.last_error}")

    def _push(self, title, message, priority, tags) -> None:
        body = {
            "topic": self.topic,
            "title": title,
            "message": message,
            "priority": PRIORITIES.get(priority, PRIORITIES["default"]),
        }
        if tags:
            body["tags"] = list(tags)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.server + "/", data=data, method="POST",
            headers={"Content-Type": "application/json"})
        with self._urlopen(req, timeout=self.timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status >= 300:
                raise RuntimeError(f"HTTP {status}")

    def _desktop_mirror(self, title, message) -> None:
        exe = shutil.which("notify-send")
        if not exe:
            return
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        try:
            subprocess.run([exe, "-u", "critical", "-a", "MABIP", title, message],
                           env=env, timeout=5, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"notify-send failed: {e}")

    @staticmethod
    def _describe(e: Exception) -> str:
        if isinstance(e, urllib.error.HTTPError):
            return f"HTTP {e.code} from server"
        if isinstance(e, urllib.error.URLError):
            return f"Network error: {e.reason}"
        return f"{type(e).__name__}: {e}"

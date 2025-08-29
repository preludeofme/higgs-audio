# SPDX-License-Identifier: Apache-2.0
"""
Minimal sidecar Live Player server (no extra deps) using Python's http.server and SSE.
- GET /live-player  -> HTML + JS UI with WebAudio, connects to SSE at /live-audio
- GET /live-audio   -> Server-Sent Events stream; broadcasts JSON messages published in-process

Import and use from app_gradio.py:

    from webui.live_player_server import start_server, publish_chunk, publish_reset
    start_server(port=7861)  # once at startup (non-blocking)
    publish_chunk(sr, b64_payload)  # whenever a new Float32 PCM chunk is ready

Messages sent to clients:
- data: {"type":"chunk","sr":<int>,"b64":"<base64 Float32LE mono>"}
- data: {"type":"reset"}

This runs in the same process; no CORS issues since the iframe loads the player from this origin.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Tuple

# ---------------- Broadcast Hub ----------------

class _BroadcastHub:
    def __init__(self) -> None:
        self._subs: List[Tuple[BaseHTTPRequestHandler, threading.Event]] = []
        self._lock = threading.RLock()

    def add(self, handler: BaseHTTPRequestHandler) -> threading.Event:
        stop_evt = threading.Event()
        with self._lock:
            self._subs.append((handler, stop_evt))
        return stop_evt

    def remove(self, handler: BaseHTTPRequestHandler) -> None:
        with self._lock:
            self._subs = [(h, e) for (h, e) in self._subs if h is not handler]

    def broadcast(self, obj: dict) -> None:
        data = f"data: {json.dumps(obj)}\n\n".encode("utf-8")
        to_remove: List[BaseHTTPRequestHandler] = []
        with self._lock:
            for h, _evt in list(self._subs):
                try:
                    h.wfile.write(data)
                    h.wfile.flush()
                except Exception:
                    to_remove.append(h)
            for h in to_remove:
                self._subs = [(hh, ee) for (hh, ee) in self._subs if hh is not h]

    def ping_all(self) -> None:
        """Send a comment to keep connections alive."""
        keep = b": ping\n\n"
        to_remove: List[BaseHTTPRequestHandler] = []
        with self._lock:
            for h, _evt in list(self._subs):
                try:
                    h.wfile.write(keep)
                    h.wfile.flush()
                except Exception:
                    to_remove.append(h)
            for h in to_remove:
                self._subs = [(hh, ee) for (hh, ee) in self._subs if hh is not h]

_hub = _BroadcastHub()

# ---------------- Public API ----------------

_server_thread: threading.Thread | None = None
_httpd: HTTPServer | None = None


def start_server(host: str = "127.0.0.1", port: int = 7861) -> None:
    """Start the sidecar server in a background thread if not running."""
    global _server_thread, _httpd
    if _server_thread and _server_thread.is_alive():
        return

    class Handler(BaseHTTPRequestHandler):
        server_version = "HiggsLivePlayer/0.1"

        def log_message(self, fmt: str, *args) -> None:
            # Quieter logs
            return

        def _send_common_headers(self, status=HTTPStatus.OK, content_type="text/html; charset=utf-8", cache=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if not cache:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            # Allow embedding in iframe from Gradio (same machine)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

        def do_GET(self):  # noqa: N802 (http method)
            if self.path.startswith("/live-player"):
                self._send_common_headers(HTTPStatus.OK, "text/html; charset=utf-8")
                self.wfile.write(_PLAYER_HTML.encode("utf-8"))
                return
            if self.path.startswith("/live-audio"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                stop_evt = _hub.add(self)
                # Initial hello
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                except Exception:
                    _hub.remove(self)
                    return
                # Keep the handler thread alive; hub pushes data
                try:
                    while not stop_evt.is_set():
                        time.sleep(10)
                        _hub.ping_all()
                finally:
                    _hub.remove(self)
                return
            # Fallback 404
            self._send_common_headers(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8")
            self.wfile.write(b"Not Found")

    _httpd = HTTPServer((host, port), Handler)

    def serve():
        try:
            _httpd.serve_forever(poll_interval=0.5)
        except Exception:
            pass

    _server_thread = threading.Thread(target=serve, daemon=True)
    _server_thread.start()


def stop_server() -> None:
    global _httpd, _server_thread
    if _httpd is not None:
        try:
            _httpd.shutdown()
        except Exception:
            pass
        _httpd.server_close()
        _httpd = None
    if _server_thread is not None:
        _server_thread.join(timeout=1.0)
        _server_thread = None


def publish_chunk(sr: int, b64: str) -> None:
    """Publish a PCM chunk to all connected clients.
    Args:
        sr: sampling rate (int)
        b64: base64 of Float32LE mono PCM
    """
    _hub.broadcast({"type": "chunk", "sr": int(sr), "b64": str(b64)})


def publish_reset() -> None:
    """Signal clients to reset their playback queue."""
    _hub.broadcast({"type": "reset"})


_PLAYER_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Higgs Live Player</title>
  <style>
    :root { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { margin: 0; padding: 12px; background: #0b1220; color: #e5e7eb; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 12px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .btn { appearance: none; border: 1px solid #374151; background: #1f2937; color: #e5e7eb; padding: 8px 12px; border-radius: 8px; cursor: pointer; }
    .btn:hover { background: #374151; }
    .pill { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid #374151; }
    .ok { color: #10b981; border-color: #065f46; }
    .bad { color: #ef4444; border-color: #7f1d1d; }
    .muted { color: #9ca3af; }
    #queue { font-variant-numeric: tabular-nums; }
    iframe { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div class="card">
    <div class="row">
      <button id="enable" class="btn">Enable</button>
      <button id="beep" class="btn">Test Beep</button>
      <span id="status" class="pill bad">Disconnected</span>
      <span id="queue" class="pill muted">0.00s queued</span>
      <span id="errors" class="pill muted">—</span>
    </div>
  </div>

  <script>
    (() => {
      let audioCtx = null;
      let sampleRate = 48000; // will be replaced by first chunk's sr
      let nextTime = 0;
      let connected = false;
      let totalQueuedSec = 0;

      const statusEl = document.getElementById('status');
      const queueEl = document.getElementById('queue');
      const errorsEl = document.getElementById('errors');
      const enableBtn = document.getElementById('enable');
      const beepBtn = document.getElementById('beep');

      function setStatus(ok) {
        connected = ok;
        statusEl.textContent = ok ? 'Connected' : 'Disconnected';
        statusEl.className = 'pill ' + (ok ? 'ok' : 'bad');
      }

      function ensureCtx() {
        if (!audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          nextTime = audioCtx.currentTime + 0.05;
        }
      }

      function updateQueueDisplay() {
        const now = audioCtx ? audioCtx.currentTime : 0;
        const queued = Math.max(0, nextTime - now);
        queueEl.textContent = queued.toFixed(2) + 's queued';
      }

      function scheduleBuffer(float32, sr) {
        ensureCtx();
        if (audioCtx.state === 'suspended') {
          audioCtx.resume();
        }
        if (sr && sr !== sampleRate) {
          sampleRate = sr;
        }
        const buf = audioCtx.createBuffer(1, float32.length, sampleRate);
        buf.getChannelData(0).set(float32);
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        const startAt = Math.max(nextTime, audioCtx.currentTime + 0.02);
        try { src.start(startAt); } catch (e) { console.warn('start failed', e); }
        nextTime = startAt + buf.duration;
        updateQueueDisplay();
      }

      function handleReset() {
        if (!audioCtx) return;
        // Create a zero-length buffer to force end; then reset clock
        nextTime = audioCtx.currentTime + 0.02;
        updateQueueDisplay();
      }

      function decodeBase64ToFloat32(b64) {
        const binStr = atob(b64);
        const len = binStr.length / 4;
        const bytes = new Uint8Array(binStr.length);
        for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
        const f32 = new Float32Array(bytes.buffer);
        return new Float32Array(f32); // copy
      }

      function connectSSE() {
        try {
          const es = new EventSource('/live-audio');
          es.onopen = () => setStatus(true);
          es.onerror = () => setStatus(false);
          es.onmessage = (ev) => {
            if (!ev.data) return;
            try {
              const obj = JSON.parse(ev.data);
              if (obj.type === 'chunk') {
                const f32 = decodeBase64ToFloat32(obj.b64);
                scheduleBuffer(f32, obj.sr || sampleRate);
              } else if (obj.type === 'reset') {
                handleReset();
              }
            } catch (e) {
              errorsEl.textContent = 'ERR: ' + (e && e.message ? e.message : String(e));
            }
          };
        } catch (e) {
          errorsEl.textContent = 'SSE failed';
        }
      }

      enableBtn.addEventListener('click', () => {
        ensureCtx();
        audioCtx.resume();
        connectSSE();
      });

      beepBtn.addEventListener('click', () => {
        ensureCtx();
        // 0.3s test tone
        const dur = 0.3;
        const freq = 440;
        const n = Math.floor(sampleRate * dur);
        const f32 = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          f32[i] = 0.2 * Math.sin(2 * Math.PI * freq * (i / sampleRate));
        }
        scheduleBuffer(f32, sampleRate);
      });

      setInterval(() => { if (audioCtx) updateQueueDisplay(); }, 200);
    })();
  </script>
</body>
</html>
"""

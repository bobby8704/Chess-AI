"""
Production entry point: waitress on $PORT, one process.

ONE PROCESS is a requirement, not a tuning choice — game sessions live in
app.py's module dicts, so a second worker process would see none of the
first's games. Scale within the process by threads; AI CPU is additionally
bounded by app.py's CHESS_MAX_CONCURRENT_AI semaphore, so extra threads serve
static files and light routes while at most that many searches run.

    pip install -r requirements-serve.txt
    python serve.py                       # PORT env respected, default 5000

Knobs (all env vars): PORT, CHESS_MAX_CONCURRENT_AI, CHESS_RATE_LIMIT_PER_MIN,
CHESS_MAX_LIVE_GAMES, CHESS_GAME_TTL_S, CHESS_TABLEBASE, CHESS_ADMIN_TOKEN.
"""

import os
import sys

# Line-buffer BOTH streams before app import (app prints its model resolution at
# import time). Piped stdout — every log collector, Render included — is otherwise
# block-buffered by Python, so the lines that prove WHICH model is being served sat
# invisible in a 4KB buffer while onnxruntime's unbuffered C++ stderr got through.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from waitress import serve

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Serving chess AI on 0.0.0.0:{port} (waitress, single process)")
    serve(app, host="0.0.0.0", port=port, threads=8)

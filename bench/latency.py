"""
Single-request serving latency through the exact serving path.

The preset table in app.py used to quote median/p90 latencies that existed nowhere in
bench/results/ — the numbers came from ad-hoc runs that were never written down. This
records them: fresh player per move (as app.py builds one), temperature=0,
add_noise=False, model resolved exactly as serving resolves it (int8 preferred), on an
evenly-spaced sample of the committed ACPL suite so the distribution is over real
middlegame positions rather than best-of-N on a hand-picked few.

Run on an idle box — this measures serving latency, not contention.

Usage:
    python bench/latency.py                     # sims 100/600/1300, 48 positions
    python bench/latency.py --sims 600 --limit 24 --label smoke
"""

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import chess

from mcts import MCTSPlayer, MCTSConfig

RESULTS_DIR = os.path.join(_HERE, "results")
SUITE_PATH = os.path.join(_HERE, "suites", "acpl.json")


def load_positions(limit):
    """Evenly-spaced sample of the frozen suite — same rationale as strength.sample():
    the suite is sorted by ply, so a prefix would be an early-game slice, not a sample."""
    with open(SUITE_PATH) as f:
        positions = json.load(f)["positions"]
    if not limit or limit >= len(positions):
        return positions
    step = len(positions) / limit
    return [positions[int(i * step)] for i in range(limit)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, nargs="+", default=[100, 600, 1300])
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--label", default="int8")
    args = ap.parse_args()

    entries = load_positions(args.limit)

    # Resolve the session the way serving does, and record WHICH model answered —
    # a latency number that cannot name its model is the mistake this file replaces.
    from inference import get_onnx_session
    session = get_onnx_session()
    if session is None:
        # With no session AND model=None the search runs on UNIFORM priors — it still
        # produces plausible-looking timings, which is exactly how this project once
        # nearly deployed a random-move AI. Refuse rather than measure the wrong engine.
        raise SystemExit("FATAL: no ONNX session (is this the right interpreter? "
                         "the project venv is .venv/Scripts/python.exe) — refusing to "
                         "time a uniform-prior search and call it serving latency.")
    model_name = os.path.basename(session.path)
    threads = os.environ.get("CHESS_NUM_THREADS", "4 (default)")
    print(f"model: {model_name}   threads: {threads}   positions: {len(entries)}")

    # Warm-up: session init and import costs must not land in the first timing.
    warm = MCTSPlayer(model=None, config=MCTSConfig(
        num_simulations=50, temperature=0, add_noise=False))
    warm.select_move(chess.Board())

    out = {"model": model_name, "threads": str(threads),
           "n_positions": len(entries), "presets": {}}
    for sims in args.sims:
        times = []
        for e in entries:
            board = chess.Board(e["fen"])
            player = MCTSPlayer(model=None, config=MCTSConfig(
                num_simulations=sims, temperature=0, add_noise=False))
            t0 = time.perf_counter()
            player.select_move(board)
            times.append((time.perf_counter() - t0) * 1000.0)
        s = sorted(times)
        med = statistics.median(s)
        p90 = s[min(len(s) - 1, int(round(0.9 * len(s))) - 1)] if len(s) >= 10 else s[-1]
        out["presets"][str(sims)] = {
            "median_ms": round(med, 1), "p90_ms": round(p90, 1),
            "mean_ms": round(statistics.mean(s), 1),
            "min_ms": round(s[0], 1), "max_ms": round(s[-1], 1),
            "times_ms": [round(t, 1) for t in times],
        }
        print(f"  sims {sims:>5}: median {med:7.0f} ms   p90 {p90:7.0f} ms   "
              f"range [{s[0]:.0f}, {s[-1]:.0f}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"latency_{args.label}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

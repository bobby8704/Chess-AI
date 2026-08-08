"""
Where does a move's time actually go — measured, not derived.

Every prior estimate of the post-int8 forward-vs-Python split was arithmetic on
other measurements, and derived numbers are 0-for-3 on this project (ACPL preset
cut, value-head screen, "quiescence is ~55%"). This wraps the real functions in
perf_counter and attributes a move's wall clock directly:

  select_move total
    evaluate_full    root only: heuristic pipeline + blunder scan
    evaluate_fast    per leaf: ONNX policy + quiescence value, split into
      onnx_run         session.run only
      tensorize        board_to_tensor_2d
      legal_idx        legal_move_indices
      quiescence       evaluate_quiescence
    remainder        tree walk: select_child loops, node creation, backprop, vetoes

  Overlap colour (counted INSIDE the buckets above, reported separately):
      board_copy       chess.Board.copy
      board_push       chess.Board.push

MCTSPlayer methods are wrapped on the CLASS before the player is constructed —
__init__ binds them into MCTS, and patching an instance afterwards silently does
nothing (the lesson a 120-game elo run once taught).

Run on an IDLE box:  .venv/Scripts/python.exe bench/profile_move.py
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import chess

BUCKETS = defaultdict(float)
COUNTS = defaultdict(int)


def _timed(fn, key):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            BUCKETS[key] += time.perf_counter() - t0
            COUNTS[key] += 1
    return wrapper


def instrument():
    import evaluation
    import features
    import inference

    # These three are imported function-locally per call inside mcts, so a module
    # attribute patch is picked up on every call.
    evaluation.evaluate_quiescence = _timed(evaluation.evaluate_quiescence, "quiescence")
    features.board_to_tensor_2d = _timed(features.board_to_tensor_2d, "tensorize")
    inference.legal_move_indices = _timed(inference.legal_move_indices, "legal_idx")

    session = inference.get_onnx_session()
    if session is None:
        raise SystemExit("FATAL: no ONNX session — wrong interpreter? "
                         "Use .venv/Scripts/python.exe")
    session.run = _timed(session.run, "onnx_run")

    from mcts import MCTSPlayer
    MCTSPlayer._evaluate_fast = _timed(MCTSPlayer._evaluate_fast, "evaluate_fast")
    MCTSPlayer._evaluate_full = _timed(MCTSPlayer._evaluate_full, "evaluate_full")

    chess.Board.copy = _timed(chess.Board.copy, "board_copy")
    chess.Board.push = _timed(chess.Board.push, "board_push")

    # Finer split INSIDE quiescence (overlap colour too): the root draw-claim probe
    # and the stand-pat evaluation are the two candidates for where its time hides,
    # and they need very different treatment in a native port.
    chess.Board.can_claim_draw = _timed(chess.Board.can_claim_draw, "can_claim_draw")
    evaluation._evaluate_raw = _timed(evaluation._evaluate_raw, "evaluate_raw")


def load_positions(limit):
    with open(os.path.join(_HERE, "suites", "acpl.json")) as f:
        positions = json.load(f)["positions"]
    step = len(positions) / limit
    return [positions[int(i * step)] for i in range(limit)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=1300)
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    instrument()
    from mcts import MCTSPlayer, MCTSConfig

    entries = load_positions(args.limit)
    player = MCTSPlayer(model=None, config=MCTSConfig(
        num_simulations=args.sims, temperature=0, add_noise=False))
    player.select_move(chess.Board())        # warm-up; not counted
    BUCKETS.clear()
    COUNTS.clear()

    t0 = time.perf_counter()
    for e in entries:
        player.select_move(chess.Board(e["fen"]))
    total = time.perf_counter() - t0

    n = len(entries)
    fast = BUCKETS["evaluate_fast"]
    full = BUCKETS["evaluate_full"]
    inner_keys = ("onnx_run", "tensorize", "legal_idx", "quiescence")
    inner = sum(BUCKETS[k] for k in inner_keys)
    fast_overhead = fast - inner            # leaf_value glue, dict building, softmax
    remainder = total - fast - full         # tree walk + vetoes

    def row(name, secs, calls=None, indent=0):
        pct = 100.0 * secs / total
        per_call = f"   {secs / calls * 1e6:8.1f} us x {calls}" if calls else ""
        print(f"  {' ' * indent}{name:<26} {secs / n * 1000:8.1f} ms/move  "
              f"{pct:5.1f}%{per_call}")

    print(f"\n{args.sims} sims, {n} positions, {total / n * 1000:.0f} ms/move average\n")
    row("evaluate_full (root)", full, COUNTS["evaluate_full"])
    row("evaluate_fast (leaves)", fast, COUNTS["evaluate_fast"])
    for k in inner_keys:
        row(k, BUCKETS[k], COUNTS[k], indent=2)
    row("fast overhead (glue)", fast_overhead, indent=2)
    row("tree walk + vetoes", remainder)
    print()
    row("of which board_copy", BUCKETS["board_copy"], COUNTS["board_copy"])
    row("of which board_push", BUCKETS["board_push"], COUNTS["board_push"])
    row("of which can_claim_draw", BUCKETS["can_claim_draw"], COUNTS["can_claim_draw"])
    row("of which evaluate_raw", BUCKETS["evaluate_raw"], COUNTS["evaluate_raw"])

    out = {"sims": args.sims, "n_positions": n, "total_s": round(total, 3),
           "ms_per_move": round(total / n * 1000, 1),
           "buckets_s": {k: round(v, 4) for k, v in BUCKETS.items()},
           "counts": dict(COUNTS)}
    path = os.path.join(_HERE, "results", f"profile_move_{args.sims}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

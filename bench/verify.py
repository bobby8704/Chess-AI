"""
Move-equivalence and latency harness for the MCTS engine.

Two jobs:

  run     — play every position in the suite at several simulation counts, recording
            the chosen move, the FULL root visit distribution, and wall-clock time.
  compare — diff two runs. A change is "move-preserving" only if every chosen move
            AND every visit distribution matches exactly.

The player is built exactly the way app.py builds it (temperature=0, add_noise=False),
so what is measured here is what a real web move does.

Usage:
    python bench/verify.py run  baseline
    python bench/verify.py run  after-lazy-expand
    python bench/verify.py compare baseline after-lazy-expand
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.positions import build_suite, rebuild  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_SIMS = (200, 600, 1300)


def _make_player(dual_model, num_sims):
    """Mirror app.py:_make_player exactly."""
    from mcts import MCTSPlayer, MCTSConfig
    config = MCTSConfig(num_simulations=num_sims, temperature=0, add_noise=False)
    return MCTSPlayer(model=dual_model, config=config)


def run(label, sims=DEFAULT_SIMS, reps=3, threads=None):
    import torch
    if threads is not None:
        torch.set_num_threads(threads)

    from neural_network import load_dual_model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "dual_model_mcts.pt",
    )
    dual_model = load_dual_model(model_path)

    suite = build_suite()
    out = {
        "label": label,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "reps": reps,
        "positions": {},
    }

    for entry in suite:
        for n in sims:
            key = f"{entry['name']}@{n}"
            best_ms = None
            record = None
            for _ in range(reps):
                board = rebuild(entry)          # fresh board WITH move stack
                player = _make_player(dual_model, n)
                t0 = time.perf_counter()
                move, probs = player.select_move(board, return_policy=True)
                ms = (time.perf_counter() - t0) * 1000.0
                rec = {
                    "move": move.uci() if move else None,
                    # ordered as the engine produced it, so key-set changes are visible
                    "visits": {m.uci(): round(p, 12) for m, p in probs.items()},
                }
                if record is None:
                    record = rec
                elif rec != record:
                    record["NONDETERMINISTIC"] = True
                if best_ms is None or ms < best_ms:
                    best_ms = ms
            record["ms"] = round(best_ms, 1)
            record["ply"] = entry["ply"]
            out["positions"][key] = record
            print(f"  {key:<18} {record['move']:<6} {record['ms']:>9.1f} ms"
                  f"{'  <== NONDETERMINISTIC' if record.get('NONDETERMINISTIC') else ''}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")
    return out


def compare(label_a, label_b):
    with open(os.path.join(RESULTS_DIR, f"{label_a}.json")) as f:
        a = json.load(f)
    with open(os.path.join(RESULTS_DIR, f"{label_b}.json")) as f:
        b = json.load(f)

    keys = sorted(set(a["positions"]) | set(b["positions"]))
    move_diffs, visit_diffs, rows = [], [], []
    tot_a = tot_b = 0.0

    for k in keys:
        ra, rb = a["positions"].get(k), b["positions"].get(k)
        if ra is None or rb is None:
            move_diffs.append((k, "MISSING", "MISSING"))
            continue
        if ra["move"] != rb["move"]:
            move_diffs.append((k, ra["move"], rb["move"]))
        if ra["visits"] != rb["visits"]:
            if set(ra["visits"]) != set(rb["visits"]):
                detail = (f"key set differs "
                          f"({len(ra['visits'])} -> {len(rb['visits'])} moves)")
            else:
                worst = max(abs(ra["visits"][m] - rb["visits"][m]) for m in ra["visits"])
                detail = f"max |delta visit prob| = {worst:.2e}"
            visit_diffs.append((k, detail))
        tot_a += ra["ms"]
        tot_b += rb["ms"]
        rows.append((k, ra["ply"], ra["ms"], rb["ms"], ra["ms"] / rb["ms"] if rb["ms"] else 0))

    print(f"\n{'position':<18}{'ply':>5}{label_a[:11]:>13}{label_b[:11]:>13}{'speedup':>10}")
    print("-" * 59)
    for k, ply, ma, mb, sp in rows:
        print(f"{k:<18}{ply:>5}{ma:>12.1f}m{mb:>12.1f}m{sp:>9.2f}x")
    print("-" * 59)
    print(f"{'TOTAL':<18}{'':>5}{tot_a:>12.1f}m{tot_b:>12.1f}m"
          f"{(tot_a / tot_b if tot_b else 0):>9.2f}x")

    print(f"\nchosen-move differences : {len(move_diffs)} / {len(rows)}")
    for k, x, y in move_diffs:
        print(f"    {k}: {x} -> {y}")
    print(f"visit-distribution diffs: {len(visit_diffs)} / {len(rows)}")
    for k, d in visit_diffs:
        print(f"    {k}: {d}")

    ok = not move_diffs and not visit_diffs
    print("\n" + ("MOVE-PRESERVING: identical moves and identical visit distributions"
                  if ok else "NOT move-preserving — see differences above"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--sims", type=int, nargs="+", default=list(DEFAULT_SIMS))
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--threads", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("label_a")
    c.add_argument("label_b")
    args = ap.parse_args()

    if args.cmd == "run":
        run(args.label, sims=tuple(args.sims), reps=args.reps, threads=args.threads)
    else:
        sys.exit(0 if compare(args.label_a, args.label_b) else 1)

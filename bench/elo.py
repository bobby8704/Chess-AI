"""
Head-to-head Elo measurement: engine versus engine, and engine versus Stockfish.

Why this exists even though bench/strength.py already measures strength. ACPL is
one-move agreement with depth-16 Stockfish on a static position. It never plays a game
out, so it cannot see error compounding, conversion of a won position, or endgame
technique — and it carries a ~7.3cp floor even when the engine plays Stockfish's own
best move. It is a useful, cheap proxy. It is NOT Elo. Nothing in this project had ever
been validated by actually playing games, so "the PUCT change is worth -10.33cp" could
have meant 30 Elo or 3.

Design decisions that matter for variance, because games are expensive:

  * PAIRED OPENINGS WITH COLOUR REVERSAL. Every opening is played twice, once with each
    engine as White. Statistics are computed over PAIRS, not games. This cancels both
    opening bias and colour bias, and it is why a few hundred games can say anything at
    all — unpaired, the same precision needs several times more games.
  * NEAR-BALANCED OPENINGS. Positions are drawn from the committed ACPL suite filtered
    to |base_cp| <= OPENING_BALANCE_CP, using the Stockfish evaluations already cached
    there. Starting from decided positions wastes games on foregone conclusions.
  * INDEPENDENT ADJUDICATION. Games are adjudicated by Stockfish, not by the engine's
    own evaluation, which would be circular. Without adjudication a weak engine shuffles
    for hundreds of plies and the run never finishes.
  * TABLEBASE PROBE DISABLED. select_move otherwise makes a blocking HTTPS call to
    tablebase.lichess.ovh at <=7 pieces. A measurement must not depend on a third-party
    service, and perfect endgame play from an oracle would mask exactly the endgame
    differences we are trying to measure. Both arms are affected identically.

Usage:
    python bench/elo.py h2h --a current:600 --b inf-rule:600 --pairs 150
    python bench/elo.py gauntlet --a current:600 --levels 0,2,4 --pairs 40
    python bench/elo.py variants
"""

import argparse
import json
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import chess  # noqa: E402
import chess.engine  # noqa: E402

from bench.strength import (  # noqa: E402
    SF_PATH, RESULTS_DIR, open_engine, analyse, rebuild, load_suite, sample,
    normal_cdf, MATE_SCORE,
)

OPENING_BALANCE_CP = 80     # only start from roughly level positions
ADJ_DEPTH = 10              # Stockfish depth for adjudication (cheap, independent)
RESIGN_CP = 700             # |score| this large for RESIGN_PLIES -> adjudicate a win
RESIGN_PLIES = 6
DRAW_CP = 15                # |score| this small for DRAW_PLIES after DRAW_AFTER -> draw
DRAW_PLIES = 10
DRAW_AFTER = 60
MAX_PLIES = 250


# ---------------------------------------------------------------------------
# Engine variants
# ---------------------------------------------------------------------------
#
# Both engines must play inside ONE process, so a variant is a monkeypatch rather than a
# git revision. That is a real limitation: it can only express changes small enough to
# re-implement as a patch. For anything larger the honest options are a UCI-style
# subprocess engine, or bench/strength.py's cross-commit worktree pattern.
#
# CRITICAL: a variant patches CLASS attributes, which are global to the process. Applying
# both arms' patches at startup would make both engines identical — and it did, silently:
# the first run of this harness reported 12 games where every colour-reversed pair had an
# identical ply count, because it was measuring inf-rule against itself. So a variant is
# expressed as a set of overrides and ACTIVATED immediately before each move, for
# whichever arm is to play. Switching is just attribute assignment, so it is free.

def _overrides_current():
    """Ship-as-is."""
    return {}


def _overrides_inf_rule():
    """Restore the pre-2964a7b selection rule: unvisited children score float('inf')."""
    import math as _m
    import mcts as M

    def ucb_score(self, c_puct, parent_visits):
        if self.visit_count == 0:
            return float('inf')
        exploration = c_puct * self.prior * _m.sqrt(parent_visits) / (1 + self.visit_count)
        return -self.value + exploration

    def select_child(self, c_puct):
        best_score, best_move, best_child = float('-inf'), None, None
        for move in self.child_priors:
            child = self.children.get(move)
            score = float('inf') if child is None else child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        if best_move is None:
            return None, None
        if best_child is None:
            cb = self.board.copy(stack=M.TREE_STACK_DEPTH)
            cb.push(best_move)
            best_child = M.MCTSNode(board=cb, parent=self, move=best_move,
                                    prior=self.child_priors[best_move], own_board=True)
            self.children[best_move] = best_child
        return best_move, best_child

    # Reverting BOTH halves matters: the live first-play-urgency is select_child's inline
    # term, and ucb_score's zero-visit branch is unreachable in the shipped engine.
    return {("MCTSNode", "ucb_score"): ucb_score,
            ("MCTSNode", "select_child"): select_child}


VARIANTS = {
    "current": _overrides_current,
    "inf-rule": _overrides_inf_rule,
}

_PRISTINE = {}
_OVERRIDES = {}
_ACTIVE = None


def _prepare_variants(names):
    """Capture pristine class attributes, then build each variant's override table."""
    import mcts as M
    global _ACTIVE
    for name in names:
        ov = VARIANTS[name]()
        for (cls_name, attr) in ov:
            key = (cls_name, attr)
            if key not in _PRISTINE:
                _PRISTINE[key] = getattr(getattr(M, cls_name), attr)
        _OVERRIDES[name] = ov
    _ACTIVE = None


def activate(name):
    """Make `name` the engine that the next select_move call will use."""
    global _ACTIVE
    if name == _ACTIVE:
        return
    import mcts as M
    for (cls_name, attr), val in _PRISTINE.items():        # reset everything first
        setattr(getattr(M, cls_name), attr, val)
    for (cls_name, attr), val in _OVERRIDES[name].items():  # then apply this variant
        setattr(getattr(M, cls_name), attr, val)
    _ACTIVE = name


def parse_arm(spec):
    """'current:600' -> ('current', 600)."""
    if ":" not in spec:
        raise SystemExit(f"arm must look like NAME:SIMS, got {spec!r}")
    name, sims = spec.rsplit(":", 1)
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}")
    return name, int(sims)


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

def build_openings(n):
    """Near-balanced positions from the committed ACPL suite, spread across the suite."""
    pos = [e for e in load_suite("acpl")["positions"]
           if abs(e["base_cp"]) <= OPENING_BALANCE_CP]
    if not pos:
        raise SystemExit("no balanced openings found")
    return sample(pos, n)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ENGINE = None
_PLAYERS = {}
_ARM_VARIANT = {}
_SF_OPPONENT = None
_SF_MOVETIME = 0.1
_OPP_KEY = "b"


def _init_worker(arms, threads, sf_elo=None, sf_movetime=0.1):
    """arms: {'a': (variant, sims), ...}. 'sf' is a reserved arm key for Stockfish."""
    global _ENGINE, _PLAYERS, _ARM_VARIANT, _SF_OPPONENT, _SF_MOVETIME, _OPP_KEY
    os.environ["CHESS_NUM_THREADS"] = str(threads)
    _SF_MOVETIME = sf_movetime
    # Arm 'a' is always the engine under test; its opponent is either the second engine
    # arm 'b' (h2h) or Stockfish (gauntlet).
    _OPP_KEY = "sf" if "sf" in arms else "b"

    if sf_elo is not None:
        # A SECOND Stockfish, deliberately weakened. Kept separate from the adjudicator
        # below, which must stay full strength — sharing one process would silently
        # adjudicate every game with a 1320-rated engine.
        _SF_OPPONENT = chess.engine.SimpleEngine.popen_uci(SF_PATH)
        _SF_OPPONENT.configure({"Threads": 1, "Hash": 16,
                                "UCI_LimitStrength": True, "UCI_Elo": sf_elo})

    _ARM_VARIANT = {k: v for k, (v, _) in arms.items()}
    _prepare_variants(set(_ARM_VARIANT.values()))

    # Kill the network tablebase probe — see module docstring.
    import tablebase
    tablebase.should_probe = lambda board: False

    from inference import get_onnx_session
    session = get_onnx_session()
    model = None
    if session is None:
        from neural_network import load_dual_model, DEFAULT_CKPT_PATH
        model = load_dual_model(DEFAULT_CKPT_PATH)
    if session is None and model is None:
        raise RuntimeError("no model available; refusing to measure a random policy")

    from mcts import MCTSPlayer, MCTSConfig
    for key, (variant, sims) in arms.items():
        if key == "sf":
            continue
        _PLAYERS[key] = MCTSPlayer(
            model=model,
            config=MCTSConfig(num_simulations=sims, temperature=0, add_noise=False))
    _ENGINE = open_engine()


def _adjudicate(board, history):
    """
    Return '1-0' / '0-1' / '1/2-1/2' if the game should stop, else None.

    Uses Stockfish rather than the engine's own evaluation, which would be circular.
    """
    info = analyse(_ENGINE, board, depth=ADJ_DEPTH)
    cp = info["score"].white().score(mate_score=MATE_SCORE)
    history.append(cp)

    if len(history) >= RESIGN_PLIES:
        tail = history[-RESIGN_PLIES:]
        if all(c >= RESIGN_CP for c in tail):
            return "1-0"
        if all(c <= -RESIGN_CP for c in tail):
            return "0-1"
    if len(board.move_stack) >= DRAW_AFTER and len(history) >= DRAW_PLIES:
        if all(abs(c) <= DRAW_CP for c in history[-DRAW_PLIES:]):
            return "1/2-1/2"
    return None


def _play_game(task):
    """Play one game. Returns the score for arm 'a' in {0, 0.5, 1}."""
    opening, a_is_white, opening_idx = task
    board = rebuild(opening["moves"])
    white_key, black_key = (("a", _OPP_KEY) if a_is_white else (_OPP_KEY, "a"))

    history = []
    moves_played = []
    result = None
    t0 = time.perf_counter()
    while True:
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            break
        if len(board.move_stack) - len(opening["moves"]) >= MAX_PLIES:
            info = analyse(_ENGINE, board, depth=ADJ_DEPTH)
            cp = info["score"].white().score(mate_score=MATE_SCORE)
            result = "1-0" if cp >= 200 else "0-1" if cp <= -200 else "1/2-1/2"
            break

        key = white_key if board.turn == chess.WHITE else black_key
        if key == "sf":
            move = _SF_OPPONENT.play(
                board, chess.engine.Limit(time=_SF_MOVETIME)).move
        else:
            activate(_ARM_VARIANT[key])  # per-move: variants are process-global
            move = _PLAYERS[key].select_move(board)
        if move is None:
            result = "1/2-1/2"
            break
        board.push(move)
        moves_played.append(move.uci())

        adj = _adjudicate(board, history)
        if adj:
            result = adj
            break

    if result == "1-0":
        score_a = 1.0 if a_is_white else 0.0
    elif result == "0-1":
        score_a = 0.0 if a_is_white else 1.0
    else:
        score_a = 0.5

    return {
        "opening": opening_idx,
        "a_white": a_is_white,
        "result": result,
        "score_a": score_a,
        "plies": len(board.move_stack) - len(opening["moves"]),
        # Fingerprint of the actual moves. If the two colour-reversed games of a pair
        # share it, the arms are playing identically and the run measures nothing — which
        # is exactly how the class-level monkeypatch bug hid itself the first time.
        # crc32, NOT hash(): Python salts string hashing per process, and the two games
        # of a pair run in different workers, so builtin hash would never match.
        "fp": zlib.crc32(" ".join(moves_played).encode()),
        "s": round(time.perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def elo(score):
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


def pair_stats(games):
    """
    Aggregate into per-opening PAIRS and report Elo with a confidence interval.

    Pairing is the whole point: each opening contributes one number (arm A's share of
    the two games), so opening difficulty and colour advantage cancel instead of
    inflating the variance.
    """
    by_opening = {}
    for g in games:
        by_opening.setdefault(g["opening"], []).append(g["score_a"])
    pairs = [sum(v) / len(v) for v in by_opening.values() if len(v) == 2]
    if len(pairs) < 2:
        return None

    n = len(pairs)
    mean = sum(pairs) / n
    var = sum((x - mean) ** 2 for x in pairs) / (n - 1)
    se = math.sqrt(var / n)

    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    los = normal_cdf((mean - 0.5) / se) if se > 0 else (1.0 if mean > 0.5 else 0.0)
    return {
        "pairs": n, "score": mean, "se": se,
        "elo": elo(mean),
        "elo_lo": elo(max(1e-9, min(1 - 1e-9, lo))),
        "elo_hi": elo(max(1e-9, min(1 - 1e-9, hi))),
        "los": los,
        # Smallest Elo gap this many pairs could resolve at 80% power, alpha 0.05.
        "detectable_elo": abs(elo(min(0.999999, 0.5 + 2.8 * se)) - 0.0),
    }


def _tally(games):
    w = sum(1 for g in games if g["score_a"] == 1.0)
    d = sum(1 for g in games if g["score_a"] == 0.5)
    return w, d, len(games) - w - d


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _run(tasks, arms, workers, threads, label, sf_elo=None, sf_movetime=0.1):
    t0 = time.perf_counter()
    out = []
    with Pool(workers, initializer=_init_worker,
              initargs=(arms, threads, sf_elo, sf_movetime)) as pool:
        for i, g in enumerate(pool.imap_unordered(_play_game, tasks), 1):
            out.append(g)
            if i % 10 == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                w, d, l = _tally(out)
                print(f"  {i}/{len(tasks)} games  {el:5.0f}s  "
                      f"+{w} ={d} -{l}   {(len(tasks)-i)/(i/el):5.0f}s left", flush=True)
    return out, time.perf_counter() - t0


def cmd_h2h(a_spec, b_spec, n_pairs, workers, threads):
    a, b = parse_arm(a_spec), parse_arm(b_spec)
    openings = build_openings(n_pairs)
    # Each opening twice, colours reversed.
    tasks = []
    for i, o in enumerate(openings):
        tasks.append((o, True, i))
        tasks.append((o, False, i))

    print(f"Head-to-head: A={a_spec}  vs  B={b_spec}")
    print(f"  {len(openings)} openings x 2 colours = {len(tasks)} games, "
          f"{workers} workers x {threads} thread(s)")
    games, dt = _run(tasks, {"a": a, "b": b}, workers, threads, "h2h")

    st = pair_stats(games)
    w, d, l = _tally(games)
    plies = sorted(g["plies"] for g in games)

    # Sanity gate. If the colour-reversed games of a pair played out move-for-move
    # identically, the two arms are the same engine and the run measures only colour.
    by_op = {}
    for g in games:
        by_op.setdefault(g["opening"], []).append(g["fp"])
    twins = sum(1 for v in by_op.values() if len(v) == 2 and v[0] == v[1])
    complete = sum(1 for v in by_op.values() if len(v) == 2)
    if complete and twins == complete and a != b:
        print(f"\n  *** ALL {complete} pairs played IDENTICAL games. The two arms are "
              f"behaving as one engine — this run measures nothing. ***")
    elif twins:
        print(f"\n  note: {twins}/{complete} pairs played identical games "
              f"(expected when the arms agree on every move)")

    white_wins = sum(1 for g in games if g["result"] == "1-0")
    black_wins = sum(1 for g in games if g["result"] == "0-1")
    draws = sum(1 for g in games if g["result"] == "1/2-1/2")
    print(f"\n  by colour: White {white_wins}, Black {black_wins}, draws {draws}"
          f"   (a large imbalance here is an engine property, not an arm difference)")

    out = {"a": a_spec, "b": b_spec, "games": len(games), "wdl": [w, d, l],
           "stats": st, "wall_s": round(dt, 1), "rows": games}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"elo_{a[0]}{a[1]}_vs_{b[0]}{b[1]}"
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(out, f)

    print(f"\n  A scored  +{w} ={d} -{l}  of {len(games)} games "
          f"({(w + 0.5 * d) / len(games):.1%})")
    print(f"  median game length {plies[len(plies)//2]} plies, wall clock {dt:.0f}s")
    if st:
        print(f"\n  paired score   {st['score']:.4f} +/- {st['se']:.4f}  ({st['pairs']} pairs)")
        print(f"  Elo(A) - Elo(B) {st['elo']:+.1f}   95% CI [{st['elo_lo']:+.1f}, "
              f"{st['elo_hi']:+.1f}]")
        print(f"  likelihood A is stronger: {st['los']:.1%}")
        print(f"\n  This run could resolve about +/-{st['detectable_elo']:.0f} Elo. "
              f"A null here means 'not measured' below that.")
    print(f"\n  wrote {os.path.join(RESULTS_DIR, name + '.json')}")
    return out


def cmd_gauntlet(a_spec, elos, n_pairs, workers, threads, movetime):
    """Play the engine against Stockfish at calibrated UCI_Elo levels."""
    a = parse_arm(a_spec)
    openings = build_openings(n_pairs)
    print(f"Gauntlet: {a_spec} vs Stockfish at UCI_Elo {elos}")
    print(f"  {len(openings)} openings x 2 colours per level, "
          f"Stockfish {movetime * 1000:.0f} ms/move\n")

    rows = []
    for sf_elo in elos:
        tasks = []
        for i, o in enumerate(openings):
            tasks.append((o, True, i))
            tasks.append((o, False, i))
        print(f"  --- vs UCI_Elo {sf_elo} ---")
        games, dt = _run(tasks, {"a": a, "sf": ("current", 0)}, workers, threads,
                         "gauntlet", sf_elo, movetime)
        st = pair_stats(games)
        w, d, l = _tally(games)
        implied = None
        if st and 0.0 < st["score"] < 1.0:
            implied = sf_elo + st["elo"]
        # Keep the per-game records. Without them an odd aggregate cannot be diagnosed:
        # two gauntlet runs once returned an identical +12 =1 -37 and the only way to
        # show it was coincidence rather than a bug was that their standard errors
        # differed — which needed the pair-level data, not the summary.
        rows.append({"sf_elo": sf_elo, "wdl": [w, d, l], "stats": st,
                     "implied_elo": implied, "wall_s": round(dt, 1), "games": games})
        score = (w + 0.5 * d) / len(games)
        print(f"  +{w} ={d} -{l}  score {score:.1%}"
              + (f"   implied engine Elo ~{implied:.0f}" if implied is not None
                 else "   (shutout: engine Elo is outside this level's range)"))
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"gauntlet_{a[0]}{a[1]}"
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump({"a": a_spec, "movetime": movetime, "rows": rows}, f)

    usable = [r for r in rows if r["implied_elo"] is not None]
    print("=" * 66)
    if usable:
        # Inverse-variance weight, using each level's Elo standard error.
        tot_w = tot = 0.0
        for r in usable:
            se_elo = max(1.0, (r["stats"]["elo_hi"] - r["stats"]["elo_lo"]) / 3.92)
            wt = 1.0 / (se_elo ** 2)
            tot += wt * r["implied_elo"]
            tot_w += wt
        print(f"  ESTIMATED ENGINE RATING: ~{tot / tot_w:.0f} Elo")
        for r in usable:
            print(f"    vs {r['sf_elo']}: implied {r['implied_elo']:.0f}")
    else:
        lo = min(r["sf_elo"] for r in rows)
        allzero = all((r["wdl"][0] + 0.5 * r["wdl"][1]) == 0 for r in rows)
        print(f"  No level produced a usable estimate — the engine "
              f"{'lost every game' if allzero else 'won every game'}.")
        print(f"  Its rating is {'below' if allzero else 'above'} the levels tested "
              f"(lowest was {lo}).")
    print("\n  CAVEAT: UCI_Elo is calibrated for standard time controls. At "
          f"{movetime * 1000:.0f} ms/move Stockfish plays BELOW its nominal rating, so "
          "this\n  anchor flatters the engine and should be read as a soft upper bound. "
          "The\n  engine-vs-engine Elo differences from `h2h` do not have this problem.")
    return rows


def cmd_variants():
    print("engine variants available as arms (NAME:SIMS):\n")
    for name, fn in VARIANTS.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:<12} {doc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("h2h", help="engine vs engine, paired openings")
    h.add_argument("--a", required=True, help="arm A as VARIANT:SIMS, e.g. current:600")
    h.add_argument("--b", required=True, help="arm B as VARIANT:SIMS")
    h.add_argument("--pairs", type=int, default=100)
    h.add_argument("--workers", type=int, default=None)
    h.add_argument("--threads", type=int, default=2)

    g = sub.add_parser("gauntlet", help="engine vs Stockfish at calibrated UCI_Elo")
    g.add_argument("--a", required=True, help="engine arm as VARIANT:SIMS")
    g.add_argument("--elos", default="1320,1500,1700",
                   help="comma-separated Stockfish UCI_Elo levels (min 1320)")
    g.add_argument("--pairs", type=int, default=25)
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--threads", type=int, default=2)
    g.add_argument("--movetime", type=float, default=0.1,
                   help="seconds per Stockfish move")

    sub.add_parser("variants", help="list engine variants")

    args = ap.parse_args()
    if args.cmd == "variants":
        return cmd_variants()

    if args.workers is None:
        cores = os.cpu_count() or 4
        args.workers = max(1, (cores - 2) // max(1, args.threads))
        print(f"(using {args.workers} workers x {args.threads} thread(s) "
              f"on {cores} logical cores)")
    if args.cmd == "gauntlet":
        elos = [int(x) for x in args.elos.split(",")]
        cmd_gauntlet(args.a, elos, args.pairs, args.workers, args.threads, args.movetime)
    else:
        cmd_h2h(args.a, args.b, args.pairs, args.workers, args.threads)


if __name__ == "__main__":
    main()

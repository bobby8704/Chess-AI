"""
Edge-case checks for the lazy-expansion / ONNX search rewrite.

The move-equivalence harness (verify.py) runs the engine exactly as app.py does:
temperature=0, add_noise=False, model loaded. That leaves several rewritten paths
completely untested, which is what this file covers:

  1. Dirichlet noise      — add_dirichlet_noise now edits child_priors, not children.
  2. Terminal nodes       — children are created lazily, so is_game_over() now runs on
                            first selection rather than at expansion.
  3. Uniform fallback     — model=None (what a failed model load produces).
  4. ONNX vs torch        — the two inference backends must choose the same moves.
  5. Visit-distribution   — every legal move must still appear, including unvisited
     key set              ones, or callers that read the policy get a short dict.

Run:  python bench/test_edge_cases.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "dual_model_mcts.pt")

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{'  — ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)


def main():
    import torch
    torch.set_num_threads(4)

    from neural_network import load_dual_model
    from mcts import MCTSPlayer, MCTSConfig, MCTSNode, MCTS

    model = load_dual_model(MODEL)

    # --- 1. Dirichlet noise (never exercised by the web app, default-on in MCTSConfig)
    print("\n1. Dirichlet noise path")
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        board.push(chess.Move.from_uci(uci))

    player = MCTSPlayer(model=model, config=MCTSConfig(
        num_simulations=60, temperature=0, add_noise=True,
        dirichlet_alpha=0.3, dirichlet_epsilon=0.25))
    move, probs = player.select_move(board.copy(), return_policy=True)
    check("noise search returns a legal move", move in board.legal_moves, str(move))
    check("noise search returns full policy",
          len(probs) == board.legal_moves.count(),
          f"{len(probs)} entries vs {board.legal_moves.count()} legal")

    # Priors must actually be perturbed and still normalised.
    node = MCTSNode(chess.Board())
    node.expand({m: 1.0 / 20 for m in chess.Board().legal_moves})
    before = dict(node.child_priors)
    node.add_dirichlet_noise(0.3, 0.25)
    changed = sum(1 for m in before if abs(before[m] - node.child_priors[m]) > 1e-9)
    check("noise perturbs priors", changed == len(before), f"{changed}/{len(before)} changed")
    check("noised priors still sum to 1",
          abs(sum(node.child_priors.values()) - 1.0) < 1e-6,
          f"sum={sum(node.child_priors.values()):.9f}")

    # --- 2. Terminal nodes reached through lazy creation
    print("\n2. Terminal-node handling")
    # Scholar's mate available: Qxf7#
    mate_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    player = MCTSPlayer(model=model, config=MCTSConfig(
        num_simulations=200, temperature=0, add_noise=False))
    move = player.select_move(mate_board.copy())
    check("search completes on a position containing mate", move is not None, str(move))

    # A position where the side to move is already stalemated must not crash callers.
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    check("stalemate detected as game over", stale.is_stalemate())
    mv, pr = MCTS(MCTSConfig(num_simulations=20)).search(stale)
    check("search on terminal board returns (None, {})", mv is None and pr == {})

    # --- 3. Uniform fallback (what a failed model load leaves behind)
    print("\n3. model=None uniform fallback")
    player = MCTSPlayer(model=None, config=MCTSConfig(
        num_simulations=80, temperature=0, add_noise=False))
    move, probs = player.select_move(board.copy(), return_policy=True)
    check("uniform-policy search returns a legal move", move in board.legal_moves, str(move))
    check("uniform-policy returns full policy", len(probs) == board.legal_moves.count())

    # --- 4. Visit-distribution key set must cover every legal move
    print("\n4. Visit distribution key set")
    wide = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9")
    n_legal = wide.legal_moves.count()
    # Deliberately fewer simulations than legal moves, so a large number of moves can
    # never be reached and therefore never get a node built. They must still appear in
    # the returned distribution with probability 0, or callers reading the policy
    # (selfplay targets, the anti-repetition fallback) silently see a truncated dict.
    sims = 20
    assert sims < n_legal
    player = MCTSPlayer(model=model, config=MCTSConfig(
        num_simulations=sims, temperature=0, add_noise=False))
    _, probs = player.select_move(wide.copy(), return_policy=True)
    check(f"all {n_legal} legal moves present at only {sims} sims",
          set(probs) == set(wide.legal_moves),
          f"{len(probs)} entries vs {n_legal} legal")
    zeros = sum(1 for p in probs.values() if p == 0.0)
    check("never-selected moves carry probability 0", zeros >= n_legal - sims,
          f"{zeros} zero-visit moves (expected >= {n_legal - sims})")

    # --- 5. ONNX and torch backends must agree on the chosen move
    print("\n5. ONNX vs torch backend agreement")
    import neural_network as nn_mod
    from bench.positions import build_suite, rebuild

    # Pin the fp32 export here. Serving PREFERS the int8 quantisation, but int8 is a
    # deliberate approximation — it agrees with fp32 on ~96% of positions by construction,
    # so demanding exact agreement with torch over six positions would fail roughly a
    # quarter of the time and mean nothing when it did. The invariant worth protecting is
    # that the EXPORT is faithful to the trained model. int8's fidelity is gated instead in
    # export_onnx.quantize_int8 (policy top-1 and KL), and was settled where it counts:
    # +175.7 Elo at 1300 sims against fp32 at 600, 120 games.
    os.environ["CHESS_ONNX_MODEL"] = "dual_model_mcts_fp32.onnx"
    suite = [e for e in build_suite() if e["ply"] in (26, 52)]
    agree = 0
    for entry in suite:
        moves = {}
        for use_onnx in ("1", "0"):
            os.environ["CHESS_USE_ONNX"] = use_onnx
            nn_mod._ort_session = None
            nn_mod._ort_resolved = False
            p = MCTSPlayer(model=model, config=MCTSConfig(
                num_simulations=200, temperature=0, add_noise=False))
            moves[use_onnx] = p.select_move(rebuild(entry))
        same = moves["1"] == moves["0"]
        agree += same
        if not same:
            print(f"      {entry['name']}: onnx={moves['1']} torch={moves['0']}")
    os.environ["CHESS_USE_ONNX"] = "1"
    os.environ.pop("CHESS_ONNX_MODEL", None)
    nn_mod._ort_session = None
    nn_mod._ort_resolved = False
    check("fp32 export and torch pick the same move", agree == len(suite),
          f"{agree}/{len(suite)} positions agree")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all edge-case checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

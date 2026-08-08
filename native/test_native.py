"""
Differential test: native movegen vs python-chess, position by position.

python-chess is the reference implementation the engine has always run on, so
stage 1 of the port is correct exactly when the native kernel agrees with it on
every position: full legal-move set, the captures+promotions subset the
quiescence search recurses on, check status, and legal-move existence
(mate/stalemate detection). Positions: the committed 1474-position suite, a
random-walk extension of each (to reach en-passant/promotion/castling states
the suite lacks), and the classic perft trap positions, cross-checked with
perft counts against python-chess.

Run:  .venv/Scripts/python.exe native/test_native.py [--quick]
"""

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import chess

import chesskernel as ck
import evaluation


def pack(board: chess.Board):
    return (board.pawns, board.knights, board.bishops, board.rooks,
            board.queens, board.kings,
            board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
            0 if board.turn == chess.WHITE else 1,
            board.ep_square if board.ep_square is not None else -1,
            board.castling_rights)


def py_moves(board):
    return {(m.from_square, m.to_square, m.promotion or 0)
            for m in board.legal_moves}


def py_qmoves(board):
    return {(m.from_square, m.to_square, m.promotion or 0)
            for m in board.legal_moves
            if board.is_capture(m) or m.promotion}


def check_position(board, failures, tag):
    args = pack(board)
    native_legal = set(map(tuple, ck.legal_moves(*args)))
    native_q = set(map(tuple, ck.qmoves(*args)))
    ref_legal = py_moves(board)
    ref_q = py_qmoves(board)
    if native_legal != ref_legal:
        failures.append((tag, board.fen(), "legal",
                         sorted(ref_legal - native_legal),
                         sorted(native_legal - ref_legal)))
    if native_q != ref_q:
        failures.append((tag, board.fen(), "qmoves",
                         sorted(ref_q - native_q), sorted(native_q - ref_q)))
    if ck.in_check(*args) != board.is_check():
        failures.append((tag, board.fen(), "in_check", board.is_check(), None))
    if ck.has_legal(*args) != any(board.generate_legal_moves()):
        failures.append((tag, board.fen(), "has_legal", None, None))
    ref_eval = evaluation._evaluate_raw(board)
    native_eval = ck.evaluate_raw(*args, board.fullmove_number)
    if native_eval != ref_eval:
        failures.append((tag, board.fen(), "evaluate_raw", ref_eval, native_eval))
    # The full quiescence search, raw centipawns. `claimable` is computed here
    # exactly as the production wrapper will compute it (draw probe stays in
    # Python for v1). Any mismatch is either a port bug or the tie-order/delta
    # interaction the C++ comment describes — both are stop-the-line findings.
    claimable = board.can_claim_draw()
    ref_q = evaluation._quiescence(board, 2, -100000, 100000, True)
    native_q = ck.qsearch(*args, board.fullmove_number, 2, claimable)
    if native_q != ref_q:
        failures.append((tag, board.fen(), "qsearch", ref_q, native_q))


# Positions that exercise each evaluate_raw term the middlegame suite cannot:
# every insufficient-material clause, checkmate-forcing with a lone king,
# back-rank traps on both sides of the fullmove gate, mate and stalemate
# terminals, and phase-boundary material levels.
EVAL_POSITIONS = [
    ("kk",            "8/8/8/4k3/8/8/4K3/8 w - - 0 40"),
    ("kn-k",          "8/8/8/4k3/8/3N4/4K3/8 w - - 0 40"),
    ("kb-k",          "8/8/8/4k3/8/3B4/4K3/8 b - - 0 40"),
    ("knn-k",         "8/8/8/4k3/8/1NN5/4K3/8 w - - 0 40"),
    # ILLEGAL on purpose (Nd3 attacks the king with White to move): python-chess
    # tolerates it and allows capturing the king, which reaches a KINGLESS
    # board. The first sweep of this file found exactly that as UB in the
    # kernel; it stays as a permanent robustness case.
    ("knn-k-illegal", "8/8/8/4k3/8/2NN4/4K3/8 w - - 0 40"),
    ("kn-kn",         "8/8/8/3nk3/8/3N4/4K3/8 w - - 0 40"),
    ("kb-kb-same",    "8/8/8/3bk3/8/3B4/4K3/8 w - - 0 40"),
    ("kb-kb-opp",     "8/8/8/3bk3/8/4B3/4K3/8 w - - 0 40"),
    ("kn-kr",         "8/8/8/3rk3/8/3N4/4K3/8 w - - 0 40"),
    ("kq-k-forcing",  "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 40"),
    ("kr-k-forcing",  "8/8/4k3/8/8/8/4K3/7R b - - 0 40"),
    ("kq-k-corner",   "k7/8/1K6/8/8/8/8/7Q b - - 0 60"),
    ("backrank-trap", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 20"),
    ("backrank-early","6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 5"),
    ("backrank-mate", "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 30"),
    ("stalemate",     "7k/5Q2/6K1/8/8/8/8/8 b - - 0 40"),
    ("phase-mid",     "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 6"),
    ("promo-race",    "8/P6k/8/8/8/8/p6K/8 w - - 0 50"),
]

# Classic perft positions — the standard traps: castling through attack,
# en-passant pins, promotions, discovered checks.
PERFT_POSITIONS = [
    ("startpos", chess.STARTING_FEN, 3),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
    ("pos3-ep", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
    ("pos4-promo", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
    ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
]


def py_perft(board, depth):
    if depth == 0:
        return 1
    n = 0
    for m in board.legal_moves:
        board.push(m)
        n += py_perft(board, depth - 1)
        board.pop()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="suite positions only, no random walks or perft")
    ap.add_argument("--walks", type=int, default=6,
                    help="random moves walked from each suite position")
    args = ap.parse_args()

    rng = random.Random(1474)   # deterministic: same walk every run
    failures = []
    checked = 0

    with open(os.path.join(_HERE, "..", "bench", "suites", "acpl.json")) as f:
        suite = json.load(f)["positions"]

    t0 = time.perf_counter()
    for e in suite:
        board = chess.Board(e["fen"])
        check_position(board, failures, e["name"])
        checked += 1
        if not args.quick:
            walk = chess.Board(e["fen"])
            for _ in range(args.walks):
                moves = list(walk.legal_moves)
                if not moves:
                    break
                walk.push(rng.choice(moves))
                check_position(walk, failures, e["name"] + "-walk")
                checked += 1

    for name, fen in EVAL_POSITIONS:
        board = chess.Board(fen)
        check_position(board, failures, name)
        checked += 1

    if not args.quick:
        for name, fen, depth in PERFT_POSITIONS:
            board = chess.Board(fen)
            check_position(board, failures, name)
            checked += 1
            native = ck.perft(*pack(board), depth)
            ref = py_perft(board, depth)
            status = "ok" if native == ref else "MISMATCH"
            if native != ref:
                failures.append((name, fen, f"perft({depth})", ref, native))
            print(f"  perft {name:<10} depth {depth}: native {native}, "
                  f"python-chess {ref}  [{status}]")

    dt = time.perf_counter() - t0
    print(f"\n{checked} positions checked in {dt:.1f}s")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f_ in failures[:20]:
            print("  ", f_)
        return 1
    print("native movegen AND evaluate_raw agree with the Python reference "
          "on every position")
    return 0


if __name__ == "__main__":
    sys.exit(main())

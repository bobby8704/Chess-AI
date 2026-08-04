"""
Reproducible position suite for latency and move-equivalence benchmarking.

Positions are built by PUSHING moves onto a single board, never via chess.Board(fen).
This matters: a FEN-constructed board has an empty move_stack, and chess.Board.copy()
clones the whole stack, so a FEN benchmark hides a large share of the real per-move cost
and makes the engine look much faster than it is in an actual game.

Positions are filtered to keep >= MIN_PIECES pieces so the <=7-piece tablebase probe
(a blocking HTTPS call) never fires and pollutes timings.
"""

import random

import chess

MIN_PIECES = 12      # tablebase probes at <= 7; stay well clear
MIN_LEGAL = 24       # realistic middlegame branching
MAX_LEGAL = 48

DEFAULT_SEED = 20260805
DEFAULT_PLIES = (0, 12, 26, 40, 52)


def _piece_count(board: chess.Board) -> int:
    return chess.popcount(board.occupied)


def _playable(board: chess.Board) -> bool:
    if board.is_game_over():
        return False
    if _piece_count(board) < MIN_PIECES:
        return False
    return MIN_LEGAL <= board.legal_moves.count() <= MAX_LEGAL


def _play_to(target_ply: int, rng: random.Random):
    """Play pseudo-random but capture-aware legal moves up to target_ply."""
    board = chess.Board()
    while len(board.move_stack) < target_ply:
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            return None
        captures = [m for m in moves if board.is_capture(m)]
        if captures and rng.random() < 0.35:
            board.push(rng.choice(captures))
        else:
            board.push(rng.choice(moves))
    return board if _playable(board) else None


def build_suite(seed: int = DEFAULT_SEED, plies=DEFAULT_PLIES, per_ply: int = 3):
    """
    Return a list of position dicts:
        {name, ply, moves (list of uci), fen, n_legal, n_pieces}

    `moves` is the authoritative field — rebuild with rebuild() so the move_stack
    is present. `fen` is recorded for human inspection only.
    """
    suite = []
    for ply in plies:
        # Ply 0 is the start position: only one exists, and its 20 legal moves sit
        # below MIN_LEGAL. Keep it anyway — it is the reference point for showing
        # that move time grows with game length.
        if ply == 0:
            board = chess.Board()
            suite.append({
                "name": "ply00_0",
                "ply": 0,
                "moves": [],
                "fen": board.fen(),
                "n_legal": board.legal_moves.count(),
                "n_pieces": _piece_count(board),
            })
            continue

        found = 0
        attempt = 0
        while found < per_ply:
            attempt += 1
            if attempt > 4000:
                raise RuntimeError(f"could not build {per_ply} positions at ply {ply}")
            board = _play_to(ply, random.Random(seed + ply * 10_000 + attempt))
            if board is None:
                continue
            suite.append({
                "name": f"ply{ply:02d}_{found}",
                "ply": ply,
                "moves": [m.uci() for m in board.move_stack],
                "fen": board.fen(),
                "n_legal": board.legal_moves.count(),
                "n_pieces": _piece_count(board),
            })
            found += 1
    return suite


def rebuild(entry) -> chess.Board:
    """Rebuild a board WITH its move stack from a suite entry."""
    board = chess.Board()
    for uci in entry["moves"]:
        board.push(chess.Move.from_uci(uci))
    return board


if __name__ == "__main__":
    for e in build_suite():
        print(f"{e['name']:>12}  ply={e['ply']:>2}  legal={e['n_legal']:>2}  "
              f"pieces={e['n_pieces']:>2}  {e['fen']}")

"""
Endgame Tablebase Probe

Queries the Lichess Syzygy tablebase API for perfect endgame play
in positions with ≤7 pieces. Returns the optimal move guaranteed
to achieve the best possible result (win/draw).

API: https://tablebase.lichess.ovh/standard?fen=...

Falls back gracefully when offline or API unavailable.
"""

import chess
import json
import urllib.request
import urllib.error
from typing import Optional, Tuple

# Maximum pieces for tablebase lookup
MAX_TABLEBASE_PIECES = 7

# API endpoint
_API_URL = "https://tablebase.lichess.ovh/standard"

# Simple cache to avoid repeated API calls for the same position
_cache: dict[str, Optional[dict]] = {}
_MAX_CACHE = 500


def piece_count(board: chess.Board) -> int:
    """Count total pieces on the board (including kings)."""
    return len(board.piece_map())


def should_probe(board: chess.Board) -> bool:
    """Check if this position is eligible for tablebase lookup."""
    if board.is_game_over():
        return False
    return piece_count(board) <= MAX_TABLEBASE_PIECES


def probe(board: chess.Board) -> Optional[chess.Move]:
    """
    Query the tablebase for the optimal move.

    Returns the best move for perfect play, or None if:
    - Position has too many pieces (>7)
    - API is unavailable
    - Position is not in the tablebase

    The returned move is guaranteed optimal (shortest path to
    win, or best drawing move if the position is drawn).
    """
    if not should_probe(board):
        return None

    fen = board.fen()

    # Check cache
    if fen in _cache:
        cached = _cache[fen]
        if cached is None:
            return None
        return _parse_best_move(cached, board)

    # Query API
    try:
        encoded_fen = urllib.parse.quote(fen)
        url = f"{_API_URL}?fen={encoded_fen}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChessAI/1.0"})
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read().decode())

        # Cache the result
        if len(_cache) >= _MAX_CACHE:
            # Evict oldest entries
            keys = list(_cache.keys())
            for k in keys[:_MAX_CACHE // 2]:
                del _cache[k]
        _cache[fen] = data

        return _parse_best_move(data, board)

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception):
        # API unavailable — cache the failure briefly and return None
        _cache[fen] = None
        return None


def probe_with_info(board: chess.Board) -> Tuple[Optional[chess.Move], Optional[str], Optional[int]]:
    """
    Query the tablebase and return move + result info.

    Returns:
        (best_move, category, dtm)
        - best_move: optimal move or None
        - category: "win", "loss", "draw", or None
        - dtm: distance to mate (half-moves) or None
    """
    if not should_probe(board):
        return None, None, None

    fen = board.fen()

    try:
        encoded_fen = urllib.parse.quote(fen)
        url = f"{_API_URL}?fen={encoded_fen}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChessAI/1.0"})
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read().decode())

        if len(_cache) >= _MAX_CACHE:
            keys = list(_cache.keys())
            for k in keys[:_MAX_CACHE // 2]:
                del _cache[k]
        _cache[fen] = data

        move = _parse_best_move(data, board)
        category = data.get("category")
        dtm = data.get("dtm")

        return move, category, dtm

    except Exception:
        return None, None, None


def _parse_best_move(data: dict, board: chess.Board) -> Optional[chess.Move]:
    """Extract the best move from the API response."""
    moves = data.get("moves", [])
    if not moves:
        return None

    # Moves are sorted best-first by the API
    best = moves[0]
    uci = best.get("uci")
    if uci is None:
        return None

    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    return None

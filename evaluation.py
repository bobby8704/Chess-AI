"""
Classical Chess Position Evaluation

A hand-coded evaluation function based on established chess principles.
Used as the value component in MCTS (replacing the weak NN value head).

Returns evaluation from the CURRENT PLAYER's perspective:
  Positive = good for side to move
  Negative = bad for side to move
  Range: approximately [-1, +1] (clamped via tanh)

Components:
  1. Material balance (piece values)
  2. Piece-square tables (positional bonuses)
  3. Pawn structure (doubled, isolated, passed pawns)
  4. King safety (pawn shelter, open files near king)
  5. Mobility (number of legal-like moves)
  6. Bishop pair bonus
  7. Rook on open/semi-open file
  8. Game phase detection (opening/middlegame/endgame)
"""

import chess
import math

# ============================================================
# Piece values (centipawns)
# ============================================================

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# ============================================================
# Piece-square tables (from White's perspective, index 0 = a1)
# Values in centipawns. Flipped for Black.
# ============================================================

# fmt: off
PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING_MG = [  # Middlegame: stay castled
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PST_KING_EG = [  # Endgame: centralize
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
]
# fmt: on

PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
}


def _pst_value(piece_type: int, square: int, color: bool, endgame_weight: float) -> int:
    """Get piece-square table value for a piece."""
    if piece_type == chess.KING:
        mg = PST_KING_MG[square if color == chess.WHITE else chess.square_mirror(square)]
        eg = PST_KING_EG[square if color == chess.WHITE else chess.square_mirror(square)]
        return int(mg * (1 - endgame_weight) + eg * endgame_weight)

    table = PST.get(piece_type)
    if table is None:
        return 0

    # Tables are from White's perspective (rank 8 at top = index 0)
    # For White: use square directly (a1=0 maps to bottom-left)
    # For Black: mirror vertically
    idx = square if color == chess.WHITE else chess.square_mirror(square)
    return table[idx]


# ============================================================
# Game phase detection
# ============================================================

def _game_phase(board: chess.Board) -> float:
    """Return endgame weight: 0.0 = opening/middlegame, 1.0 = pure endgame."""
    # Total non-pawn, non-king material
    npm = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type not in (chess.PAWN, chess.KING):
            npm += PIECE_VALUE[p.piece_type]

    # Full material (no pawns/kings) ~= 2*(320+330+500+900) = 4100
    # Endgame starts around npm < 1500
    if npm >= 4000:
        return 0.0
    if npm <= 1000:
        return 1.0
    return 1.0 - (npm - 1000) / 3000.0


# ============================================================
# Pawn structure
# ============================================================

def _pawn_structure(board: chess.Board, color: bool) -> int:
    """Evaluate pawn structure: doubled, isolated, passed pawns. Returns centipawns."""
    score = 0
    our_pawns = board.pieces(chess.PAWN, color)
    opp_pawns = board.pieces(chess.PAWN, not color)

    files_with_pawns = set()
    for sq in our_pawns:
        f = chess.square_file(sq)
        files_with_pawns.add(f)

    for sq in our_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        # Doubled pawns (another pawn on same file)
        same_file = [s for s in our_pawns if chess.square_file(s) == f and s != sq]
        if same_file:
            score -= 15

        # Isolated pawns (no friendly pawns on adjacent files)
        has_neighbor = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7 and adj_f in files_with_pawns:
                has_neighbor = True
                break
        if not has_neighbor:
            score -= 20

        # Passed pawns (no opposing pawns ahead on same or adjacent files)
        is_passed = True
        for check_f in range(max(0, f - 1), min(7, f + 1) + 1):
            if color == chess.WHITE:
                check_ranks = range(r + 1, 8)
            else:
                check_ranks = range(0, r)
            for check_r in check_ranks:
                check_sq = chess.square(check_f, check_r)
                if check_sq in opp_pawns:
                    is_passed = False
                    break
            if not is_passed:
                break

        if is_passed:
            # Bonus increases as pawn advances
            if color == chess.WHITE:
                advance = r - 1  # 0 at rank 2, 5 at rank 7
            else:
                advance = 6 - r  # 0 at rank 7, 5 at rank 2
            score += 20 + advance * 15

    return score


# ============================================================
# King safety
# ============================================================

def _king_safety(board: chess.Board, color: bool) -> int:
    """Evaluate king safety. Returns centipawns."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    score = 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    # Pawn shelter (pawns in front of king)
    shelter_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1
    if 0 <= shelter_rank <= 7:
        for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            sq = chess.square(f, shelter_rank)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                score += 15  # Pawn sheltering king
            else:
                score -= 15  # Missing shelter pawn

    # Penalize king on open file (no pawns of either color)
    has_pawn_on_file = False
    for r in range(8):
        sq = chess.square(king_file, r)
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            has_pawn_on_file = True
            break
    if not has_pawn_on_file:
        score -= 30

    return score


# ============================================================
# Mobility
# ============================================================

def _mobility(board: chess.Board, color: bool) -> int:
    """Estimate piece mobility. Returns centipawns."""
    # Count squares attacked by each piece type
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type not in (chess.PAWN, chess.KING):
            attacks = len(board.attacks(sq))
            # Small bonus per attacked square
            score += attacks * 2
    return score


# ============================================================
# Main evaluation function
# ============================================================

def evaluate(board: chess.Board) -> float:
    """
    Evaluate a chess position from the CURRENT PLAYER's perspective.

    Returns a float in approximately [-1, +1]:
      Positive = good for side to move
      Negative = bad for side to move

    Uses tanh normalization so extreme advantages compress toward +/-1.
    """
    if board.is_checkmate():
        return -1.0  # Current player is checkmated

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    if board.can_claim_draw():
        return 0.0

    score = 0  # Centipawns from White's perspective
    eg_weight = _game_phase(board)

    # 1. Material + piece-square tables
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        val = PIECE_VALUE[piece.piece_type]
        val += _pst_value(piece.piece_type, sq, piece.color, eg_weight)

        if piece.color == chess.WHITE:
            score += val
        else:
            score -= val

    # 2. Pawn structure
    score += _pawn_structure(board, chess.WHITE)
    score -= _pawn_structure(board, chess.BLACK)

    # 3. King safety (less important in endgame)
    mg_weight = 1.0 - eg_weight
    score += int(_king_safety(board, chess.WHITE) * mg_weight)
    score -= int(_king_safety(board, chess.BLACK) * mg_weight)

    # 4. Mobility
    score += _mobility(board, chess.WHITE)
    score -= _mobility(board, chess.BLACK)

    # 5. Bishop pair bonus
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    if white_bishops >= 2:
        score += 30
    if black_bishops >= 2:
        score -= 30

    # 6. Rook on open file
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own_pawns = any(
                board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color)
                for r in range(8)
            )
            opp_pawns = any(
                board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, not color)
                for r in range(8)
            )
            if not own_pawns and not opp_pawns:
                score += sign * 25  # Open file
            elif not own_pawns:
                score += sign * 15  # Semi-open file

    # Convert to current player's perspective and normalize
    if board.turn == chess.BLACK:
        score = -score

    # Normalize: 100cp = ~0.25, 400cp = ~0.76, 900cp = ~0.97
    return math.tanh(score / 400.0)

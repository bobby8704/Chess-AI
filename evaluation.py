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
# Knight outposts
# ============================================================

def _knight_outposts(board: chess.Board, color: bool) -> int:
    """Bonus for knights on outpost squares (protected by pawn, can't be attacked by enemy pawn)."""
    score = 0
    opponent = not color
    for sq in board.pieces(chess.KNIGHT, color):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        # Outpost = on rank 4-6 (for white), protected by own pawn, no enemy pawn can attack
        if color == chess.WHITE:
            if rank < 3 or rank > 5:
                continue
        else:
            if rank < 2 or rank > 4:
                continue

        # Check if protected by own pawn
        pawn_protectors = board.attackers(color, sq) & board.pieces(chess.PAWN, color)
        if not pawn_protectors:
            continue

        # Check no enemy pawn can attack this square (on adjacent files, ahead)
        can_be_kicked = False
        for adj_f in [file - 1, file + 1]:
            if adj_f < 0 or adj_f > 7:
                continue
            if color == chess.WHITE:
                check_ranks = range(rank + 1, 8)
            else:
                check_ranks = range(0, rank)
            for r in check_ranks:
                p = board.piece_at(chess.square(adj_f, r))
                if p and p.piece_type == chess.PAWN and p.color == opponent:
                    can_be_kicked = True
                    break
            if can_be_kicked:
                break

        if not can_be_kicked:
            score += 30  # Strong outpost

    return score


# ============================================================
# Connected rooks
# ============================================================

def _connected_rooks(board: chess.Board, color: bool) -> int:
    """Bonus for rooks that defend each other (on same rank/file with nothing between)."""
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return 0

    r1, r2 = rooks[0], rooks[1]
    f1, rk1 = chess.square_file(r1), chess.square_rank(r1)
    f2, rk2 = chess.square_file(r2), chess.square_rank(r2)

    # Same rank or same file?
    if f1 == f2 or rk1 == rk2:
        # Check if they can see each other (nothing between)
        if board.is_attacked_by(color, r1) and r2 in board.attacks(r1):
            return 15

    return 0


# ============================================================
# King tropism (opponent pieces near our king = danger)
# ============================================================

def _king_tropism(board: chess.Board, color: bool) -> int:
    """Penalize when opponent pieces are close to our king."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    opponent = not color
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    score = 0

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != opponent:
            continue
        if p.piece_type in (chess.PAWN, chess.KING):
            continue

        # Chebyshev distance from opponent piece to our king
        dist = max(abs(chess.square_file(sq) - king_file),
                   abs(chess.square_rank(sq) - king_rank))

        # Closer = more dangerous (scaled by piece value)
        if dist <= 3:
            threat = PIECE_VALUE.get(p.piece_type, 0) // 100
            score -= (4 - dist) * threat * 3

    return score


# ============================================================
# Checkmate forcing (endgame with major pieces vs lone king)
# ============================================================

# Distance from center: corners = 3, edges = 2, center = 0
_CENTER_DISTANCE = [
    3, 2, 2, 2, 2, 2, 2, 3,
    2, 1, 1, 1, 1, 1, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 0, 0, 0, 0, 1, 2,
    2, 1, 1, 1, 1, 1, 1, 2,
    3, 2, 2, 2, 2, 2, 2, 3,
]


def _king_distance(sq1: int, sq2: int) -> int:
    """Chebyshev distance between two squares."""
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
    return max(abs(f1 - f2), abs(r1 - r2))


def _checkmate_forcing(board: chess.Board, strong_side: bool) -> int:
    """
    When strong_side has major pieces and weak_side has only a king,
    return bonuses for driving the lone king to the edge and keeping
    the strong king close. Returns centipawns for strong_side.
    """
    weak_side = not strong_side

    # Check if weak side has only a king
    weak_pieces = board.pieces(chess.PAWN, weak_side) | \
                  board.pieces(chess.KNIGHT, weak_side) | \
                  board.pieces(chess.BISHOP, weak_side) | \
                  board.pieces(chess.ROOK, weak_side) | \
                  board.pieces(chess.QUEEN, weak_side)
    if len(weak_pieces) > 0:
        return 0  # Opponent still has pieces, not a lone king situation

    # Check that strong side has enough mating material
    strong_queens = len(board.pieces(chess.QUEEN, strong_side))
    strong_rooks = len(board.pieces(chess.ROOK, strong_side))
    if strong_queens == 0 and strong_rooks == 0:
        return 0  # Need at least a queen or rook to force mate

    score = 0
    weak_king = board.king(weak_side)
    strong_king = board.king(strong_side)
    if weak_king is None or strong_king is None:
        return 0

    # 1. Reward pushing opponent king to edge/corner (most important)
    # Center distance: 0 (center) to 3 (corner)
    score += _CENTER_DISTANCE[weak_king] * 150

    # 2. Reward keeping strong king close to weak king (assists mate)
    king_dist = _king_distance(weak_king, strong_king)
    score += (7 - king_dist) * 80

    # 3. Bonus for restricting opponent king's mobility
    weak_king_moves = 0
    for sq in chess.SQUARES:
        if _king_distance(weak_king, sq) == 1:
            if not board.is_attacked_by(strong_side, sq):
                p = board.piece_at(sq)
                if p is None or p.color == strong_side:
                    weak_king_moves += 1
    # Fewer escape squares = better (max 8, min 0)
    score += (8 - weak_king_moves) * 50

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

    # 7. Knight outposts
    score += _knight_outposts(board, chess.WHITE)
    score -= _knight_outposts(board, chess.BLACK)

    # 8. Connected rooks
    score += _connected_rooks(board, chess.WHITE)
    score -= _connected_rooks(board, chess.BLACK)

    # 9. King tropism (opponent pieces near king = danger, middlegame only)
    score += int(_king_tropism(board, chess.WHITE) * mg_weight)
    score -= int(_king_tropism(board, chess.BLACK) * mg_weight)

    # 10. Checkmate forcing: when one side has overwhelming material vs lone king
    score += _checkmate_forcing(board, chess.WHITE)
    score -= _checkmate_forcing(board, chess.BLACK)

    # 8. Back-rank safety: penalize king trapped on back rank with no escape
    score += _back_rank_safety(board, chess.WHITE)
    score -= _back_rank_safety(board, chess.BLACK)

    # Convert to current player's perspective and normalize
    if board.turn == chess.BLACK:
        score = -score

    # Normalize: 100cp = ~0.25, 400cp = ~0.76, 900cp = ~0.97
    return math.tanh(score / 400.0)


# ============================================================
# Back-rank safety
# ============================================================

def _back_rank_safety(board: chess.Board, color: bool) -> int:
    """Penalize king trapped on back rank with no escape (back-rank mate risk)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    back_rank = 0 if color == chess.WHITE else 7
    king_rank = chess.square_rank(king_sq)

    if king_rank != back_rank:
        return 0  # King not on back rank, no penalty

    # Only relevant after castling (middlegame+), not in opening setup
    if board.fullmove_number < 8:
        return 0

    king_file = chess.square_file(king_sq)

    # Check if king has an escape square (one rank forward)
    escape_rank = 1 if color == chess.WHITE else 6
    has_escape = False
    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        sq = chess.square(f, escape_rank)
        piece = board.piece_at(sq)
        # Square is an escape if it's empty or has an opponent piece (king can capture)
        # and isn't attacked by opponent
        if piece is None or piece.color != color:
            if not board.is_attacked_by(not color, sq):
                has_escape = True
                break

    if has_escape:
        return 0

    # King is trapped on back rank — check if opponent has a rook/queen
    opponent = not color
    opp_rooks = len(board.pieces(chess.ROOK, opponent))
    opp_queens = len(board.pieces(chess.QUEEN, opponent))

    if opp_rooks + opp_queens == 0:
        return 0  # No back-rank mate threat without rooks/queens

    # Significant penalty — back rank is very dangerous
    return -80 * (opp_rooks + opp_queens)


# ============================================================
# Quiescence search
# ============================================================

def evaluate_quiescence(board: chess.Board, max_depth: int = 2) -> float:
    """
    Evaluate a position with quiescence search.

    Extends the evaluation by playing out captures until the position
    is "quiet" (no more profitable captures). Max depth 3 plies to
    prevent explosion in complex middlegame positions.

    Returns value from current player's perspective in [-1, +1].
    """
    return math.tanh(_quiescence(board, max_depth, -100000, 100000) / 400.0)


def _quiescence(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """
    Quiescence search in centipawns from current player's perspective.
    Only searches capture moves to resolve tactical instability.
    Uses alpha-beta pruning for efficiency.
    """
    if board.is_checkmate():
        return -30000

    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0

    # Standing pat: evaluate the position statically
    stand_pat = _evaluate_raw(board)

    if depth <= 0:
        return stand_pat

    if stand_pat >= beta:
        return beta  # Beta cutoff

    if stand_pat > alpha:
        alpha = stand_pat

    # Delta pruning margin (skip captures that can't possibly improve alpha)
    DELTA_MARGIN = 200  # ~2 pawns

    # Search only captures (and promotions), ordered by MVV-LVA
    captures = []
    for i, move in enumerate(board.legal_moves):
        if board.is_capture(move) or move.promotion:
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            victim = board.piece_at(move.to_square)
            victim_val = PIECE_VALUE.get(victim.piece_type, 0) if victim else 0

            # Delta pruning: if capturing this piece can't beat alpha, skip
            if stand_pat + victim_val + DELTA_MARGIN < alpha and not move.promotion:
                continue

            attacker = board.piece_at(move.from_square)
            attacker_val = PIECE_VALUE.get(attacker.piece_type, 0) if attacker else 0
            # Also give bonus for promotions
            promo_val = 800 if move.promotion == chess.QUEEN else 0
            captures.append((-(victim_val - attacker_val + promo_val), i, move))

    captures.sort(key=lambda x: x[0])  # Best captures first

    for _, _, move in captures:
        board.push(move)
        score = -_quiescence(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta

        if score > alpha:
            alpha = score

    return alpha


def _evaluate_raw(board: chess.Board) -> int:
    """
    Raw evaluation in centipawns from current player's perspective.
    Used internally by quiescence search (no tanh normalization).
    """
    if board.is_checkmate():
        return -30000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    eg_weight = _game_phase(board)

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

    score += _pawn_structure(board, chess.WHITE)
    score -= _pawn_structure(board, chess.BLACK)

    mg_weight = 1.0 - eg_weight
    score += int(_king_safety(board, chess.WHITE) * mg_weight)
    score -= int(_king_safety(board, chess.BLACK) * mg_weight)

    score += _back_rank_safety(board, chess.WHITE)
    score -= _back_rank_safety(board, chess.BLACK)

    score += _checkmate_forcing(board, chess.WHITE)
    score -= _checkmate_forcing(board, chess.BLACK)

    # Bishop pair (cheap check)
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 30

    # Convert to current player's perspective
    if board.turn == chess.BLACK:
        score = -score

    return score

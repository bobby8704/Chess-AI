
import numpy as np
import chess

PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Return a flat feature vector for a board.
    12x8x8 one-hot planes (white then black) + aux features:
    side-to-move, castling rights (4), en-passant file (8).
    Total dims: 12*64 + 1 + 4 + 8 = 781
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            # 0-5 white pieces, 6-11 black pieces
            idx = PIECE_MAP[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            r = 7 - chess.square_rank(sq)  # rank 8 at top row
            f = chess.square_file(sq)
            planes[idx, r, f] = 1.0

    aux = np.zeros(1 + 4 + 8, dtype=np.float32)
    # side to move
    aux[0] = 1.0 if board.turn == chess.WHITE else 0.0
    # castling rights
    aux[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    # en passant file (one-hot)
    if board.ep_square is not None:
        aux[5 + chess.square_file(board.ep_square)] = 1.0

    return np.concatenate([planes.reshape(-1), aux], axis=0)

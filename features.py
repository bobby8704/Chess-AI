
import numpy as np
import chess

PIECE_MAP = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

# Number of input channels for CNN: 12 piece planes + 1 aux plane
CNN_INPUT_CHANNELS = 13


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Return a flat feature vector for a board (781 dims, for MLP).
    12x8x8 one-hot planes (white then black) + aux features:
    side-to-move, castling rights (4), en-passant file (8).
    Total dims: 12*64 + 1 + 4 + 8 = 781
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = PIECE_MAP[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            r = 7 - chess.square_rank(sq)
            f = chess.square_file(sq)
            planes[idx, r, f] = 1.0

    aux = np.zeros(1 + 4 + 8, dtype=np.float32)
    aux[0] = 1.0 if board.turn == chess.WHITE else 0.0
    aux[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    aux[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    aux[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    aux[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        aux[5 + chess.square_file(board.ep_square)] = 1.0

    return np.concatenate([planes.reshape(-1), aux], axis=0)


def board_to_tensor_2d(board: chess.Board) -> np.ndarray:
    """Return a (13, 8, 8) tensor for CNN input.

    Channels 0-11: piece planes (6 white + 6 black)
    Channel 12:    auxiliary features encoded as an 8x8 plane:
        - Row 0: side-to-move (all 1s if White, all 0s if Black)
        - Row 1, cols 0-3: castling rights (WK, WQ, BK, BQ)
        - Row 2, cols 0-7: en passant file (one-hot)
        - Remaining rows: zeros
    """
    planes = np.zeros((13, 8, 8), dtype=np.float32)

    # Channels 0-11: piece positions
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = PIECE_MAP[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            r = 7 - chess.square_rank(sq)
            f = chess.square_file(sq)
            planes[idx, r, f] = 1.0

    # Channel 12: auxiliary features
    if board.turn == chess.WHITE:
        planes[12, 0, :] = 1.0  # Side to move
    planes[12, 1, 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[12, 1, 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[12, 1, 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[12, 1, 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        planes[12, 2, chess.square_file(board.ep_square)] = 1.0

    return planes

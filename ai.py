# ai.py
import chess
import random

# piece‐square tables and values you defined in the notebook
piece_values = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
}
# …and any piece_values_with_position tables you used…

def evaluate_board(board: chess.Board) -> int:
    """Simple material evaluation."""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            sign = 1 if piece.color == chess.WHITE else -1
            score += sign * piece_values[piece.piece_type]
    return score

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, white_to_play: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    if white_to_play:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth-1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval, best_move = eval_score, move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth-1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval, best_move = eval_score, move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def best_move_using_minimax(board: chess.Board, depth: int) -> chess.Move:
    _, move = minimax(board, depth, -float('inf'), float('inf'), board.turn)
    return move or random.choice(list(board.legal_moves))

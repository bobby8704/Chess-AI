
import torch
import numpy as np
import chess
from features import board_to_tensor

class ValueNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_input_dim = None

def load_value_model(path: str = "value_model.pt"):
    """Load a trained value network from disk."""
    global _model, _input_dim
    ckpt = torch.load(path, map_location=_device)
    _input_dim = ckpt["input_dim"]
    _model = ValueNet(_input_dim).to(_device)
    _model.load_state_dict(ckpt["state_dict"])
    _model.eval()

def evaluate_board_nn(board: chess.Board) -> float:
    """Return value in [-1,1] from White's perspective.
    If no model is loaded, return 0.0 (neutral).
    """
    global _model
    if _model is None:
        return 0.0
    x = board_to_tensor(board)
    x = torch.from_numpy(x).float().to(_device).unsqueeze(0)
    with torch.no_grad():
        v = _model(x).item()
    # v is from white perspective; if black to move, flip when maximizing/minimizing
    return v

def _minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool):
    if depth == 0 or board.is_game_over():
        # When it's black to move, the side-to-move is minimizing wrt White's value
        val = evaluate_board_nn(board)
        return (val if maximizing else -val), None

    best_move = None
    if maximizing:
        max_eval = -1e9
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = _minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval, best_move = eval_score, move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = 1e9
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = _minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval, best_move = eval_score, move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def best_move_nn(board: chess.Board, depth: int = 2) -> chess.Move | None:
    if board.is_game_over():
        return None
    maximizing = (board.turn == chess.WHITE)
    _, move = _minimax(board, depth, -1e9, 1e9, maximizing)
    if move is None:
        # safe fallback: return None if no legal moves exist
        for m in board.legal_moves:
            return m
        return None
    return move

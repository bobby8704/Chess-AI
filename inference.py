"""
Torch-free inference for the chess engine.

Everything the web server needs to turn a board into a policy lives here: move
encoding, legal-move indices, and the onnxruntime session. Nothing in this module
imports torch, which is the point — it lets a deployment install onnxruntime (~42MB)
instead of torch (~433MB) and still serve real moves.

neural_network.py re-exports these names, so existing callers keep working. Training,
model definitions, and checkpoint loading stay there, where torch is required.

Thread count is controlled by CHESS_NUM_THREADS (default 4); ONNX inference is
controlled by CHESS_USE_ONNX (set to 0 to force the torch path).
"""

import os
from typing import Optional

import numpy as np
import chess

# Total possible moves: 64*64 = 4096 base moves, plus 64*3 underpromotions = 4288.
POLICY_OUTPUT_SIZE = 4096 + 192


def encode_move(move: chess.Move) -> int:
    """
    Encode a chess move as an integer index.

    Encoding scheme:
    - Base moves: from_square * 64 + to_square (0-4095)
    - Underpromotions: 4096 + from_square * 3 + (promotion_type - 2)
      where promotion_type: 2=knight, 3=bishop, 4=rook

    Queen promotions use the base encoding since they're the default.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotion: knight=2, bishop=3, rook=4
        promo_idx = move.promotion - 2  # 0, 1, or 2
        return 4096 + from_sq * 3 + promo_idx
    else:
        # Regular move or queen promotion
        return from_sq * 64 + to_sq


def decode_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Decode an integer index back to a chess move.

    Returns None if the decoded move is not legal in the given position.
    """
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64

        # Check if this is a pawn promotion
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            # Check if moving to promotion rank
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                # Default to queen promotion
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                if move in board.legal_moves:
                    return move

        move = chess.Move(from_sq, to_sq)
        if move in board.legal_moves:
            return move
    else:
        # Underpromotion
        idx = index - 4096
        from_sq = idx // 3
        promo_idx = idx % 3
        promotion = promo_idx + 2  # knight=2, bishop=3, rook=4

        # Determine to_square based on pawn position
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                to_sq = from_sq + 8
            else:
                to_sq = from_sq - 8

            move = chess.Move(from_sq, to_sq, promotion=promotion)
            if move in board.legal_moves:
                return move

    return None


def legal_move_indices(board: chess.Board):
    """
    Return (moves, indices) for the current position, in legal-move order.

    Computing the policy indices once lets a caller build the legal-move mask AND
    gather the resulting probabilities without walking the move list — and calling
    encode_move on every move — a second time.
    """
    moves = []
    idxs = []
    for move in board.legal_moves:
        idx = encode_move(move)
        if idx < POLICY_OUTPUT_SIZE:
            moves.append(move)
            idxs.append(idx)
    return moves, np.asarray(idxs, dtype=np.int64)


# ==================== ONNX Runtime session ====================

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Served models, in preference order. The int8 export is a static QDQ quantisation of the
# fp32 one: ~4x faster at batch size 1, 3.9x smaller (4.8MB vs 18.4MB), and measured
# strength-neutral at equal simulations. fp32 stays as the fallback AND as the thing int8
# is quantised from, so both are tracked.
ONNX_PREFERENCE = ("dual_model_mcts_int8.onnx", "dual_model_mcts_fp32.onnx")

# export_onnx.py writes this one, so it is what error messages should name.
DEFAULT_ONNX_PATH = os.path.join(_MODELS_DIR, ONNX_PREFERENCE[-1])
DEFAULT_CKPT_PATH = os.path.join(_MODELS_DIR, "dual_model_mcts.pt")

_ort_session = None
_ort_resolved = False


def onnx_candidates():
    """Absolute model paths in preference order, honouring CHESS_ONNX_MODEL."""
    override = os.environ.get("CHESS_ONNX_MODEL")
    if override:
        return [override if os.path.isabs(override)
                else os.path.join(_MODELS_DIR, override)]
    return [os.path.join(_MODELS_DIR, name) for name in ONNX_PREFERENCE]


def resolve_onnx_path():
    """
    First existing candidate, skipping a quantisation that has gone stale.

    int8 is DERIVED from the fp32 export, which is derived from the checkpoint. So a
    re-export that is not followed by a re-quantise leaves an int8 file built from the
    previous weights — and serving it would look perfectly healthy while quietly playing
    the old model. That is the same silent-staleness failure get_onnx_session already
    guards against for checkpoint-vs-export, one level further down the chain.
    """
    fp32 = os.path.join(_MODELS_DIR, "dual_model_mcts_fp32.onnx")
    for path in onnx_candidates():
        if not os.path.exists(path):
            continue
        if (path.endswith("_int8.onnx") and os.path.exists(fp32)
                and os.path.getmtime(fp32) > os.path.getmtime(path)):
            print(f"WARNING: {os.path.basename(fp32)} is newer than "
                  f"{os.path.basename(path)} - the int8 quantisation is stale, so it is "
                  f"being skipped. Re-quantise with: python export_onnx.py --int8")
            continue
        return path
    return None


def num_threads() -> int:
    """Intra-op thread count for both onnxruntime and torch. See module docstring."""
    return int(os.environ.get("CHESS_NUM_THREADS", "4"))


class OnnxPolicyValue:
    """
    onnxruntime wrapper around the dual net's forward pass.

    Used instead of torch at serving time: it is faster at batch size 1 (which is what
    MCTS does) and it means a deployment does not have to ship torch at all.
    Returns raw policy logits and value — masking and softmax stay with the caller.
    """

    def __init__(self, path: str, threads: int = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or num_threads()
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Stop ORT's worker threads spinning between calls. The isolated forward gets
        # marginally slower, but a real search gets faster overall: most of a move is
        # python-chess, and spinning ORT threads were holding cores it needed.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self.session = ort.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.path = path

    def run(self, x: np.ndarray):
        """x: (B, 13, 8, 8) float32 -> (policy_logits, value) as numpy arrays."""
        return self.session.run(None, {self.input_name: x})


def get_onnx_session(path: str = None):
    """
    Return a cached ONNX session for inference, or None to fall back to torch.

    Resolved once per process. Set CHESS_USE_ONNX=0 to force the torch path, or
    CHESS_ONNX_MODEL=<name-or-path> to pin a specific export.
    """
    global _ort_session, _ort_resolved
    if _ort_resolved:
        return _ort_session
    _ort_resolved = True

    if os.environ.get("CHESS_USE_ONNX", "1") == "0":
        return None

    path = path or resolve_onnx_path()
    if path is None or not os.path.exists(path):
        return None

    # Refuse a stale export. If the checkpoint has been retrained since the ONNX file
    # was written, serving from ONNX would quietly use the OLD weights while every log
    # line still names the new checkpoint - a silent strength regression that looks
    # like the model got worse for no reason. Fall back to torch and say so.
    if (os.path.exists(DEFAULT_CKPT_PATH)
            and os.path.getmtime(DEFAULT_CKPT_PATH) > os.path.getmtime(path)):
        print(f"WARNING: {os.path.basename(DEFAULT_CKPT_PATH)} is newer than "
              f"{os.path.basename(path)} - the ONNX export is stale, so falling back "
              f"to torch inference. Re-export with: python export_onnx.py")
        return None

    try:
        _ort_session = OnnxPolicyValue(path)
        print(f"ONNX inference enabled: {os.path.basename(path)}")
    except Exception as e:
        print(f"WARNING: could not start ONNX session ({e}); using torch inference.")
        _ort_session = None
    return _ort_session

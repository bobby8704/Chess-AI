"""
Enhanced Neural Network Architecture for Chess AI

This module provides:
- PolicyNet: Predicts move probabilities
- ValueNet: Evaluates board positions (existing, re-exported)
- DualNet: Combined policy + value network (AlphaZero-style)
- Move encoding/decoding utilities

The policy network outputs a probability distribution over all possible
chess moves (encoded as from_square * 64 + to_square with promotion handling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from typing import Tuple, List, Optional, Dict
from features import board_to_tensor

# Total possible moves: 64 * 64 = 4096 base moves
# Plus promotions: 8 files * 4 promotion pieces * 2 directions (forward, capture) * 2 colors
# Simplified: we use 64*64 + 64*3 (underpromotions) = 4288
# For simplicity, we'll use a flat 4672 output (73 * 64 - covers all legal moves)
# Actually, let's use the simpler approach: 64*73 = 4672 (queen-like moves + knight moves + pawn moves)
# Even simpler: 64 * 64 = 4096 for from-to, promotions encoded separately
POLICY_OUTPUT_SIZE = 4096 + 192  # 4096 base + 64*3 underpromotions = 4288


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


def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """
    Create a mask of legal moves for the current position.

    Returns a tensor of shape (POLICY_OUTPUT_SIZE,) with 1.0 for legal moves
    and 0.0 for illegal moves.
    """
    mask = torch.zeros(POLICY_OUTPUT_SIZE)
    for move in board.legal_moves:
        idx = encode_move(move)
        if idx < POLICY_OUTPUT_SIZE:
            mask[idx] = 1.0
    return mask


class PolicyNet(nn.Module):
    """
    Neural network for predicting move probabilities.

    Input: Board features (781 dimensions)
    Output: Log-probabilities over all possible moves
    """

    def __init__(self, input_dim: int = 781):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, POLICY_OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (not softmax-ed)."""
        return self.net(x)

    def get_move_probs(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability distribution over legal moves only.

        Args:
            x: Board features
            legal_mask: Binary mask of legal moves

        Returns:
            Probability distribution (sums to 1 over legal moves)
        """
        logits = self.forward(x)
        # Mask illegal moves with large negative value
        masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)


class ValueNet(nn.Module):
    """
    Neural network for evaluating board positions.

    Input: Board features (781 dimensions)
    Output: Value in [-1, 1] from current player's perspective
    """

    def __init__(self, input_dim: int = 781):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualNet(nn.Module):
    """
    Combined Policy + Value network (AlphaZero-style).

    Uses a shared backbone with separate heads for policy and value.
    More parameter-efficient than separate networks.

    Input: Board features (781 dimensions)
    Output: (policy_logits, value)
    """

    def __init__(self, input_dim: int = 781, hidden_dim: int = 512):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, POLICY_OUTPUT_SIZE),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Returns:
            (policy_logits, value) where:
            - policy_logits: Raw logits of shape (batch, POLICY_OUTPUT_SIZE)
            - value: Scalar value of shape (batch, 1)
        """
        features = self.backbone(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value

    def get_policy_value(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy probabilities and value.

        Args:
            x: Board features
            legal_mask: Binary mask of legal moves

        Returns:
            (policy_probs, value)
        """
        policy_logits, value = self.forward(x)
        masked_logits = policy_logits.masked_fill(legal_mask == 0, float('-inf'))
        policy_probs = F.softmax(masked_logits, dim=-1)
        return policy_probs, value


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class DualNetResidual(nn.Module):
    """
    Deeper dual network with residual connections.

    Uses residual blocks for better gradient flow in deep networks.
    """

    def __init__(
        self,
        input_dim: int = 781,
        hidden_dim: int = 512,
        num_residual_blocks: int = 4
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, POLICY_OUTPUT_SIZE),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.input_proj(x)
        features = self.residual_tower(features)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value

    def get_policy_value(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value = self.forward(x)
        masked_logits = policy_logits.masked_fill(legal_mask == 0, float('-inf'))
        policy_probs = F.softmax(masked_logits, dim=-1)
        return policy_probs, value


# ==================== CNN Architecture ====================

class ConvResBlock(nn.Module):
    """Convolutional residual block: Conv -> BN -> ReLU -> Conv -> BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DualNetCNN(nn.Module):
    """
    CNN with residual blocks for chess (AlphaZero-style).

    Input: (batch, 13, 8, 8) — 12 piece planes + 1 auxiliary plane
    Output: policy logits (4288) + value scalar

    Architecture:
        Input conv (13 -> filters) -> N residual blocks -> policy head + value head
    """

    def __init__(
        self,
        in_channels: int = 13,
        num_filters: int = 128,
        num_res_blocks: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ConvResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head: conv 1x1 -> flatten -> FC -> 4288
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * 64, POLICY_OUTPUT_SIZE)

        # Value head: conv 1x1 -> flatten -> FC -> FC -> tanh
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 13, 8, 8)
        features = self.input_conv(x)
        features = self.res_tower(features)

        # Policy
        p = self.policy_conv(features)
        p = p.view(p.size(0), -1)  # flatten
        policy = self.policy_fc(p)

        # Value
        v = self.value_conv(features)
        v = v.view(v.size(0), -1)  # flatten
        value = self.value_fc(v)

        return policy, value

    def get_policy_value(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value = self.forward(x)
        masked_logits = policy_logits.masked_fill(legal_mask == 0, float('-inf'))
        policy_probs = F.softmax(masked_logits, dim=-1)
        return policy_probs, value


# ==================== Model Loading Utilities ====================

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dual_model: Optional[DualNet] = None


def load_dual_model(path: str):
    """Load a trained dual network from disk. Supports DualNet, DualNetResidual, and DualNetCNN."""
    global _dual_model
    ckpt = torch.load(path, map_location=_device, weights_only=False)

    model_type = ckpt.get("model_type", "DualNet")

    if model_type == "DualNetCNN":
        in_channels = ckpt.get("in_channels", 13)
        num_filters = ckpt.get("num_filters", 128)
        num_res_blocks = ckpt.get("num_res_blocks", 6)
        _dual_model = DualNetCNN(in_channels, num_filters, num_res_blocks).to(_device)
    elif model_type == "DualNetResidual":
        input_dim = ckpt.get("input_dim", 781)
        hidden_dim = ckpt.get("hidden_dim", 512)
        num_blocks = ckpt.get("num_residual_blocks", 4)
        _dual_model = DualNetResidual(input_dim, hidden_dim, num_blocks).to(_device)
    else:
        input_dim = ckpt.get("input_dim", 781)
        hidden_dim = ckpt.get("hidden_dim", 512)
        _dual_model = DualNet(input_dim, hidden_dim).to(_device)

    _dual_model.load_state_dict(ckpt["state_dict"])
    _dual_model.eval()
    return _dual_model


def is_cnn_model(model) -> bool:
    """Check if the loaded model is a CNN (needs 2D input)."""
    return isinstance(model, DualNetCNN)


def get_dual_model() -> Optional[DualNet]:
    """Get the currently loaded dual model."""
    return _dual_model


def evaluate_position(board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
    """
    Evaluate a position using the dual network.

    Returns:
        (move_probs, value) where:
        - move_probs: Dict mapping legal moves to their probabilities
        - value: Position value from current player's perspective
    """
    global _dual_model

    if _dual_model is None:
        # Return uniform policy and neutral value
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0
        uniform_prob = 1.0 / len(legal_moves)
        return {move: uniform_prob for move in legal_moves}, 0.0

    # Convert board to tensor
    x = torch.from_numpy(board_to_tensor(board)).float().to(_device).unsqueeze(0)
    legal_mask = get_legal_move_mask(board).to(_device).unsqueeze(0)

    with torch.no_grad():
        policy_probs, value = _dual_model.get_policy_value(x, legal_mask)

    # Convert policy to move dictionary
    policy_probs = policy_probs.squeeze(0).cpu().numpy()
    value = value.item()

    move_probs = {}
    for move in board.legal_moves:
        idx = encode_move(move)
        if idx < POLICY_OUTPUT_SIZE:
            move_probs[move] = float(policy_probs[idx])

    # Normalize (should already sum to 1, but ensure numerical stability)
    total = sum(move_probs.values())
    if total > 0:
        move_probs = {m: p / total for m, p in move_probs.items()}

    return move_probs, value


if __name__ == "__main__":
    # Test the neural network components
    print("Testing Neural Network Components...")
    print("=" * 50)

    # Test move encoding/decoding
    board = chess.Board()
    print("\nTesting move encoding/decoding:")
    for move in list(board.legal_moves)[:5]:
        idx = encode_move(move)
        decoded = decode_move(idx, board)
        status = "OK" if decoded == move else "FAIL"
        print(f"  {move} -> {idx} -> {decoded} [{status}]")

    # Test legal move mask
    mask = get_legal_move_mask(board)
    num_legal = int(mask.sum().item())
    actual_legal = len(list(board.legal_moves))
    print(f"\nLegal move mask: {num_legal} moves (actual: {actual_legal})")

    # Test PolicyNet
    print("\nTesting PolicyNet:")
    policy_net = PolicyNet()
    x = torch.randn(1, 781)
    logits = policy_net(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in policy_net.parameters()):,}")

    # Test ValueNet
    print("\nTesting ValueNet:")
    value_net = ValueNet()
    value = value_net(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {value.shape}")
    print(f"  Value: {value.item():.4f}")
    print(f"  Parameters: {sum(p.numel() for p in value_net.parameters()):,}")

    # Test DualNet
    print("\nTesting DualNet:")
    dual_net = DualNet()
    policy, value = dual_net(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dual_net.parameters()):,}")

    # Test DualNetResidual
    print("\nTesting DualNetResidual:")
    dual_res = DualNetResidual(num_residual_blocks=4)
    policy, value = dual_res(x)
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Parameters: {sum(p.numel() for p in dual_res.parameters()):,}")

    # Test with actual board
    print("\nTesting with actual board position:")
    x = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0)
    mask = get_legal_move_mask(board).unsqueeze(0)
    probs, value = dual_net.get_policy_value(x, mask)
    print(f"  Value: {value.item():.4f}")
    print(f"  Top 5 move probabilities:")
    probs_np = probs.squeeze(0).detach().numpy()
    top_indices = np.argsort(probs_np)[-5:][::-1]
    for idx in top_indices:
        move = decode_move(idx, board)
        if move:
            print(f"    {move}: {probs_np[idx]:.4f}")

    print("\nAll tests passed!")

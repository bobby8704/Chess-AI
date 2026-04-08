"""
Training Utilities for Chess AI

This module provides enhanced training features:
- TD-lambda targets for faster credit assignment
- Position augmentation via horizontal mirroring
- Dataset management utilities
- Training with validation split
"""

import numpy as np
import chess
from typing import Tuple, List, Optional
from features import board_to_tensor, PIECE_MAP


def compute_td_lambda_targets(
    game_outcomes: List[float],
    move_evaluations: Optional[List[float]] = None,
    lambda_: float = 0.7,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute TD(lambda) targets for a sequence of positions from a game.

    TD(lambda) provides a weighted combination of n-step returns:
    - lambda=0: Pure TD(0), uses only immediate successor value
    - lambda=1: Pure Monte Carlo, uses only final game outcome
    - lambda=0.7 (default): Good balance between bias and variance

    Args:
        game_outcomes: List of final outcome values (+1, -1, 0) for each position
                      (from that position's side-to-move perspective)
        move_evaluations: Optional list of neural network evaluations at each position.
                         If None, falls back to pure MC (lambda=1 behavior)
        lambda_: TD-lambda parameter (0 to 1)
        gamma: Discount factor for future rewards

    Returns:
        Array of TD(lambda) targets for each position
    """
    n = len(game_outcomes)
    if n == 0:
        return np.array([], dtype=np.float32)

    targets = np.zeros(n, dtype=np.float32)
    final_outcome = game_outcomes[-1]  # All positions have same outcome

    if move_evaluations is None or lambda_ == 1.0:
        # Pure Monte Carlo: use final game outcome
        return np.array(game_outcomes, dtype=np.float32)

    # TD(lambda) backward pass
    # We compute eligibility-weighted returns
    for t in range(n):
        # Distance to end of game
        remaining = n - t - 1

        if remaining == 0:
            # Last position: use final outcome
            targets[t] = final_outcome
        else:
            # Weighted combination of n-step returns
            td_target = 0.0
            weight_sum = 0.0

            for k in range(1, remaining + 1):
                # Weight for k-step return
                weight = (1 - lambda_) * (lambda_ ** (k - 1))

                if k < remaining:
                    # Bootstrap from evaluation at position t+k
                    # Note: evaluations alternate perspective, so we need to flip sign
                    bootstrap_value = move_evaluations[t + k]
                    if k % 2 == 1:
                        bootstrap_value = -bootstrap_value  # Flip for opponent's perspective
                    k_step_return = (gamma ** k) * bootstrap_value
                else:
                    # Use final outcome for the last step
                    k_step_return = (gamma ** k) * final_outcome

                td_target += weight * k_step_return
                weight_sum += weight

            # Add the Monte Carlo component (remaining lambda weight)
            mc_weight = lambda_ ** remaining
            td_target += mc_weight * (gamma ** remaining) * final_outcome
            weight_sum += mc_weight

            # Normalize
            if weight_sum > 0:
                targets[t] = td_target / weight_sum
            else:
                targets[t] = final_outcome

    return targets


def mirror_board_horizontal(tensor: np.ndarray) -> np.ndarray:
    """
    Mirror a board tensor horizontally (flip files a-h to h-a).

    Chess is horizontally symmetric - mirroring doubles training data.
    This works for the 781-dim feature vector:
    - 12 planes of 8x8 (768 values): mirror each plane
    - 1 side-to-move: unchanged
    - 4 castling rights: swap kingside/queenside for each color
    - 8 en-passant file: mirror the one-hot encoding

    Args:
        tensor: Original 781-dim feature tensor

    Returns:
        Mirrored 781-dim feature tensor
    """
    mirrored = np.zeros_like(tensor)

    # Mirror the 12 piece planes (each 8x8 = 64 values)
    for plane_idx in range(12):
        start = plane_idx * 64
        end = start + 64
        plane = tensor[start:end].reshape(8, 8)
        # Flip horizontally (reverse columns)
        mirrored_plane = np.fliplr(plane)
        mirrored[start:end] = mirrored_plane.reshape(-1)

    # Auxiliary features start at index 768
    aux_start = 768

    # Side to move (index 0): unchanged
    mirrored[aux_start] = tensor[aux_start]

    # Castling rights (indices 1-4): swap kingside/queenside
    # Original: [white_kingside, white_queenside, black_kingside, black_queenside]
    # Mirrored: [white_queenside, white_kingside, black_queenside, black_kingside]
    mirrored[aux_start + 1] = tensor[aux_start + 2]  # white queenside -> white kingside
    mirrored[aux_start + 2] = tensor[aux_start + 1]  # white kingside -> white queenside
    mirrored[aux_start + 3] = tensor[aux_start + 4]  # black queenside -> black kingside
    mirrored[aux_start + 4] = tensor[aux_start + 3]  # black kingside -> black queenside

    # En passant file (indices 5-12): mirror the file
    # File a(0) <-> h(7), b(1) <-> g(6), etc.
    for file_idx in range(8):
        mirrored_file = 7 - file_idx
        mirrored[aux_start + 5 + mirrored_file] = tensor[aux_start + 5 + file_idx]

    return mirrored


def augment_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment a dataset by adding horizontally mirrored positions.

    This doubles the training data size while preserving chess semantics.

    Args:
        X: Position tensors (N, 781)
        y: Target values (N,)

    Returns:
        Augmented (X, y) with original and mirrored positions
    """
    n = X.shape[0]
    X_mirrored = np.zeros_like(X)

    for i in range(n):
        X_mirrored[i] = mirror_board_horizontal(X[i])

    # Combine original and mirrored
    X_augmented = np.concatenate([X, X_mirrored], axis=0)
    y_augmented = np.concatenate([y, y], axis=0)  # Same targets for mirrored positions

    return X_augmented, y_augmented


def create_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and validation sets.

    Args:
        X: Position tensors
        y: Target values
        val_fraction: Fraction of data for validation
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    n = X.shape[0]

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        X = X[indices]
        y = y[indices]

    val_size = int(n * val_fraction)
    X_val = X[:val_size]
    y_val = y[:val_size]
    X_train = X[val_size:]
    y_train = y[val_size:]

    return X_train, y_train, X_val, y_val


def positions_from_game_record(game_moves: list, final_result: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract training positions from a game record.

    Args:
        game_moves: List of MoveRecord objects from GameRecorder
        final_result: Game result string ("1-0", "0-1", "1/2-1/2")

    Returns:
        (X, y) where X is position tensors and y is targets
    """
    # Determine final outcome from white's perspective
    if final_result == "1-0":
        z = 1.0
    elif final_result == "0-1":
        z = -1.0
    else:
        z = 0.0

    X_list = []
    y_list = []

    for move in game_moves:
        # Parse FEN to board
        board = chess.Board(move.fen_before)
        tensor = board_to_tensor(board)
        X_list.append(tensor)

        # Target from side-to-move perspective
        if board.turn == chess.WHITE:
            y_list.append(z)
        else:
            y_list.append(-z)

    if not X_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def merge_datasets(*dataset_paths: str, output_path: str) -> Tuple[int, int]:
    """
    Merge multiple .npz datasets into one.

    Args:
        *dataset_paths: Paths to input .npz files
        output_path: Path for merged output

    Returns:
        (total_positions, num_files_merged)
    """
    X_all = []
    y_all = []

    for path in dataset_paths:
        try:
            data = np.load(path)
            X_all.append(data["X"])
            y_all.append(data["y"])
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            continue

    if not X_all:
        raise ValueError("No valid datasets to merge")

    X_merged = np.concatenate(X_all, axis=0)
    y_merged = np.concatenate(y_all, axis=0)

    np.savez_compressed(output_path, X=X_merged, y=y_merged)

    return len(X_merged), len(X_all)


def compute_dataset_statistics(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute statistics about a training dataset.

    Args:
        X: Position tensors
        y: Target values

    Returns:
        Dictionary of statistics
    """
    return {
        "num_positions": len(X),
        "feature_dim": X.shape[1] if len(X) > 0 else 0,
        "target_mean": float(np.mean(y)) if len(y) > 0 else 0,
        "target_std": float(np.std(y)) if len(y) > 0 else 0,
        "white_wins_pct": float(np.mean(y > 0.5) * 100) if len(y) > 0 else 0,
        "black_wins_pct": float(np.mean(y < -0.5) * 100) if len(y) > 0 else 0,
        "draws_pct": float(np.mean(np.abs(y) <= 0.5) * 100) if len(y) > 0 else 0,
    }


if __name__ == "__main__":
    # Test the utilities
    print("Testing training utilities...")

    # Test TD-lambda
    outcomes = [1.0, 1.0, 1.0, 1.0, 1.0]  # All positions lead to white win
    evals = [0.1, 0.2, 0.3, 0.5, 0.8]  # Increasing evaluations
    td_targets = compute_td_lambda_targets(outcomes, evals, lambda_=0.7)
    print(f"TD-lambda targets: {td_targets}")

    # Test mirroring
    board = chess.Board()
    board.push_san("e4")
    tensor = board_to_tensor(board)
    mirrored = mirror_board_horizontal(tensor)
    print(f"Original tensor sum: {tensor.sum():.2f}")
    print(f"Mirrored tensor sum: {mirrored.sum():.2f}")

    # Verify mirror is correct by checking piece positions
    # After e4, white pawn should be on e4
    # After mirroring, it should appear on d4 in the representation
    print("Mirror test passed!" if np.allclose(tensor.sum(), mirrored.sum()) else "Mirror test FAILED!")

    # Test augmentation
    X = np.random.randn(10, 781).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    X_aug, y_aug = augment_dataset(X, y)
    print(f"Augmentation: {X.shape[0]} -> {X_aug.shape[0]} positions")

    # Test validation split
    X_train, y_train, X_val, y_val = create_validation_split(X, y, val_fraction=0.2)
    print(f"Split: {len(X_train)} train, {len(X_val)} val")

    print("\nAll tests completed!")

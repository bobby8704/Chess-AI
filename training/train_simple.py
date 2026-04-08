#!/usr/bin/env python3
"""
Simple but Effective Training Script

Memory-efficient training with balanced data:
- Random games for wins/losses
- Minimax games for quality
- Limits positions to avoid memory issues
"""

import os
import time
import shutil
from pathlib import Path

import numpy as np


def run_training(duration_minutes: int = 60, verbose: bool = True):
    """Run memory-efficient training."""
    from performance import parallel_selfplay
    from train_improved import train

    output_dir = "data/simple_training"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIMPLE CHESS AI TRAINING")
    print(f"Duration: {duration_minutes} minutes")
    print("=" * 60)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    all_X = []
    all_y = []
    total_games = 0
    max_positions = 200000  # Limit to avoid memory issues

    def time_remaining():
        return max(0, end_time - time.time())

    def log(msg):
        if verbose:
            elapsed = (time.time() - start_time) / 60
            print(f"[{elapsed:.1f}m] {msg}")

    # Phase 1: Random games (decisive outcomes)
    log("Phase 1: Random games for wins/losses")
    phase1_games = 200

    pgns, X, y = parallel_selfplay(
        n_games=phase1_games,
        engine_w="random",
        engine_b="random",
        n_workers=8,
        verbose=True
    )

    # Limit positions from random games (they can be very long)
    if len(X) > 50000:
        indices = np.random.choice(len(X), 50000, replace=False)
        X = X[indices]
        y = y[indices]

    all_X.append(X)
    all_y.append(y)
    total_games += len(pgns)
    log(f"  Got {len(pgns)} games, {len(X)} positions")

    # Phase 2: Asymmetric games (minimax vs random)
    log("Phase 2: Minimax vs Random")

    while time_remaining() > 300 and sum(len(x) for x in all_X) < max_positions:
        # Minimax white vs random black
        pgns, X, y = parallel_selfplay(
            n_games=50,
            engine_w="minimax",
            engine_b="random",
            depth_w=2,
            depth_b=1,
            n_workers=8,
            verbose=False
        )
        all_X.append(X)
        all_y.append(y)
        total_games += len(pgns)

        # Random white vs minimax black
        pgns, X, y = parallel_selfplay(
            n_games=50,
            engine_w="random",
            engine_b="minimax",
            depth_w=1,
            depth_b=2,
            n_workers=8,
            verbose=False
        )
        all_X.append(X)
        all_y.append(y)
        total_games += len(pgns)

        total_pos = sum(len(x) for x in all_X)
        log(f"  {total_games} games, {total_pos} positions")

        if total_pos >= max_positions:
            break

    # Combine data
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Sample if too many positions
    if len(X_all) > max_positions:
        indices = np.random.choice(len(X_all), max_positions, replace=False)
        X_all = X_all[indices]
        y_all = y_all[indices]

    # Check distribution
    wins = (y_all > 0.5).sum()
    losses = (y_all < -0.5).sum()
    draws = len(y_all) - wins - losses
    log(f"Data: W:{wins} L:{losses} D:{draws} (Total: {len(y_all)})")

    # Save dataset
    dataset_path = os.path.join(output_dir, "dataset.npz")
    np.savez_compressed(dataset_path, X=X_all, y=y_all)

    # Train
    log("Training model...")
    model_path = os.path.join(output_dir, "model.pt")

    train(
        dataset_path=dataset_path,
        epochs=50,
        batch_size=512,
        lr=1e-3,
        out_path=model_path,
        augment=True,
        early_stopping=True,
        patience=10,
        verbose=verbose
    )

    # Copy to main location
    final_path = "value_model_simple.pt"
    shutil.copy(model_path, final_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Total games: {total_games}")
    print(f"Total positions: {len(X_all)}")
    print(f"Model saved to: {final_path}")
    print("=" * 60)
    print("\nTo play: python main.py --load_model value_model_simple.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()
    run_training(duration_minutes=args.duration)

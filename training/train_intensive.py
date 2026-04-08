#!/usr/bin/env python3
"""
Intensive Training Script for Chess AI

This script runs an intensive training session designed to create a strong AI
that humans will struggle against. It uses all available CPU cores and runs
for a specified duration.

Usage:
    python train_intensive.py --duration 60  # 60 minutes of training
"""

import argparse
import os
import time
import shutil
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def get_system_info():
    """Get system information for training."""
    info = {
        "cpu_count": mp.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    return info


def run_intensive_training(
    duration_minutes: int = 60,
    output_dir: str = "data/intensive_training",
    verbose: bool = True
):
    """
    Run intensive training for the specified duration.

    Strategy:
    - Phase 1 (first 30%): Minimax self-play at depth 2-3 for high-quality games
    - Phase 2 (30-100%): NN self-play with periodic retraining
    """
    from performance import parallel_selfplay
    from train_dual import train_dual

    # Setup directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    datasets_dir = os.path.join(output_dir, "datasets")
    Path(models_dir).mkdir(exist_ok=True)
    Path(datasets_dir).mkdir(exist_ok=True)

    # System info
    sys_info = get_system_info()
    n_workers = max(1, sys_info["cpu_count"] - 2)

    print("=" * 70)
    print("INTENSIVE CHESS AI TRAINING")
    print("=" * 70)
    print(f"Duration: {duration_minutes} minutes")
    print(f"CPU Workers: {n_workers}")
    print(f"CUDA: {sys_info['cuda_available']}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    # Training state
    iteration = 0
    best_model_path = None
    all_X = []
    all_y = []
    total_games = 0
    total_positions = 0

    def time_remaining():
        return max(0, end_time - time.time())

    def time_elapsed():
        return time.time() - start_time

    def log(msg):
        if verbose:
            elapsed = time_elapsed()
            remaining = time_remaining()
            print(f"[{elapsed/60:.1f}m / {remaining/60:.1f}m left] {msg}")

    # ===== PHASE 1: Minimax Self-Play =====
    phase1_end = start_time + (duration_minutes * 60 * 0.25)  # First 25%

    log("PHASE 1: Generating high-quality games with minimax")

    while time.time() < phase1_end and time_remaining() > 60:
        iteration += 1

        # Use minimax at depth 2 (faster but still good)
        games_this_iter = 50

        log(f"Iteration {iteration}: Minimax self-play ({games_this_iter} games, depth 2)...")

        try:
            pgns, X, y = parallel_selfplay(
                n_games=games_this_iter,
                engine_w="minimax",
                engine_b="minimax",
                depth_w=2,
                depth_b=2,
                n_workers=n_workers,
                verbose=False
            )

            all_X.append(X)
            all_y.append(y)
            total_games += len(pgns)
            total_positions += len(X)

            log(f"  Generated {len(pgns)} games, {len(X)} positions (total: {total_positions})")

        except Exception as e:
            log(f"  Error in self-play: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ===== Initial Training =====
    if all_X:
        log("Initial training on minimax games...")

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        dataset_path = os.path.join(datasets_dir, "dataset_phase1.npz")
        np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

        model_path = os.path.join(models_dir, "model_phase1.pt")

        try:
            # Use simpler architecture for faster training
            results = train_dual(
                dataset_path=dataset_path,
                epochs=20,  # Reduced epochs
                batch_size=512,
                lr=1e-3,
                out_path=model_path,
                use_residual=False,  # Use simpler DualNet
                hidden_dim=256,
                augment=True,
                early_stopping=True,
                patience=5,
                verbose=verbose
            )

            best_model_path = model_path
            log(f"  Trained model: val_loss={results['best_val_loss']:.5f}")

        except Exception as e:
            log(f"  Training error: {e}")
            import traceback
            traceback.print_exc()

    # ===== PHASE 2: NN Self-Play =====
    log("PHASE 2: Self-play with neural network")

    nn_iteration = 0
    nn_games_X = []
    nn_games_y = []

    while time_remaining() > 180:  # Keep going until 3 min left
        iteration += 1
        nn_iteration += 1

        games_this_iter = 30
        log(f"Iteration {iteration}: NN self-play ({games_this_iter} games)...")

        try:
            pgns, X, y = parallel_selfplay(
                n_games=games_this_iter,
                engine_w="nn",
                engine_b="nn",
                depth_w=2,
                depth_b=2,
                model_path=best_model_path,
                n_workers=n_workers,
                verbose=False
            )

            nn_games_X.append(X)
            nn_games_y.append(y)
            total_games += len(pgns)
            total_positions += len(X)

            log(f"  Generated {len(pgns)} games, {len(X)} positions (total: {total_positions})")

        except Exception as e:
            log(f"  Self-play error: {e}")
            continue

        # Retrain every 3 iterations
        if nn_iteration % 3 == 0 and time_remaining() > 300:
            log("  Retraining on accumulated data...")

            X_new = np.concatenate(nn_games_X, axis=0)
            y_new = np.concatenate(nn_games_y, axis=0)

            X_combined = np.concatenate([X_combined, X_new], axis=0)
            y_combined = np.concatenate([y_combined, y_new], axis=0)

            dataset_path = os.path.join(datasets_dir, f"dataset_iter{iteration}.npz")
            np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

            model_path = os.path.join(models_dir, f"model_iter{iteration}.pt")

            try:
                results = train_dual(
                    dataset_path=dataset_path,
                    epochs=15,
                    batch_size=512,
                    lr=5e-4,
                    out_path=model_path,
                    use_residual=False,
                    hidden_dim=256,
                    augment=True,
                    early_stopping=True,
                    patience=5,
                    verbose=False
                )

                best_model_path = model_path
                log(f"  Model updated: val_loss={results['best_val_loss']:.5f}")

            except Exception as e:
                log(f"  Training error: {e}")

            nn_games_X = []
            nn_games_y = []

    # ===== PHASE 3: Final Training =====
    log("PHASE 3: Final intensive training")

    if nn_games_X:
        X_new = np.concatenate(nn_games_X, axis=0)
        y_new = np.concatenate(nn_games_y, axis=0)
        X_combined = np.concatenate([X_combined, X_new], axis=0)
        y_combined = np.concatenate([y_combined, y_new], axis=0)

    if len(X_combined) > 0:
        dataset_path = os.path.join(datasets_dir, "dataset_final.npz")
        np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

        log(f"Final training on {len(X_combined)} positions...")

        final_model_path = os.path.join(models_dir, "model_final.pt")

        try:
            # Final training with more epochs
            results = train_dual(
                dataset_path=dataset_path,
                epochs=30,
                batch_size=512,
                lr=1e-4,
                out_path=final_model_path,
                use_residual=False,
                hidden_dim=256,
                augment=True,
                early_stopping=True,
                patience=10,
                verbose=verbose
            )

            best_model_path = final_model_path
            log(f"Final model: val_loss={results['best_val_loss']:.5f}")

        except Exception as e:
            log(f"Final training error: {e}")

    # ===== Save final model =====
    if best_model_path and os.path.exists(best_model_path):
        final_path = "value_model_intensive.pt"
        shutil.copy(best_model_path, final_path)
        log(f"Saved final model to: {final_path}")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Iterations: {iteration}")
    print(f"Total games: {total_games}")
    print(f"Total positions: {total_positions}")
    print(f"Final model: {best_model_path}")
    print("=" * 70)
    print("\nTo play against the AI, run:")
    print(f"  python main.py --load_model value_model_intensive.pt")
    print("=" * 70)

    return best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Run intensive training to create a strong chess AI"
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Training duration in minutes (default: 60)")
    parser.add_argument("--output", type=str, default="data/intensive_training",
                        help="Output directory")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output")

    args = parser.parse_args()

    run_intensive_training(
        duration_minutes=args.duration,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

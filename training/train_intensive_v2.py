#!/usr/bin/env python3
"""
Intensive Training Script v2 - Fixed for Better Learning

The key fix: Generate games with DECISIVE outcomes (wins/losses, not just draws)
by mixing random play with minimax to get varied game results.
"""

import argparse
import os
import time
import shutil
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch


def run_intensive_training(
    duration_minutes: int = 60,
    output_dir: str = "data/intensive_training_v2",
    verbose: bool = True
):
    """
    Run intensive training with better data generation.

    Key improvements:
    1. Use random moves early game to create variety
    2. Mix skill levels to get decisive games
    3. Ensure balanced win/loss/draw distribution
    """
    from performance import parallel_selfplay
    from train_dual import train_dual

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    datasets_dir = os.path.join(output_dir, "datasets")
    Path(models_dir).mkdir(exist_ok=True)
    Path(datasets_dir).mkdir(exist_ok=True)

    n_workers = max(1, mp.cpu_count() - 2)

    print("=" * 70)
    print("INTENSIVE CHESS AI TRAINING v2")
    print("=" * 70)
    print(f"Duration: {duration_minutes} minutes")
    print(f"CPU Workers: {n_workers}")
    print(f"Strategy: Mixed skill levels for decisive games")
    print("=" * 70)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

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

    def check_distribution(y_arr):
        wins = (y_arr > 0.5).sum()
        losses = (y_arr < -0.5).sum()
        draws = len(y_arr) - wins - losses
        total = len(y_arr)
        log(f"  Distribution: W:{wins} ({100*wins/total:.1f}%) L:{losses} ({100*losses/total:.1f}%) D:{draws} ({100*draws/total:.1f}%)")

    # ===== PHASE 1: Random games for decisive outcomes =====
    phase1_end = start_time + (duration_minutes * 60 * 0.15)

    log("PHASE 1: Random games (to get wins/losses)")

    while time.time() < phase1_end and time_remaining() > 60:
        iteration += 1
        games_this_iter = 100

        log(f"Iteration {iteration}: Random self-play ({games_this_iter} games)...")

        try:
            pgns, X, y = parallel_selfplay(
                n_games=games_this_iter,
                engine_w="random",
                engine_b="random",
                depth_w=1,
                depth_b=1,
                n_workers=n_workers,
                verbose=False
            )

            all_X.append(X)
            all_y.append(y)
            total_games += len(pgns)
            total_positions += len(X)

            log(f"  Generated {len(pgns)} games, {len(X)} positions")

        except Exception as e:
            log(f"  Error: {e}")
            continue

    # ===== PHASE 2: Asymmetric games (strong vs weak) =====
    phase2_end = start_time + (duration_minutes * 60 * 0.35)

    log("PHASE 2: Asymmetric games (minimax vs random)")

    while time.time() < phase2_end and time_remaining() > 60:
        iteration += 1
        games_this_iter = 50

        # Alternate which side is strong
        if iteration % 2 == 0:
            eng_w, eng_b = "minimax", "random"
            depth_w, depth_b = 2, 1
        else:
            eng_w, eng_b = "random", "minimax"
            depth_w, depth_b = 1, 2

        log(f"Iteration {iteration}: {eng_w}(d{depth_w}) vs {eng_b}(d{depth_b})...")

        try:
            pgns, X, y = parallel_selfplay(
                n_games=games_this_iter,
                engine_w=eng_w,
                engine_b=eng_b,
                depth_w=depth_w,
                depth_b=depth_b,
                n_workers=n_workers,
                verbose=False
            )

            all_X.append(X)
            all_y.append(y)
            total_games += len(pgns)
            total_positions += len(X)

            log(f"  Generated {len(pgns)} games, {len(X)} positions")

        except Exception as e:
            log(f"  Error: {e}")
            continue

    # ===== Initial Training =====
    if all_X:
        log("Initial training on varied games...")

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        check_distribution(y_combined)

        dataset_path = os.path.join(datasets_dir, "dataset_phase2.npz")
        np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

        model_path = os.path.join(models_dir, "model_phase2.pt")

        try:
            # Skip augmentation for large datasets to save memory
            should_augment = len(X_combined) < 500000
            results = train_dual(
                dataset_path=dataset_path,
                epochs=30,
                batch_size=512,
                lr=1e-3,
                out_path=model_path,
                use_residual=False,
                hidden_dim=256,
                augment=should_augment,
                early_stopping=True,
                patience=7,
                verbose=verbose
            )

            best_model_path = model_path
            log(f"  Trained model: val_loss={results['best_val_loss']:.5f}")

        except Exception as e:
            log(f"  Training error: {e}")
            import traceback
            traceback.print_exc()

    # ===== PHASE 3: NN vs Random (NN should win) =====
    phase3_end = start_time + (duration_minutes * 60 * 0.60)

    log("PHASE 3: NN vs Random (teaching NN to win)")

    nn_X = []
    nn_y = []

    while time.time() < phase3_end and time_remaining() > 120:
        iteration += 1
        games_this_iter = 40

        # NN plays both sides against random
        if iteration % 2 == 0:
            eng_w, eng_b = "nn", "random"
        else:
            eng_w, eng_b = "random", "nn"

        log(f"Iteration {iteration}: {eng_w} vs {eng_b}...")

        try:
            pgns, X, y = parallel_selfplay(
                n_games=games_this_iter,
                engine_w=eng_w,
                engine_b=eng_b,
                depth_w=2,
                depth_b=1,
                model_path=best_model_path,
                n_workers=n_workers,
                verbose=False
            )

            nn_X.append(X)
            nn_y.append(y)
            total_games += len(pgns)
            total_positions += len(X)

            log(f"  Generated {len(pgns)} games, {len(X)} positions")

        except Exception as e:
            log(f"  Error: {e}")
            continue

        # Retrain every 4 iterations
        if len(nn_X) >= 4 and time_remaining() > 300:
            log("  Retraining...")

            X_new = np.concatenate(nn_X, axis=0)
            y_new = np.concatenate(nn_y, axis=0)

            X_combined = np.concatenate([X_combined, X_new], axis=0)
            y_combined = np.concatenate([y_combined, y_new], axis=0)

            check_distribution(y_combined)

            dataset_path = os.path.join(datasets_dir, f"dataset_iter{iteration}.npz")
            np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

            model_path = os.path.join(models_dir, f"model_iter{iteration}.pt")

            try:
                results = train_dual(
                    dataset_path=dataset_path,
                    epochs=20,
                    batch_size=512,
                    lr=5e-4,
                    out_path=model_path,
                    use_residual=False,
                    hidden_dim=256,
                    augment=False,  # Skip for large datasets
                    early_stopping=True,
                    patience=5,
                    verbose=False
                )

                best_model_path = model_path
                log(f"  Model updated: val_loss={results['best_val_loss']:.5f}")

            except Exception as e:
                log(f"  Training error: {e}")

            nn_X = []
            nn_y = []

    # ===== PHASE 4: NN vs Minimax (harder opponent) =====
    phase4_end = start_time + (duration_minutes * 60 * 0.90)

    log("PHASE 4: NN vs Minimax (challenging the model)")

    while time.time() < phase4_end and time_remaining() > 180:
        iteration += 1
        games_this_iter = 30

        if iteration % 2 == 0:
            eng_w, eng_b = "nn", "minimax"
        else:
            eng_w, eng_b = "minimax", "nn"

        log(f"Iteration {iteration}: {eng_w} vs {eng_b}...")

        try:
            pgns, X, y = parallel_selfplay(
                n_games=games_this_iter,
                engine_w=eng_w,
                engine_b=eng_b,
                depth_w=2,
                depth_b=2,
                model_path=best_model_path,
                n_workers=n_workers,
                verbose=False
            )

            nn_X.append(X)
            nn_y.append(y)
            total_games += len(pgns)
            total_positions += len(X)

            log(f"  Generated {len(pgns)} games, {len(X)} positions")

        except Exception as e:
            log(f"  Error: {e}")
            continue

        # Retrain every 3 iterations
        if len(nn_X) >= 3 and time_remaining() > 300:
            log("  Retraining...")

            X_new = np.concatenate(nn_X, axis=0)
            y_new = np.concatenate(nn_y, axis=0)

            X_combined = np.concatenate([X_combined, X_new], axis=0)
            y_combined = np.concatenate([y_combined, y_new], axis=0)

            dataset_path = os.path.join(datasets_dir, f"dataset_iter{iteration}.npz")
            np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

            model_path = os.path.join(models_dir, f"model_iter{iteration}.pt")

            try:
                results = train_dual(
                    dataset_path=dataset_path,
                    epochs=20,
                    batch_size=512,
                    lr=3e-4,
                    out_path=model_path,
                    use_residual=False,
                    hidden_dim=256,
                    augment=False,  # Skip for large datasets
                    early_stopping=True,
                    patience=5,
                    verbose=False
                )

                best_model_path = model_path
                log(f"  Model updated: val_loss={results['best_val_loss']:.5f}")

            except Exception as e:
                log(f"  Training error: {e}")

            nn_X = []
            nn_y = []

    # ===== PHASE 5: Final Training =====
    log("PHASE 5: Final intensive training")

    if nn_X:
        X_new = np.concatenate(nn_X, axis=0)
        y_new = np.concatenate(nn_y, axis=0)
        X_combined = np.concatenate([X_combined, X_new], axis=0)
        y_combined = np.concatenate([y_combined, y_new], axis=0)

    if len(X_combined) > 0:
        dataset_path = os.path.join(datasets_dir, "dataset_final.npz")
        np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

        check_distribution(y_combined)

        log(f"Final training on {len(X_combined)} positions...")

        final_model_path = os.path.join(models_dir, "model_final.pt")

        try:
            results = train_dual(
                dataset_path=dataset_path,
                epochs=50,
                batch_size=512,
                lr=1e-4,
                out_path=final_model_path,
                use_residual=False,
                hidden_dim=256,
                augment=False,  # Skip for large datasets
                early_stopping=True,
                patience=15,
                verbose=verbose
            )

            best_model_path = final_model_path
            log(f"Final model: val_loss={results['best_val_loss']:.5f}")

        except Exception as e:
            log(f"Final training error: {e}")

    # Save final model
    if best_model_path and os.path.exists(best_model_path):
        final_path = "value_model_intensive_v2.pt"
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
    print(f"  python main.py --load_model value_model_intensive_v2.pt")
    print("=" * 70)

    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Intensive training v2")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--output", type=str, default="data/intensive_training_v2")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    run_intensive_training(
        duration_minutes=args.duration,
        output_dir=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

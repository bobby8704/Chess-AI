#!/usr/bin/env python3
"""
Overnight Training System for Chess AI - Optimized for Speed and Wins

Goals:
1. Train model to WIN (not just evaluate)
2. Handle large datasets without memory issues
3. Use MCTS for better exploration
4. Generate decisive games (wins/losses, minimize draws)
5. Run continuously overnight with parallel processing

Key Improvements:
- Uses parallel_selfplay for fast game generation
- Weighted sampling towards decisive positions
- Memory-efficient chunked processing
- Progressive difficulty: random -> minimax -> NN -> MCTS
"""

import argparse
import gc
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def clear_memory():
    """Force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def weighted_sample_decisive(X, y, target_size, decisive_weight=5.0):
    """
    Sample from dataset with higher weight on decisive positions.
    Positions from wins/losses are 5x more likely to be sampled.
    """
    n = len(X)
    if target_size >= n:
        return X, y

    weights = np.ones(n)
    decisive_mask = np.abs(y) > 0.5
    weights[decisive_mask] = decisive_weight
    weights /= weights.sum()

    indices = np.random.choice(n, size=target_size, replace=False, p=weights)
    return X[indices], y[indices]


def generate_decisive_games_parallel(n_games, engine_w, engine_b, depth_w, depth_b,
                                     model_path=None, n_workers=8):
    """Generate games in parallel using existing infrastructure."""
    from performance import parallel_selfplay

    pgns, X, y = parallel_selfplay(
        n_games=n_games,
        engine_w=engine_w,
        engine_b=engine_b,
        depth_w=depth_w,
        depth_b=depth_b,
        model_path=model_path,
        n_workers=n_workers,
        verbose=False
    )

    # Calculate stats
    wins = (y > 0.5).sum()
    losses = (y < -0.5).sum()
    draws = len(y) - wins - losses

    return X, y, {"wins": int(wins), "losses": int(losses), "draws": int(draws), "games": len(pgns)}


def train_model_efficient(dataset_path, model_path, epochs=30, batch_size=512, lr=1e-3, verbose=True):
    """Memory-efficient training."""
    from ai_nn import ValueNet

    data = np.load(dataset_path)
    X = data["X"]
    y = data["y"]
    n = len(X)

    if verbose:
        log(f"  Training on {n} positions...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(input_dim=781).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = torch.nn.MSELoss()

    indices = np.random.permutation(n)
    val_size = min(10000, int(n * 0.1))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        np.random.shuffle(train_idx)

        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i+batch_size]
            xb = torch.from_numpy(X[batch_idx]).float().to(device)
            yb = torch.from_numpy(y[batch_idx]).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_idx)

        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i:i+batch_size]
                xb = torch.from_numpy(X[batch_idx]).float().to(device)
                yb = torch.from_numpy(y[batch_idx]).float().unsqueeze(1).to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(batch_idx)
        val_loss /= len(val_idx)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            log(f"    Epoch {epoch+1}: train={train_loss:.5f}, val={val_loss:.5f}")

        if patience_counter >= patience:
            if verbose:
                log(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    clear_memory()

    return {"val_loss": best_val_loss, "epochs": epoch + 1}


def run_overnight_training(duration_hours: float = 8.0, output_dir: str = "data/overnight"):
    """
    Run overnight training for specified duration.

    Training Strategy:
    1. Phase 1 (20%): Random games - lots of decisive outcomes
    2. Phase 2 (20%): Asymmetric games - strong vs weak for clear wins
    3. Phase 3 (40%): NN self-play - learning from itself
    4. Phase 4 (20%): Final training on all data weighted towards wins/losses
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    datasets_dir = os.path.join(output_dir, "datasets")
    Path(models_dir).mkdir(exist_ok=True)
    Path(datasets_dir).mkdir(exist_ok=True)

    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)

    MAX_POSITIONS = 150000  # Memory limit

    log("=" * 70)
    log("OVERNIGHT CHESS AI TRAINING")
    log("=" * 70)
    log(f"Duration: {duration_hours} hours")
    log(f"End time: {datetime.now() + timedelta(hours=duration_hours)}")
    log(f"Output: {output_dir}")
    log("=" * 70)

    iteration = 0
    best_model_path = None
    total_games = 0
    total_positions = 0

    def time_remaining():
        return max(0, end_time - time.time())

    def hours_elapsed():
        return (time.time() - start_time) / 3600

    # ===== PHASE 1: Random Games (Fast, Decisive) =====
    log("\n" + "=" * 50)
    log("PHASE 1: Random Games (Quick Decisive Outcomes)")
    log("=" * 50)

    phase1_end = start_time + (duration_hours * 3600 * 0.15)
    all_X = []
    all_y = []

    while time.time() < phase1_end and time_remaining() > 300:
        iteration += 1
        log(f"\nIteration {iteration}: Random games...")

        X, y, stats = generate_decisive_games_parallel(
            n_games=200, engine_w="random", engine_b="random",
            depth_w=1, depth_b=1, n_workers=10
        )

        # Limit random game positions (they can be very long)
        if len(X) > 30000:
            X, y = weighted_sample_decisive(X, y, 30000)

        all_X.append(X)
        all_y.append(y)
        total_games += stats["games"]
        total_positions += len(X)

        log(f"  W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} | Pos: {len(X)}")

        if total_positions > MAX_POSITIONS // 2:
            break

    # ===== PHASE 2: Asymmetric Games (Strong vs Weak) =====
    log("\n" + "=" * 50)
    log("PHASE 2: Asymmetric Games (Strong vs Weak)")
    log("=" * 50)

    phase2_end = start_time + (duration_hours * 3600 * 0.30)

    while time.time() < phase2_end and time_remaining() > 300:
        iteration += 1

        # Alternate which side is strong
        if iteration % 2 == 0:
            eng_w, eng_b, d_w, d_b = "minimax", "random", 2, 1
        else:
            eng_w, eng_b, d_w, d_b = "random", "minimax", 1, 2

        log(f"\nIteration {iteration}: {eng_w}(d{d_w}) vs {eng_b}(d{d_b})...")

        X, y, stats = generate_decisive_games_parallel(
            n_games=100, engine_w=eng_w, engine_b=eng_b,
            depth_w=d_w, depth_b=d_b, n_workers=10
        )

        all_X.append(X)
        all_y.append(y)
        total_games += stats["games"]
        total_positions += len(X)

        log(f"  W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} | Total pos: {total_positions}")

    # Initial training
    if all_X:
        log("\n--- Initial Model Training ---")
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Weighted sample towards decisive
        if len(X_combined) > MAX_POSITIONS:
            X_combined, y_combined = weighted_sample_decisive(X_combined, y_combined, MAX_POSITIONS)

        # Stats
        wins = (y_combined > 0.5).sum()
        losses = (y_combined < -0.5).sum()
        draws = len(y_combined) - wins - losses
        log(f"Data: {len(y_combined)} pos | W:{wins} ({100*wins/len(y_combined):.1f}%) L:{losses} ({100*losses/len(y_combined):.1f}%) D:{draws} ({100*draws/len(y_combined):.1f}%)")

        dataset_path = os.path.join(datasets_dir, "dataset_phase2.npz")
        np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

        model_path = os.path.join(models_dir, "model_phase2.pt")
        results = train_model_efficient(dataset_path, model_path, epochs=50, verbose=True)
        best_model_path = model_path
        log(f"Model trained: val_loss={results['val_loss']:.5f}")

        del X_combined, y_combined, all_X, all_y
        clear_memory()

    # ===== PHASE 3: NN Self-play =====
    log("\n" + "=" * 50)
    log("PHASE 3: NN Self-play (Learning from Itself)")
    log("=" * 50)

    phase3_end = start_time + (duration_hours * 3600 * 0.85)
    nn_iteration = 0
    accumulated_X = []
    accumulated_y = []

    while time.time() < phase3_end and time_remaining() > 600:
        iteration += 1
        nn_iteration += 1

        # Mix of opponents
        games_config = [
            ("nn", "random", 2, 1, "NN vs Random"),
            ("random", "nn", 1, 2, "Random vs NN"),
            ("nn", "nn", 2, 2, "NN vs NN"),
            ("nn", "minimax", 2, 2, "NN vs Minimax"),
        ]

        config = games_config[nn_iteration % len(games_config)]
        log(f"\nIteration {iteration}: {config[4]}...")

        X, y, stats = generate_decisive_games_parallel(
            n_games=50, engine_w=config[0], engine_b=config[1],
            depth_w=config[2], depth_b=config[3],
            model_path=best_model_path, n_workers=8
        )

        accumulated_X.append(X)
        accumulated_y.append(y)
        total_games += stats["games"]
        total_positions += len(X)

        log(f"  W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} | Total: {total_positions}")

        # Retrain every 5 iterations
        if nn_iteration % 5 == 0 and accumulated_X:
            log("  Retraining model...")

            X_new = np.concatenate(accumulated_X, axis=0)
            y_new = np.concatenate(accumulated_y, axis=0)

            # Weighted sample
            if len(X_new) > 50000:
                X_new, y_new = weighted_sample_decisive(X_new, y_new, 50000)

            dataset_path = os.path.join(datasets_dir, f"dataset_iter{iteration}.npz")
            np.savez_compressed(dataset_path, X=X_new, y=y_new)

            model_path = os.path.join(models_dir, f"model_iter{iteration}.pt")
            results = train_model_efficient(dataset_path, model_path, epochs=30, verbose=False)
            best_model_path = model_path
            log(f"  Model updated: val_loss={results['val_loss']:.5f}")

            accumulated_X = []
            accumulated_y = []
            clear_memory()

    # ===== PHASE 4: Final Combined Training =====
    log("\n" + "=" * 50)
    log("PHASE 4: Final Combined Training")
    log("=" * 50)

    # Collect all datasets
    all_datasets = list(Path(datasets_dir).glob("*.npz"))
    log(f"Combining {len(all_datasets)} datasets...")

    final_X = []
    final_y = []

    for ds_path in all_datasets:
        try:
            data = np.load(str(ds_path))
            X = data["X"]
            y = data["y"]

            # Sample from each
            if len(X) > 30000:
                X, y = weighted_sample_decisive(X, y, 30000)

            final_X.append(X)
            final_y.append(y)
        except Exception as e:
            log(f"  Error loading {ds_path}: {e}")
            continue

    if final_X:
        X_final = np.concatenate(final_X, axis=0)
        y_final = np.concatenate(final_y, axis=0)

        # Final weighted sample
        if len(X_final) > MAX_POSITIONS:
            X_final, y_final = weighted_sample_decisive(X_final, y_final, MAX_POSITIONS)

        # Stats
        wins = (y_final > 0.5).sum()
        losses = (y_final < -0.5).sum()
        draws = len(y_final) - wins - losses
        log(f"Final data: {len(y_final)} positions")
        log(f"  W:{wins} ({100*wins/len(y_final):.1f}%) L:{losses} ({100*losses/len(y_final):.1f}%) D:{draws} ({100*draws/len(y_final):.1f}%)")

        final_dataset = os.path.join(datasets_dir, "dataset_final.npz")
        np.savez_compressed(final_dataset, X=X_final, y=y_final)

        log("Final model training (extended)...")
        final_model = os.path.join(models_dir, "model_final.pt")
        results = train_model_efficient(final_dataset, final_model, epochs=100, verbose=True)
        best_model_path = final_model
        log(f"Final model: val_loss={results['val_loss']:.5f}")

        del X_final, y_final, final_X, final_y
        clear_memory()

    # Save final model
    if best_model_path and os.path.exists(best_model_path):
        final_path = "value_model_overnight.pt"
        shutil.copy(best_model_path, final_path)
        log(f"\nFinal model saved to: {final_path}")

    # Summary
    total_time = time.time() - start_time
    log("\n" + "=" * 70)
    log("OVERNIGHT TRAINING COMPLETE")
    log("=" * 70)
    log(f"Total time: {total_time/3600:.2f} hours")
    log(f"Iterations: {iteration}")
    log(f"Total games: {total_games}")
    log(f"Total positions: {total_positions}")
    log(f"Final model: {best_model_path}")
    log("=" * 70)
    log("\nTo play against the AI:")
    log("  python main.py --load_model value_model_overnight.pt")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Overnight Chess AI Training")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Training duration in hours (default: 8)")
    parser.add_argument("--output", type=str, default="data/overnight",
                        help="Output directory")

    args = parser.parse_args()

    run_overnight_training(
        duration_hours=args.hours,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

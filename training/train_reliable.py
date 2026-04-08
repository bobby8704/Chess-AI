#!/usr/bin/env python3
"""
Reliable Training System for Chess AI - Windows Compatible

Avoids multiprocessing issues by using sequential game generation.
Still efficient through optimized single-threaded game play.
"""
print("STARTUP: Beginning imports...", flush=True)

import argparse
print("STARTUP: argparse imported", flush=True)
import gc
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
print("STARTUP: Standard library imports done", flush=True)

import numpy as np
print("STARTUP: numpy imported", flush=True)
import torch
print("STARTUP: torch imported", flush=True)
import chess
print("STARTUP: chess imported", flush=True)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("STARTUP: All imports complete", flush=True)


def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def clear_memory():
    """Force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_games_sequential(n_games, engine_w, engine_b, depth_w, depth_b, model=None, max_moves=200):
    """Generate games sequentially (more reliable on Windows)."""
    from features import board_to_tensor
    from ai_engines import get_engine

    all_X = []
    all_y = []
    wins = 0
    losses = 0
    draws = 0

    # Get model path if model is provided
    model_path = None
    if model is not None:
        # Save model temporarily in correct format for load_value_model
        model_path = "temp_model.pt"
        torch.save({
            "input_dim": 781,
            "state_dict": model.state_dict()
        }, model_path)

    for game_num in range(n_games):
        board = chess.Board()
        positions = []

        # Create engines
        white_engine = get_engine(engine_w, depth=depth_w, model_path=model_path)
        black_engine = get_engine(engine_b, depth=depth_b, model_path=model_path)

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            # Store position
            tensor = board_to_tensor(board)
            positions.append((tensor, board.turn))

            # Get move
            if board.turn == chess.WHITE:
                result = white_engine.select_move(board)
            else:
                result = black_engine.select_move(board)

            if result is None or result.move is None:
                break

            board.push(result.move)
            move_count += 1

        # Determine outcome
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                outcome = -1.0  # Black won
                losses += 1
            else:
                outcome = 1.0  # White won
                wins += 1
        else:
            outcome = 0.0
            draws += 1

        # Create training data
        for tensor, turn in positions:
            value = outcome if turn == chess.WHITE else -outcome
            all_X.append(tensor)
            all_y.append(value)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)

    return X, y, {"wins": wins, "losses": losses, "draws": draws}


def weighted_sample(X, y, target_size, decisive_weight=5.0):
    """Sample with higher weight on decisive positions."""
    n = len(X)
    if target_size >= n:
        return X, y

    weights = np.ones(n)
    decisive_mask = np.abs(y) > 0.5
    weights[decisive_mask] = decisive_weight
    weights /= weights.sum()

    indices = np.random.choice(n, size=target_size, replace=False, p=weights)
    return X[indices], y[indices]


def train_model(X, y, model_path, epochs=30, batch_size=512, lr=1e-3, verbose=True):
    """Train neural network model."""
    from ai_nn import ValueNet

    n = len(X)
    if verbose:
        log(f"  Training on {n} positions...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(input_dim=781).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = torch.nn.MSELoss()

    # Split data
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

        # Validation
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

    return model, {"val_loss": best_val_loss, "epochs": epoch + 1}


def run_training(duration_hours: float = 8.0, output_dir: str = "data/overnight"):
    """Run training for specified duration."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    datasets_dir = os.path.join(output_dir, "datasets")
    Path(models_dir).mkdir(exist_ok=True)
    Path(datasets_dir).mkdir(exist_ok=True)

    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)

    MAX_POSITIONS = 100000

    log("=" * 70)
    log("CHESS AI TRAINING - RELIABLE VERSION")
    log("=" * 70)
    log(f"Duration: {duration_hours} hours")
    log(f"End time: {datetime.now() + timedelta(hours=duration_hours)}")
    log(f"Output: {output_dir}")
    log("=" * 70)

    iteration = 0
    current_model = None
    total_games = 0
    total_positions = 0

    def time_remaining():
        return max(0, end_time - time.time())

    # Phase 1: Random games (quick, many wins/losses)
    log("\n" + "=" * 50)
    log("PHASE 1: Random Games")
    log("=" * 50)

    phase1_end = start_time + (duration_hours * 3600 * 0.15)
    all_X = []
    all_y = []

    while time.time() < phase1_end and time_remaining() > 60:
        iteration += 1
        log(f"\nIteration {iteration}: Random games...")

        X, y, stats = generate_games_sequential(
            n_games=50, engine_w="random", engine_b="random",
            depth_w=1, depth_b=1
        )

        all_X.append(X)
        all_y.append(y)
        total_games += stats["wins"] + stats["losses"] + stats["draws"]
        total_positions += len(X)

        log(f"  W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} | Pos: {len(X)} | Total: {total_positions}")

        if total_positions > MAX_POSITIONS // 3:
            break

    # Phase 2: Asymmetric games
    log("\n" + "=" * 50)
    log("PHASE 2: Asymmetric Games")
    log("=" * 50)

    phase2_end = start_time + (duration_hours * 3600 * 0.35)

    while time.time() < phase2_end and time_remaining() > 60:
        iteration += 1

        # Alternate sides
        if iteration % 2 == 0:
            eng_w, eng_b, d_w, d_b = "minimax", "random", 2, 1
        else:
            eng_w, eng_b, d_w, d_b = "random", "minimax", 1, 2

        log(f"\nIteration {iteration}: {eng_w}(d{d_w}) vs {eng_b}(d{d_b})...")

        X, y, stats = generate_games_sequential(
            n_games=30, engine_w=eng_w, engine_b=eng_b,
            depth_w=d_w, depth_b=d_b
        )

        all_X.append(X)
        all_y.append(y)
        total_games += stats["wins"] + stats["losses"] + stats["draws"]
        total_positions += len(X)

        log(f"  W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} | Total pos: {total_positions}")

    # Initial training
    if all_X:
        log("\n--- Initial Model Training ---")
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        if len(X_combined) > MAX_POSITIONS:
            X_combined, y_combined = weighted_sample(X_combined, y_combined, MAX_POSITIONS)

        # Stats
        wins = (y_combined > 0.5).sum()
        losses = (y_combined < -0.5).sum()
        draws = len(y_combined) - wins - losses
        log(f"Data: {len(y_combined)} pos | W:{wins} ({100*wins/len(y_combined):.1f}%) L:{losses} ({100*losses/len(y_combined):.1f}%) D:{draws} ({100*draws/len(y_combined):.1f}%)")

        dataset_path = os.path.join(datasets_dir, "dataset_phase2.npz")
        np.savez_compressed(dataset_path, X=X_combined, y=y_combined)

        model_path = os.path.join(models_dir, "model_phase2.pt")
        current_model, results = train_model(X_combined, y_combined, model_path, epochs=50, verbose=True)
        log(f"Model trained: val_loss={results['val_loss']:.5f}")

        del X_combined, y_combined, all_X, all_y
        clear_memory()
        all_X = []
        all_y = []

    # Phase 3: NN Self-play
    log("\n" + "=" * 50)
    log("PHASE 3: NN Self-play")
    log("=" * 50)

    phase3_end = start_time + (duration_hours * 3600 * 0.90)
    nn_iteration = 0

    while time.time() < phase3_end and time_remaining() > 120:
        iteration += 1
        nn_iteration += 1

        # Mix of opponents
        configs = [
            ("nn", "random", 2, 1, "NN vs Random"),
            ("random", "nn", 1, 2, "Random vs NN"),
            ("nn", "nn", 2, 2, "NN vs NN"),
            ("nn", "minimax", 2, 2, "NN vs Minimax"),
        ]
        config = configs[nn_iteration % len(configs)]
        log(f"\nIteration {iteration}: {config[4]}...")

        X, y, stats = generate_games_sequential(
            n_games=20, engine_w=config[0], engine_b=config[1],
            depth_w=config[2], depth_b=config[3], model=current_model
        )

        all_X.append(X)
        all_y.append(y)
        total_games += stats["wins"] + stats["losses"] + stats["draws"]
        total_positions += len(X)

        log(f"  W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} | Total: {total_positions}")

        # Retrain every 5 iterations
        if nn_iteration % 5 == 0 and all_X:
            log("  Retraining model...")

            X_new = np.concatenate(all_X, axis=0)
            y_new = np.concatenate(all_y, axis=0)

            if len(X_new) > 50000:
                X_new, y_new = weighted_sample(X_new, y_new, 50000)

            dataset_path = os.path.join(datasets_dir, f"dataset_iter{iteration}.npz")
            np.savez_compressed(dataset_path, X=X_new, y=y_new)

            model_path = os.path.join(models_dir, f"model_iter{iteration}.pt")
            current_model, results = train_model(X_new, y_new, model_path, epochs=30, verbose=False)
            log(f"  Model updated: val_loss={results['val_loss']:.5f}")

            # Keep only recent data
            if len(all_X) > 10:
                all_X = all_X[-5:]
                all_y = all_y[-5:]

            clear_memory()

    # Final training
    log("\n" + "=" * 50)
    log("PHASE 4: Final Training")
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

            if len(X) > 20000:
                X, y = weighted_sample(X, y, 20000)

            final_X.append(X)
            final_y.append(y)
        except Exception as e:
            log(f"  Error loading {ds_path}: {e}")

    if final_X:
        X_final = np.concatenate(final_X, axis=0)
        y_final = np.concatenate(final_y, axis=0)

        if len(X_final) > MAX_POSITIONS:
            X_final, y_final = weighted_sample(X_final, y_final, MAX_POSITIONS)

        # Stats
        wins = (y_final > 0.5).sum()
        losses = (y_final < -0.5).sum()
        draws = len(y_final) - wins - losses
        log(f"Final data: {len(y_final)} positions")
        log(f"  W:{wins} ({100*wins/len(y_final):.1f}%) L:{losses} ({100*losses/len(y_final):.1f}%) D:{draws} ({100*draws/len(y_final):.1f}%)")

        final_dataset = os.path.join(datasets_dir, "dataset_final.npz")
        np.savez_compressed(final_dataset, X=X_final, y=y_final)

        log("Final model training (extended)...")
        final_model_path = os.path.join(models_dir, "model_final.pt")
        current_model, results = train_model(X_final, y_final, final_model_path, epochs=100, verbose=True)
        log(f"Final model: val_loss={results['val_loss']:.5f}")

        # Copy to root
        import shutil
        final_path = "value_model_overnight.pt"
        shutil.copy(final_model_path, final_path)
        log(f"\nFinal model saved to: {final_path}")

    # Summary
    total_time = time.time() - start_time
    log("\n" + "=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)
    log(f"Total time: {total_time/3600:.2f} hours")
    log(f"Iterations: {iteration}")
    log(f"Total games: {total_games}")
    log(f"Total positions: {total_positions}")
    log("=" * 70)
    log("\nTo play against the AI:")
    log("  python main.py --load_model value_model_overnight.pt")
    log("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliable Chess AI Training")
    parser.add_argument("--hours", type=float, default=8.0, help="Training duration")
    parser.add_argument("--output", type=str, default="data/overnight", help="Output directory")

    args = parser.parse_args()

    run_training(duration_hours=args.hours, output_dir=args.output)

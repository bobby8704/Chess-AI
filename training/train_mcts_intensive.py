#!/usr/bin/env python3
"""
Intensive MCTS Training System for Chess AI

This script performs comprehensive training using:
1. All existing training data (npz files from previous sessions)
2. Human vs AI game data from the database (reinforcement learning)
3. New MCTS self-play games
4. Progressive training with increasing difficulty

Run: .venv\Scripts\python.exe train_mcts_intensive.py --hours 6
"""

import argparse
import gc
import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob

print("STARTUP: Beginning imports...", flush=True)

import numpy as np
print("STARTUP: numpy imported", flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
print("STARTUP: torch imported", flush=True)
import chess
print("STARTUP: chess imported", flush=True)

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


# ==================== Neural Network ====================

from neural_network import DualNet, POLICY_OUTPUT_SIZE, encode_move, get_legal_move_mask
from features import board_to_tensor


# ==================== Data Loading ====================

def load_existing_datasets(data_dirs, max_positions=100000, max_file_size_mb=500):
    """
    Load existing npz datasets with memory limits.

    Args:
        data_dirs: List of directories to search
        max_positions: Maximum total positions to load
        max_file_size_mb: Skip files larger than this (MB)
    """
    all_X = []
    all_y = []
    total_loaded = 0

    max_file_bytes = max_file_size_mb * 1024 * 1024

    for data_dir in data_dirs:
        if total_loaded >= max_positions:
            break

        npz_files = glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)
        for npz_path in npz_files:
            if total_loaded >= max_positions:
                break

            try:
                # Check file size first - skip huge files
                file_size = os.path.getsize(npz_path)
                if file_size > max_file_bytes:
                    log(f"  Skipping large file ({file_size/1024/1024:.0f}MB): {os.path.basename(npz_path)}")
                    continue

                data = np.load(npz_path)
                X = data["X"]
                y = data["y"]

                # Sample if too large
                remaining = max_positions - total_loaded
                if len(X) > remaining:
                    indices = np.random.choice(len(X), remaining, replace=False)
                    X = X[indices]
                    y = y[indices]

                all_X.append(X)
                all_y.append(y)
                total_loaded += len(X)
                log(f"  Loaded {len(X)} positions from {os.path.basename(npz_path)}")

            except Exception as e:
                log(f"  Warning: Could not load {npz_path}: {e}")

    if all_X:
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        return X_combined, y_combined, total_loaded
    return None, None, 0


def load_human_games(db_path):
    """
    Load training data from human vs AI games in the database.
    This is crucial for reinforcement learning - learning from games against humans.
    """
    if not os.path.exists(db_path):
        return None, None, 0

    positions = []
    values = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all completed games with human players
        cursor.execute("""
            SELECT g.id, g.result, g.white_type, g.black_type
            FROM games g
            WHERE g.result != '*'
        """)
        games = cursor.fetchall()

        for game_id, result, white_type, black_type in games:
            # Determine outcome from white's perspective
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0

            # Weight human wins/losses more heavily (valuable feedback)
            weight = 1.0
            if white_type == "human" or black_type == "human":
                weight = 3.0  # Human games are more valuable

            # Get moves for this game
            cursor.execute("""
                SELECT fen_before, fen_after
                FROM moves
                WHERE game_id = ?
                ORDER BY ply
            """, (game_id,))
            moves = cursor.fetchall()

            for fen_before, fen_after in moves:
                try:
                    board = chess.Board(fen_before)
                    tensor = board_to_tensor(board)

                    # Value from current player's perspective
                    value = outcome if board.turn == chess.WHITE else -outcome

                    positions.append(tensor)
                    values.append(value * weight)
                except:
                    continue

        conn.close()

        if positions:
            X = np.array(positions, dtype=np.float32)
            y = np.array(values, dtype=np.float32)
            # Clip values to [-1, 1] after weighting
            y = np.clip(y, -1.0, 1.0)
            return X, y, len(X)

    except Exception as e:
        log(f"  Warning: Could not load games database: {e}")

    return None, None, 0


# ==================== MCTS for Training ====================

class TrainingMCTS:
    """MCTS for self-play training data generation."""

    def __init__(self, model, device, num_simulations=100, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def evaluate(self, board):
        """Get policy and value from neural network."""
        if self.model is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            return {m: 1.0/len(legal_moves) for m in legal_moves}, 0.0

        x = torch.from_numpy(board_to_tensor(board)).float().to(self.device).unsqueeze(0)
        mask = get_legal_move_mask(board).to(self.device).unsqueeze(0)

        with torch.no_grad():
            policy_probs, value = self.model.get_policy_value(x, mask)

        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        value = value.item()

        move_probs = {}
        for move in board.legal_moves:
            idx = encode_move(move)
            if idx < POLICY_OUTPUT_SIZE:
                move_probs[move] = float(policy_probs[idx])

        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p/total for m, p in move_probs.items()}

        return move_probs, value

    def search(self, board, temperature=1.0):
        """Run MCTS and return (selected_move, policy_target)."""
        if board.is_game_over():
            return None, {}

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        visit_counts = {move: 0 for move in legal_moves}
        total_values = {move: 0.0 for move in legal_moves}

        prior_policy, _ = self.evaluate(board)

        # Add Dirichlet noise to root
        if self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                prior = prior_policy.get(move, 1.0/len(legal_moves))
                prior_policy[move] = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[i]

        for _ in range(self.num_simulations):
            total_visits = sum(visit_counts.values()) + 1

            best_score = float('-inf')
            best_move = legal_moves[0]

            for move in legal_moves:
                prior = prior_policy.get(move, 1.0/len(legal_moves))

                if visit_counts[move] == 0:
                    q_value = 0.0
                else:
                    q_value = total_values[move] / visit_counts[move]

                exploration = self.c_puct * prior * np.sqrt(total_visits) / (1 + visit_counts[move])
                score = q_value + exploration

                if score > best_score:
                    best_score = score
                    best_move = move

            sim_board = board.copy()
            sim_board.push(best_move)

            if sim_board.is_game_over():
                result = sim_board.result()
                if result == "1-0":
                    value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
            else:
                _, value = self.evaluate(sim_board)
                value = -value

            visit_counts[best_move] += 1
            total_values[best_move] += value

        total_visits = sum(visit_counts.values())
        policy_target = {move: count/total_visits for move, count in visit_counts.items()}

        if temperature == 0:
            selected_move = max(policy_target.items(), key=lambda x: x[1])[0]
        else:
            moves = list(policy_target.keys())
            probs = np.array([policy_target[m] for m in moves])
            probs = np.power(probs, 1.0/temperature)
            probs = probs / probs.sum()
            selected_move = moves[np.random.choice(len(moves), p=probs)]

        return selected_move, policy_target


def play_self_play_game(mcts, max_moves=200, temperature_threshold=30):
    """Play a self-play game and collect training data."""
    board = chess.Board()
    positions = []
    policy_targets = []
    turns = []

    move_num = 0
    while not board.is_game_over() and move_num < max_moves:
        temp = 1.0 if move_num < temperature_threshold else 0.1

        move, policy = mcts.search(board, temperature=temp)

        if move is None:
            break

        positions.append(board_to_tensor(board))
        policy_targets.append(policy)
        turns.append(board.turn)

        board.push(move)
        move_num += 1

    if board.is_checkmate():
        outcome = -1.0 if board.turn == chess.WHITE else 1.0
    else:
        outcome = 0.0

    return positions, policy_targets, turns, outcome


def create_training_batch(positions, policy_targets, turns, outcome):
    """Convert game data to training tensors."""
    n = len(positions)

    X = np.array(positions, dtype=np.float32)

    y_value = np.zeros(n, dtype=np.float32)
    for i, turn in enumerate(turns):
        y_value[i] = outcome if turn == chess.WHITE else -outcome

    y_policy = np.zeros((n, POLICY_OUTPUT_SIZE), dtype=np.float32)
    for i, policy in enumerate(policy_targets):
        for move, prob in policy.items():
            idx = encode_move(move)
            if idx < POLICY_OUTPUT_SIZE:
                y_policy[i, idx] = prob

    return X, y_policy, y_value


def train_epoch(model, optimizer, X, y_policy, y_value, device, batch_size=256):
    """Train for one epoch."""
    model.train()
    n = len(X)
    indices = np.random.permutation(n)

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    n_batches = 0

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i+batch_size]

        xb = torch.from_numpy(X[batch_idx]).float().to(device)

        # Handle case where y_policy might not exist (value-only training)
        has_policy = y_policy is not None and len(y_policy) > 0

        yv = torch.from_numpy(y_value[batch_idx]).float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        policy_logits, value_pred = model(xb)

        # Value loss
        value_loss = F.mse_loss(value_pred, yv)

        # Policy loss (if available)
        if has_policy:
            yp = torch.from_numpy(y_policy[batch_idx]).float().to(device)
            # Avoid log(0) by adding small epsilon
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(yp * log_probs).sum(dim=1).mean()
            loss = policy_loss + value_loss
            total_policy_loss += policy_loss.item()
        else:
            loss = value_loss
            total_policy_loss += 0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'policy_loss': total_policy_loss / n_batches,
        'value_loss': total_value_loss / n_batches
    }


def save_dual_model(model, path, input_dim=781, hidden_dim=512):
    """Save model in the format expected by MCTS player."""
    torch.save({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'model_type': 'DualNet',
        'state_dict': model.state_dict()
    }, path)


def run_intensive_training(duration_hours=6.0, output_dir="data/mcts_intensive"):
    """
    Run intensive MCTS training.

    Training phases:
    1. Pre-training on all existing data (first 20%)
    2. MCTS self-play with progressive difficulty (remaining 80%)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    datasets_dir = os.path.join(output_dir, "datasets")
    Path(models_dir).mkdir(exist_ok=True)
    Path(datasets_dir).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)

    log("=" * 70)
    log("INTENSIVE MCTS CHESS AI TRAINING")
    log("=" * 70)
    log(f"Duration: {duration_hours} hours")
    log(f"End time: {datetime.now() + timedelta(hours=duration_hours)}")
    log(f"Device: {device}")
    log(f"Output: {output_dir}")
    log("=" * 70)

    # Initialize model
    model = DualNet(input_dim=781, hidden_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(duration_hours * 60), eta_min=1e-5
    )

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== PHASE 1: Load and train on existing data ==========
    log("\n" + "=" * 50)
    log("PHASE 1: Pre-training on Existing Data")
    log("=" * 50)

    # Load existing datasets (skip old intensive training dirs - files too large)
    data_dirs = [
        "data/datasets",
        "data/overnight/datasets",
        "data/training_data",
        "data/simple_training",
        "data/mcts_training"
    ]

    log("\nLoading existing training datasets...")
    log("  (Skipping files > 500MB to avoid memory issues)")
    X_existing, y_existing, n_existing = load_existing_datasets(
        data_dirs, max_positions=100000, max_file_size_mb=500
    )
    if X_existing is not None:
        log(f"  Loaded {n_existing:,} positions from existing datasets")

    # Load human game data
    log("\nLoading human vs AI game data...")
    X_human, y_human, n_human = load_human_games("data/games.db")
    if X_human is not None:
        log(f"  Loaded {n_human:,} positions from human games")

    # Combine all existing data
    all_X = []
    all_y_value = []

    if X_existing is not None:
        all_X.append(X_existing)
        all_y_value.append(y_existing)

    if X_human is not None:
        all_X.append(X_human)
        all_y_value.append(y_human)

    if all_X:
        X_pretrain = np.concatenate(all_X, axis=0)
        y_pretrain = np.concatenate(all_y_value, axis=0)

        # Limit size to avoid memory issues
        max_pretrain = 200000
        if len(X_pretrain) > max_pretrain:
            # Sample with preference for decisive positions
            weights = np.ones(len(X_pretrain))
            decisive = np.abs(y_pretrain) > 0.5
            weights[decisive] = 3.0
            weights /= weights.sum()
            indices = np.random.choice(len(X_pretrain), max_pretrain, replace=False, p=weights)
            X_pretrain = X_pretrain[indices]
            y_pretrain = y_pretrain[indices]

        log(f"\nPre-training on {len(X_pretrain):,} positions...")

        # Pre-train for several epochs (value only, no policy targets from old data)
        pretrain_epochs = 20
        for epoch in range(pretrain_epochs):
            losses = train_epoch(model, optimizer, X_pretrain, None, y_pretrain, device)
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                log(f"  Epoch {epoch+1}/{pretrain_epochs}: loss={losses['value_loss']:.5f}")

        # Save pre-trained model
        pretrain_path = os.path.join(models_dir, "model_pretrained.pt")
        save_dual_model(model, pretrain_path)
        log(f"Pre-trained model saved: {pretrain_path}")

        del X_pretrain, y_pretrain, all_X, all_y_value
        clear_memory()
    else:
        log("No existing data found, starting from scratch")

    # ========== PHASE 2: MCTS Self-play Training ==========
    log("\n" + "=" * 50)
    log("PHASE 2: MCTS Self-play Training")
    log("=" * 50)

    # Progressive MCTS simulations
    sim_schedule = [
        (0.3, 30),   # First 30%: 30 simulations (faster)
        (0.6, 50),   # Next 30%: 50 simulations
        (0.9, 75),   # Next 30%: 75 simulations
        (1.0, 100),  # Final 10%: 100 simulations
    ]

    iteration = 0
    total_games = 0
    total_positions = 0
    games_per_batch = 5

    # Training data buffer
    buffer_X = []
    buffer_policy = []
    buffer_value = []
    max_buffer = 100000

    while time.time() < end_time:
        iteration += 1
        elapsed_fraction = (time.time() - start_time) / (duration_hours * 3600)

        # Determine current simulation count
        num_sims = 30
        for threshold, sims in sim_schedule:
            if elapsed_fraction <= threshold:
                num_sims = sims
                break

        # Initialize MCTS with current model
        mcts = TrainingMCTS(
            model=model,
            device=device,
            num_simulations=num_sims,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25
        )

        log(f"\nIteration {iteration} (sims={num_sims}): Playing {games_per_batch} games...")

        batch_wins = 0
        batch_losses = 0
        batch_draws = 0
        batch_positions = 0

        for game_num in range(games_per_batch):
            positions, policies, turns, outcome = play_self_play_game(
                mcts, max_moves=150, temperature_threshold=20
            )

            if positions:
                X, y_policy, y_value = create_training_batch(positions, policies, turns, outcome)
                buffer_X.append(X)
                buffer_policy.append(y_policy)
                buffer_value.append(y_value)

                batch_positions += len(positions)
                total_positions += len(positions)

                if outcome > 0:
                    batch_wins += 1
                elif outcome < 0:
                    batch_losses += 1
                else:
                    batch_draws += 1

            total_games += 1

        log(f"  Games: W:{batch_wins} L:{batch_losses} D:{batch_draws} | Positions: {batch_positions}")

        # Train on buffer
        if buffer_X:
            X_train = np.concatenate(buffer_X, axis=0)
            policy_train = np.concatenate(buffer_policy, axis=0)
            value_train = np.concatenate(buffer_value, axis=0)

            # Limit training batch
            max_train = 50000
            if len(X_train) > max_train:
                indices = np.random.choice(len(X_train), max_train, replace=False)
                X_train = X_train[indices]
                policy_train = policy_train[indices]
                value_train = value_train[indices]

            log(f"  Training on {len(X_train)} positions...")

            for epoch in range(3):
                losses = train_epoch(model, optimizer, X_train, policy_train, value_train, device)
                scheduler.step()

            log(f"  Loss: total={losses['total_loss']:.4f}, policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}")

            # Manage buffer size
            total_buffered = sum(len(x) for x in buffer_X)
            while total_buffered > max_buffer and len(buffer_X) > 5:
                removed = len(buffer_X[0])
                buffer_X.pop(0)
                buffer_policy.pop(0)
                buffer_value.pop(0)
                total_buffered -= removed

        # Save checkpoint every 10 iterations
        if iteration % 10 == 0:
            checkpoint_path = os.path.join(models_dir, f"model_iter{iteration}.pt")
            save_dual_model(model, checkpoint_path)

            # Save dataset
            if buffer_X:
                dataset_path = os.path.join(datasets_dir, f"dataset_iter{iteration}.npz")
                X_save = np.concatenate(buffer_X[-10:], axis=0) if len(buffer_X) > 10 else np.concatenate(buffer_X, axis=0)
                policy_save = np.concatenate(buffer_policy[-10:], axis=0) if len(buffer_policy) > 10 else np.concatenate(buffer_policy, axis=0)
                value_save = np.concatenate(buffer_value[-10:], axis=0) if len(buffer_value) > 10 else np.concatenate(buffer_value, axis=0)
                np.savez_compressed(dataset_path, X=X_save, policy=policy_save, y=value_save)

            log(f"  Checkpoint saved")

        clear_memory()

    # ========== Final Model ==========
    log("\n" + "=" * 50)
    log("FINAL TRAINING")
    log("=" * 50)

    # Final training on all buffered data
    if buffer_X:
        X_final = np.concatenate(buffer_X, axis=0)
        policy_final = np.concatenate(buffer_policy, axis=0)
        value_final = np.concatenate(buffer_value, axis=0)

        log(f"Final training on {len(X_final):,} positions...")

        for epoch in range(10):
            losses = train_epoch(model, optimizer, X_final, policy_final, value_final, device)
            if (epoch + 1) % 5 == 0:
                log(f"  Epoch {epoch+1}/10: loss={losses['total_loss']:.5f}")

    # Save final model
    final_path = "dual_model_mcts.pt"
    save_dual_model(model, final_path)
    log(f"\nFinal model saved: {final_path}")

    # Also save to output directory
    final_path2 = os.path.join(models_dir, "model_final.pt")
    save_dual_model(model, final_path2)

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
    log("  .venv\\Scripts\\python.exe main.py --engine mcts --load_model dual_model_mcts.pt --sims 100")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Intensive MCTS Chess AI Training")
    parser.add_argument("--hours", type=float, default=6.0, help="Training duration in hours")
    parser.add_argument("--output", type=str, default="data/mcts_intensive", help="Output directory")

    args = parser.parse_args()

    run_intensive_training(
        duration_hours=args.hours,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

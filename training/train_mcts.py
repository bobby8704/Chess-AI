#!/usr/bin/env python3
"""
MCTS-Based Training System for Chess AI

This script trains a DualNet (policy + value) using MCTS self-play,
similar to the AlphaZero approach.

Key features:
- DualNet with policy and value heads
- MCTS-guided move selection during self-play
- Policy targets from MCTS visit counts
- Value targets from game outcomes
- Progressive training with increasing MCTS simulations
"""

import argparse
import gc
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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


# ==================== MCTS for Training ====================

class TrainingMCTS:
    """Simplified MCTS for training data generation."""

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
            # Uniform policy, zero value
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

        # Normalize
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p/total for m, p in move_probs.items()}

        return move_probs, value

    def search(self, board, temperature=1.0):
        """
        Run MCTS and return (selected_move, policy_target).

        policy_target is a dict of move -> visit_probability
        """
        if board.is_game_over():
            return None, {}

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        # Initialize visit counts and values
        visit_counts = {move: 0 for move in legal_moves}
        total_values = {move: 0.0 for move in legal_moves}

        # Get prior policy from network
        prior_policy, _ = self.evaluate(board)

        # Add Dirichlet noise to root
        if self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                prior = prior_policy.get(move, 1.0/len(legal_moves))
                prior_policy[move] = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[i]

        # Run simulations
        for _ in range(self.num_simulations):
            # Select move using PUCT
            total_visits = sum(visit_counts.values()) + 1

            best_score = float('-inf')
            best_move = legal_moves[0]

            for move in legal_moves:
                prior = prior_policy.get(move, 1.0/len(legal_moves))

                if visit_counts[move] == 0:
                    q_value = 0.0
                else:
                    q_value = total_values[move] / visit_counts[move]

                # PUCT formula
                exploration = self.c_puct * prior * np.sqrt(total_visits) / (1 + visit_counts[move])
                score = q_value + exploration

                if score > best_score:
                    best_score = score
                    best_move = move

            # Simulate: make move and evaluate
            sim_board = board.copy()
            sim_board.push(best_move)

            # Get value (from opponent's perspective, so negate)
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
                value = -value  # Opponent's perspective

            # Update statistics
            visit_counts[best_move] += 1
            total_values[best_move] += value

        # Calculate policy target from visit counts
        total_visits = sum(visit_counts.values())
        policy_target = {move: count/total_visits for move, count in visit_counts.items()}

        # Select move based on temperature
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
    """
    Play a self-play game and collect training data.

    Returns:
        positions: List of board tensors
        policy_targets: List of policy target dicts
        outcome: Game outcome (1.0 white wins, -1.0 black wins, 0.0 draw)
    """
    board = chess.Board()
    positions = []
    policy_targets = []
    turns = []

    move_num = 0
    while not board.is_game_over() and move_num < max_moves:
        # Temperature: high early, low later
        temp = 1.0 if move_num < temperature_threshold else 0.1

        move, policy = mcts.search(board, temperature=temp)

        if move is None:
            break

        # Store training data
        positions.append(board_to_tensor(board))
        policy_targets.append(policy)
        turns.append(board.turn)

        board.push(move)
        move_num += 1

    # Determine outcome
    if board.is_checkmate():
        outcome = -1.0 if board.turn == chess.WHITE else 1.0  # Winner is opposite of current turn
    else:
        outcome = 0.0

    return positions, policy_targets, turns, outcome


def create_training_batch(positions, policy_targets, turns, outcome):
    """Convert game data to training tensors."""
    n = len(positions)

    X = np.array(positions, dtype=np.float32)

    # Value targets: outcome from each position's perspective
    y_value = np.zeros(n, dtype=np.float32)
    for i, turn in enumerate(turns):
        y_value[i] = outcome if turn == chess.WHITE else -outcome

    # Policy targets: convert to tensor format
    y_policy = np.zeros((n, POLICY_OUTPUT_SIZE), dtype=np.float32)
    for i, policy in enumerate(policy_targets):
        for move, prob in policy.items():
            idx = encode_move(move)
            if idx < POLICY_OUTPUT_SIZE:
                y_policy[i, idx] = prob

    return X, y_policy, y_value


def train_on_batch(model, optimizer, X, y_policy, y_value, device):
    """Train on a batch of data."""
    model.train()

    X = torch.from_numpy(X).float().to(device)
    y_policy = torch.from_numpy(y_policy).float().to(device)
    y_value = torch.from_numpy(y_value).float().to(device).unsqueeze(1)

    optimizer.zero_grad()

    policy_logits, value_pred = model(X)

    # Policy loss: cross-entropy with soft targets
    policy_loss = -(y_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, y_value)

    # Combined loss
    loss = policy_loss + value_loss

    loss.backward()
    optimizer.step()

    return {
        'total_loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item()
    }


def save_model(model, path, input_dim=781, hidden_dim=512):
    """Save model in the format expected by load_dual_model."""
    torch.save({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'model_type': 'DualNet',
        'state_dict': model.state_dict()
    }, path)


def run_mcts_training(duration_hours=1.0, output_dir="data/mcts_training",
                      num_simulations=50, games_per_batch=10):
    """
    Run MCTS-based training.

    Args:
        duration_hours: Training duration
        output_dir: Where to save models and data
        num_simulations: MCTS simulations per move
        games_per_batch: Games to play before training update
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    Path(models_dir).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)

    log("=" * 70)
    log("MCTS CHESS AI TRAINING")
    log("=" * 70)
    log(f"Duration: {duration_hours} hours")
    log(f"End time: {datetime.now() + timedelta(hours=duration_hours)}")
    log(f"MCTS simulations: {num_simulations}")
    log(f"Games per batch: {games_per_batch}")
    log("=" * 70)

    # Initialize model
    model = DualNet(input_dim=781, hidden_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize MCTS
    mcts = TrainingMCTS(
        model=model,
        device=device,
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    iteration = 0
    total_games = 0
    total_positions = 0

    all_X = []
    all_policy = []
    all_value = []

    while time.time() < end_time:
        iteration += 1

        # Play self-play games
        batch_positions = 0
        batch_wins = 0
        batch_losses = 0
        batch_draws = 0

        log(f"\nIteration {iteration}: Playing {games_per_batch} games...")

        for game_num in range(games_per_batch):
            positions, policies, turns, outcome = play_self_play_game(
                mcts, max_moves=150, temperature_threshold=20
            )

            if positions:
                X, y_policy, y_value = create_training_batch(positions, policies, turns, outcome)
                all_X.append(X)
                all_policy.append(y_policy)
                all_value.append(y_value)

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

        # Train on accumulated data
        if all_X:
            X_train = np.concatenate(all_X, axis=0)
            policy_train = np.concatenate(all_policy, axis=0)
            value_train = np.concatenate(all_value, axis=0)

            # Limit training data size
            max_train = 50000
            if len(X_train) > max_train:
                indices = np.random.choice(len(X_train), max_train, replace=False)
                X_train = X_train[indices]
                policy_train = policy_train[indices]
                value_train = value_train[indices]

            # Train for a few epochs
            log(f"  Training on {len(X_train)} positions...")

            batch_size = 256
            n_epochs = 3

            for epoch in range(n_epochs):
                indices = np.random.permutation(len(X_train))
                total_loss = 0
                n_batches = 0

                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    losses = train_on_batch(
                        model, optimizer,
                        X_train[batch_idx],
                        policy_train[batch_idx],
                        value_train[batch_idx],
                        device
                    )
                    total_loss += losses['total_loss']
                    n_batches += 1

                scheduler.step()

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            log(f"  Loss: {avg_loss:.4f}")

            # Keep only recent data
            max_buffer = 100000
            total_buffered = sum(len(x) for x in all_X)
            while total_buffered > max_buffer and len(all_X) > 5:
                removed = len(all_X[0])
                all_X.pop(0)
                all_policy.pop(0)
                all_value.pop(0)
                total_buffered -= removed

        # Save checkpoint periodically
        if iteration % 10 == 0:
            checkpoint_path = os.path.join(models_dir, f"model_iter{iteration}.pt")
            save_model(model, checkpoint_path)
            log(f"  Checkpoint saved: {checkpoint_path}")

        # Update MCTS with new model (already using the same object)
        clear_memory()

    # Final save
    final_path = "dual_model_mcts.pt"
    save_model(model, final_path)
    log(f"\nFinal model saved: {final_path}")

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
    log("  python main.py --engine mcts --load_model dual_model_mcts.pt")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MCTS Chess AI Training")
    parser.add_argument("--hours", type=float, default=1.0, help="Training duration")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=10, help="Games per training batch")
    parser.add_argument("--output", type=str, default="data/mcts_training", help="Output directory")

    args = parser.parse_args()

    run_mcts_training(
        duration_hours=args.hours,
        output_dir=args.output,
        num_simulations=args.sims,
        games_per_batch=args.games
    )


if __name__ == "__main__":
    main()

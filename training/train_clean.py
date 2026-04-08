#!/usr/bin/env python3
"""
Clean MCTS Training System for Chess AI

This script implements all 5 data hygiene recommendations:
1. Fresh start with clean data
2. Data size limits (max 50,000 positions in memory)
3. Rolling window (keeps only recent data)
4. Data validation (validates game states before training)
5. Memory-efficient training

Usage:
    .venv\\Scripts\\python.exe train_clean.py --hours 2
"""

import argparse
import gc
import os
import sys
import time
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

print("STARTUP: Beginning imports...", flush=True)

import numpy as np
print("STARTUP: numpy imported", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
print("STARTUP: torch imported", flush=True)
import chess
print("STARTUP: chess imported", flush=True)

sys.stdout.reconfigure(line_buffering=True)
print("STARTUP: All imports complete", flush=True)

# ==================== Configuration ====================

# Memory limits - increased for longer training sessions
MAX_POSITIONS_IN_MEMORY = 100000  # Increased for longer training
MAX_DATASET_FILES = 20  # Keep more files for better data diversity
POSITIONS_PER_FILE = 5000  # Save after this many positions
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# MCTS quality settings - higher quality training data
MCTS_SIMS = 30  # More sims = better move quality = better training
GAMES_PER_ITER = 10  # More games per iteration for faster learning
MAX_GAME_MOVES = 150  # Even longer games - must learn to convert advantage!
STRENGTH_TEST_INTERVAL = 10  # Test every 10 iterations for long sessions
TRAIN_EPOCHS = 3  # Balance between learning and not overfitting

# Training mix - balance self-play and random opponents
# Start with more random (easy wins) then reduce for self-play (harder)
RANDOM_OPPONENT_RATIO = 0.5  # 50% vs random, 50% self-play for adaptation
USE_MATERIAL_EVAL = True  # Evaluate incomplete games by material

# Anti-repetition: penalize positions that repeat
REPETITION_PENALTY = 0.4  # Strong penalty for repetitive moves

# Temperature for move selection during training
INITIAL_TEMPERATURE = 1.3  # High exploration early - try different openings!
FINAL_TEMPERATURE = 0.2  # Strong exploitation late - play best moves

# Supervised data mixing ratio
# This ensures we ALWAYS train on expert data to prevent forgetting
SUPERVISED_DATA_RATIO = 0.3  # 30% of each batch should be supervised data

# ==================== Crash Safety Settings ====================
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N iterations (was 10)
AUTO_SAVE_INTERVAL = 3  # Save main model every N iterations
LOG_FILE = "data/training/training_log.txt"  # Log progress to file for recovery

# ==================== Utilities ====================

# Global log file handle
_log_file = None

def init_log_file():
    """Initialize log file for crash recovery."""
    global _log_file
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a", encoding="utf-8")
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Training session started: {datetime.now().isoformat()}\n")
    _log_file.write(f"{'='*60}\n")
    _log_file.flush()

def log(msg):
    """Log with timestamp to console AND file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg, flush=True)

    # Also write to log file for crash recovery
    if _log_file is not None:
        _log_file.write(full_msg + "\n")
        _log_file.flush()  # Ensure it's written immediately


def evaluate_material(board):
    """
    Evaluate board position by material count.
    Returns value from White's perspective: positive = White ahead, negative = Black ahead.
    Scale to [-1, 1] range.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King doesn't count for material
    }

    white_material = 0
    black_material = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value

    # Material difference
    diff = white_material - black_material

    # Normalize to [-1, 1] range (max diff is about 39 points)
    # Use tanh-like scaling so big advantages saturate
    normalized = diff / 15.0  # 15 points = ~1.0
    normalized = max(-1.0, min(1.0, normalized))

    return normalized


def clear_memory():
    """Force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Data Validation ====================

def validate_position(board):
    """
    Validate that a chess position is legal.
    Returns True if valid, False if corrupted.
    """
    try:
        # Check basic validity
        if not board.is_valid():
            return False

        # Check both kings exist
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))
        if white_kings != 1 or black_kings != 1:
            return False

        # Check not both in check (impossible)
        # This is already handled by is_valid() but double-check

        return True
    except:
        return False


def validate_game_result(board, result):
    """
    Validate that a game result makes sense for the final position.
    """
    if result == "1-0":
        # White won - black should be checkmated or resigned
        # Can't fully verify resignation, but can check position is reasonable
        return True
    elif result == "0-1":
        # Black won
        return True
    elif result in ["1/2-1/2", "draw"]:
        # Draw - could be stalemate, repetition, 50-move, insufficient material
        return True
    elif result == "*":
        # Incomplete game - don't use for training
        return False
    return True


# ==================== Supervised Learning Data ====================

# Store supervised learning data globally so it's always mixed into training
SUPERVISED_DATA = None

def load_supervised_data():
    """
    Load the expert data from train_supervised.py.
    This data is CRITICAL - it provides the strong foundation that RL builds on.
    Returns (X, policy, values) arrays.
    """
    global SUPERVISED_DATA

    if SUPERVISED_DATA is not None:
        return SUPERVISED_DATA

    try:
        # Import from train_supervised (which now uses expanded data automatically)
        from train_supervised import load_all_expert_data

        # Load all expert data (games, openings, tactics, endgames)
        # This function automatically uses expanded data if available
        all_data = load_all_expert_data()

        if not all_data:
            log("Warning: No supervised data loaded!")
            return None

        # Convert to arrays
        X = np.array([d[0] for d in all_data], dtype=np.float32)
        policy = np.array([d[1] for d in all_data], dtype=np.float32)
        values = np.array([d[2] for d in all_data], dtype=np.float32)

        SUPERVISED_DATA = (X, policy, values)
        log(f"Total supervised data: {len(X)} positions")

        return SUPERVISED_DATA

    except Exception as e:
        log(f"Warning: Could not load supervised data: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== Human Game Data Loading ====================

def load_human_games(db_path, data_manager):
    """
    Load training data from human vs AI games in the database.
    These are valuable because they show real human strategies and AI weaknesses.
    """
    import sqlite3

    if not os.path.exists(db_path):
        log(f"No games database found at {db_path}")
        return 0

    positions_loaded = 0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all completed games (not abandoned)
        cursor.execute("""
            SELECT game_id, result FROM games
            WHERE result != '*' AND result IS NOT NULL
        """)
        games = cursor.fetchall()

        for game_id, result in games:
            # Determine outcome
            if result == "1-0":
                outcome = 1.0  # White won
            elif result == "0-1":
                outcome = -1.0  # Black won
            else:
                outcome = 0.0  # Draw

            # Get moves for this game
            cursor.execute("""
                SELECT fen_before FROM moves
                WHERE game_id = ?
                ORDER BY ply
            """, (game_id,))
            moves = cursor.fetchall()

            for (fen_before,) in moves:
                try:
                    board = chess.Board(fen_before)

                    # Validate the position
                    if not validate_position(board):
                        continue

                    tensor = board_to_tensor(board)

                    # Value from current player's perspective
                    value = outcome if board.turn == chess.WHITE else -outcome

                    # For human games, we don't have policy targets, so use uniform
                    policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        uniform_prob = 1.0 / len(legal_moves)
                        for move in legal_moves:
                            idx = encode_move(move)
                            if idx < POLICY_OUTPUT_SIZE:
                                policy_target[idx] = uniform_prob

                    data_manager.add_position(tensor, policy_target, value)
                    positions_loaded += 1

                except Exception as e:
                    continue

        conn.close()
        log(f"Loaded {positions_loaded} positions from {len(games)} human games")

    except Exception as e:
        log(f"Warning: Could not load human games: {e}")

    return positions_loaded


# ==================== Neural Network ====================

from neural_network import DualNet, POLICY_OUTPUT_SIZE, encode_move, get_legal_move_mask
from features import board_to_tensor


def load_or_create_model(model_path, device):
    """Load existing model or create new one."""
    model = DualNet(input_dim=781).to(device)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            log(f"Loaded existing model: {model_path}")
        except Exception as e:
            log(f"Could not load model, starting fresh: {e}")
    else:
        log("No existing model found, starting fresh")

    return model


def save_model(model, path):
    """Save model with metadata."""
    torch.save({
        "input_dim": 781,
        "state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat()
    }, path)


# ==================== MCTS for Training ====================

class TrainingMCTS:
    """Simplified MCTS for generating training data."""

    def __init__(self, model, device, num_simulations=8):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = 1.5
        self.dirichlet_alpha = 0.3
        self.dirichlet_weight = 0.25

    def get_policy_value(self, board):
        """Get policy and value from neural network."""
        if self.model is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            return {m: 1.0/len(legal_moves) for m in legal_moves}, 0.0

        self.model.eval()
        with torch.no_grad():
            tensor = board_to_tensor(board)
            x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask = get_legal_move_mask(board).to(self.device).unsqueeze(0)

            policy_logits, value = self.model(x)
            policy_logits = policy_logits.masked_fill(mask == 0, -1e9)
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0)

            policy_dict = {}
            for move in board.legal_moves:
                idx = encode_move(move)
                if idx < POLICY_OUTPUT_SIZE:
                    policy_dict[move] = policy_probs[idx].item()

            # Normalize
            total = sum(policy_dict.values())
            if total > 0:
                policy_dict = {m: p/total for m, p in policy_dict.items()}

            return policy_dict, value.item()

    def search(self, board, add_noise=True, temperature=1.0):
        """Run MCTS and return visit counts."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        # Get prior policy
        prior_policy, _ = self.get_policy_value(board)

        # Add Dirichlet noise for exploration
        if add_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                prior = prior_policy.get(move, 1.0/len(legal_moves))
                prior_policy[move] = (1 - self.dirichlet_weight) * prior + self.dirichlet_weight * noise[i]

        # Simple MCTS: run simulations
        visit_counts = {move: 0 for move in legal_moves}
        total_values = {move: 0.0 for move in legal_moves}

        for _ in range(self.num_simulations):
            # Select move using UCB
            total_visits = sum(visit_counts.values()) + 1
            best_move = legal_moves[0]
            best_ucb = -float('inf')

            for move in legal_moves:
                prior = prior_policy.get(move, 1.0/len(legal_moves))
                visits = visit_counts[move]

                if visits == 0:
                    ucb = float('inf')
                else:
                    q_value = total_values[move] / visits
                    ucb = q_value + self.c_puct * prior * math.sqrt(total_visits) / (1 + visits)

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_move = move

            # Simulate
            sim_board = board.copy()
            sim_board.push(best_move)

            # Get value (from opponent's perspective, so negate)
            if sim_board.is_game_over():
                outcome = sim_board.outcome()
                if outcome and outcome.winner is not None:
                    value = -1.0  # Current player lost
                else:
                    value = 0.0  # Draw
            else:
                _, value = self.get_policy_value(sim_board)
                value = -value  # Opponent's value, negate for current player

            visit_counts[best_move] += 1
            total_values[best_move] += value

        # Select move based on visit counts
        if sum(visit_counts.values()) == 0:
            return random.choice(legal_moves), visit_counts

        # Temperature-based selection (temperature passed as parameter)
        visits_array = np.array([visit_counts[m] for m in legal_moves], dtype=np.float64)

        if temperature <= 0.01:
            best_idx = np.argmax(visits_array)
        else:
            visits_temp = visits_array ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                probs = visits_temp / total
            else:
                probs = np.ones(len(legal_moves)) / len(legal_moves)
            best_idx = np.random.choice(len(legal_moves), p=probs)

        return legal_moves[best_idx], visit_counts


# ==================== Data Management ====================

class DataManager:
    """Manages training data with size limits and rolling window."""

    def __init__(self, data_dir, max_positions=MAX_POSITIONS_IN_MEMORY, max_files=MAX_DATASET_FILES):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.max_positions = max_positions
        self.max_files = max_files

        # In-memory buffers
        self.positions = []  # (tensor, policy_target, value_target)

        log(f"DataManager initialized: max_positions={max_positions}, max_files={max_files}")

    def add_position(self, tensor, policy_target, value_target):
        """Add a validated position to the buffer."""
        self.positions.append((tensor, policy_target, value_target))

        # Check if we need to save
        if len(self.positions) >= POSITIONS_PER_FILE:
            self.save_buffer()

    def save_buffer(self):
        """Save current buffer to disk and apply rolling window."""
        if not self.positions:
            return

        # Prepare data
        X = np.array([p[0] for p in self.positions], dtype=np.float32)
        policy = np.array([p[1] for p in self.positions], dtype=np.float32)
        values = np.array([p[2] for p in self.positions], dtype=np.float32)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.npz"
        filepath = self.data_dir / filename

        np.savez_compressed(filepath, X=X, policy=policy, values=values)
        log(f"Saved {len(self.positions)} positions to {filename}")

        # Clear buffer
        self.positions = []

        # Apply rolling window - delete old files
        self._apply_rolling_window()

    def _apply_rolling_window(self):
        """Keep only the most recent dataset files."""
        files = sorted(self.data_dir.glob("dataset_*.npz"))

        if len(files) > self.max_files:
            files_to_delete = files[:-self.max_files]
            for f in files_to_delete:
                f.unlink()
                log(f"Rolling window: deleted old file {f.name}")

    def load_training_data(self, include_supervised=True):
        """
        Load all data for training (with size limit).
        CRITICAL: Always include supervised data to prevent catastrophic forgetting!
        """
        all_X = []
        all_policy = []
        all_values = []
        total_loaded = 0

        # FIRST: Always include supervised data (expert games, tactics, endgames)
        # This prevents RL from "forgetting" the fundamentals
        if include_supervised:
            supervised = load_supervised_data()
            if supervised is not None:
                sup_X, sup_policy, sup_values = supervised
                all_X.append(sup_X)
                all_policy.append(sup_policy)
                all_values.append(sup_values)
                total_loaded += len(sup_X)

        # Then load self-play data
        files = sorted(self.data_dir.glob("dataset_*.npz"), reverse=True)  # Newest first

        for filepath in files:
            if total_loaded >= self.max_positions:
                break

            try:
                data = np.load(filepath)
                X = data["X"]
                policy = data["policy"]
                values = data["values"]

                # Take only what we need
                remaining = self.max_positions - total_loaded
                if len(X) > remaining:
                    indices = np.random.choice(len(X), remaining, replace=False)
                    X = X[indices]
                    policy = policy[indices]
                    values = values[indices]

                all_X.append(X)
                all_policy.append(policy)
                all_values.append(values)
                total_loaded += len(X)

            except Exception as e:
                log(f"Warning: Could not load {filepath.name}: {e}")

        # Also add current buffer
        if self.positions:
            buf_X = np.array([p[0] for p in self.positions], dtype=np.float32)
            buf_policy = np.array([p[1] for p in self.positions], dtype=np.float32)
            buf_values = np.array([p[2] for p in self.positions], dtype=np.float32)
            all_X.append(buf_X)
            all_policy.append(buf_policy)
            all_values.append(buf_values)
            total_loaded += len(buf_X)

        if not all_X:
            return None, None, None

        return (
            np.concatenate(all_X, axis=0),
            np.concatenate(all_policy, axis=0),
            np.concatenate(all_values, axis=0)
        )

    def get_stats(self):
        """Get statistics about stored data."""
        files = list(self.data_dir.glob("dataset_*.npz"))
        total_positions = 0
        for f in files:
            try:
                data = np.load(f)
                total_positions += len(data["X"])
            except:
                pass

        return {
            "files": len(files),
            "total_positions": total_positions,
            "buffer_size": len(self.positions)
        }


# ==================== Training ====================

def train_epoch(model, optimizer, X, policy_targets, value_targets, device):
    """Train for one epoch."""
    model.train()

    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    total_policy_loss = 0
    total_value_loss = 0
    n_batches = 0

    for i in range(0, n_samples, BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]

        x_batch = torch.tensor(X[batch_idx], dtype=torch.float32).to(device)
        policy_batch = torch.tensor(policy_targets[batch_idx], dtype=torch.float32).to(device)
        value_batch = torch.tensor(value_targets[batch_idx], dtype=torch.float32).to(device)

        optimizer.zero_grad()

        policy_logits, value_pred = model(x_batch)

        # Policy loss (cross-entropy)
        policy_loss = -torch.sum(policy_batch * torch.log_softmax(policy_logits, dim=1)) / len(batch_idx)

        # Value loss (MSE)
        value_loss = torch.mean((value_pred.squeeze() - value_batch) ** 2)

        # Combined loss
        loss = policy_loss + value_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return {
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss": total_value_loss / max(n_batches, 1)
    }


def generate_game(mcts, data_manager, opponent_type="random"):
    """
    Play one game and collect training data.
    opponent_type: "random" or "self"
    """
    board = chess.Board()
    game_data = []  # (tensor, policy_target, turn, is_repetition)

    move_count = 0
    max_moves = MAX_GAME_MOVES

    # Track position hashes to detect repetition
    position_counts = {}

    # Randomly decide if current model plays white or black
    current_is_white = random.choice([True, False])

    while not board.is_game_over() and move_count < max_moves:
        # Validate position before using it
        if not validate_position(board):
            break

        # Check for repetition
        pos_hash = board.board_fen()
        position_counts[pos_hash] = position_counts.get(pos_hash, 0) + 1
        is_repetition = position_counts[pos_hash] > 1

        # Determine whose turn it is
        is_current_model_turn = (board.turn == chess.WHITE) == current_is_white

        # Temperature decay: high early (exploration), low later (exploitation)
        # Use configured temperatures for more control
        progress = move_count / max_moves
        temperature = INITIAL_TEMPERATURE - (INITIAL_TEMPERATURE - FINAL_TEMPERATURE) * progress
        temperature = max(FINAL_TEMPERATURE, temperature)

        if is_current_model_turn:
            # Current model's turn - collect training data
            tensor = board_to_tensor(board)
            move, visit_counts = mcts.search(board, add_noise=True, temperature=temperature)

            if move is None:
                break

            # Create policy target from visit counts
            policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                for m, visits in visit_counts.items():
                    idx = encode_move(m)
                    if idx < POLICY_OUTPUT_SIZE:
                        policy_target[idx] = visits / total_visits

            game_data.append((tensor, policy_target, board.turn, is_repetition))
        else:
            # Opponent's turn
            if opponent_type == "random":
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = random.choice(legal)
            else:
                # Self-play uses same MCTS
                move, _ = mcts.search(board, add_noise=False, temperature=0.1)
                if move is None:
                    break

        # Make move
        board.push(move)
        move_count += 1

    # Determine game outcome
    game_finished = board.is_game_over()

    if game_finished:
        outcome = board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                result_value = 1.0
                result_type_raw = "checkmate_white"
            elif outcome.winner == chess.BLACK:
                result_value = -1.0
                result_type_raw = "checkmate_black"
            else:
                result_value = 0.0  # Stalemate/draw
                result_type_raw = "draw"
        else:
            result_value = 0.0
            result_type_raw = "draw"
    else:
        # Game didn't finish - use material evaluation!
        if USE_MATERIAL_EVAL:
            result_value = evaluate_material(board)
            result_type_raw = f"material_{result_value:.2f}"
        else:
            result_value = 0.0
            result_type_raw = "timeout"

    # Calculate result from current model's perspective
    if current_is_white:
        current_model_result = result_value
    else:
        current_model_result = -result_value

    # Add positions to data manager with correct values
    positions_added = 0
    for tensor, policy_target, turn, is_repetition in game_data:
        # Value from the perspective of the player to move
        value = result_value if turn == chess.WHITE else -result_value

        # Apply repetition penalty - repetitive moves are bad!
        if is_repetition:
            # Push value toward 0 (draw) as penalty for repetition
            value = value * (1.0 - REPETITION_PENALTY)

        data_manager.add_position(tensor, policy_target, value)
        positions_added += 1

    # Determine W/L/D for logging (from model's perspective)
    if current_model_result > 0.3:
        result_type = "win"
    elif current_model_result < -0.3:
        result_type = "loss"
    else:
        result_type = "draw"

    return {
        "moves": move_count,
        "result": current_model_result,
        "result_type": result_type,
        "positions": positions_added,
        "opponent": opponent_type,
        "finished": game_finished
    }


def test_against_random(mcts, num_games=10):
    """Test model strength by playing against random moves."""
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(num_games):
        board = chess.Board()
        model_is_white = (game_num % 2 == 0)

        while not board.is_game_over():
            if (board.turn == chess.WHITE) == model_is_white:
                # Model's turn
                move, _ = mcts.search(board, add_noise=False)
                if move is None:
                    break
            else:
                # Random's turn
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = random.choice(legal)

            board.push(move)

        outcome = board.outcome()
        if outcome:
            if outcome.winner is None:
                draws += 1
            elif (outcome.winner == chess.WHITE) == model_is_white:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    return {"wins": wins, "losses": losses, "draws": draws}


def test_against_opponent(current_mcts, opponent_mcts, num_games=10):
    """Test model strength by playing against opponent model."""
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(num_games):
        board = chess.Board()
        current_is_white = (game_num % 2 == 0)

        while not board.is_game_over():
            if (board.turn == chess.WHITE) == current_is_white:
                # Current model's turn
                move, _ = current_mcts.search(board, add_noise=False)
            else:
                # Opponent model's turn
                move, _ = opponent_mcts.search(board, add_noise=False)

            if move is None:
                break

            board.push(move)

        outcome = board.outcome()
        if outcome:
            if outcome.winner is None:
                draws += 1
            elif (outcome.winner == chess.WHITE) == current_is_white:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    return {"wins": wins, "losses": losses, "draws": draws}


# ==================== Main Training Loop ====================

def run_training(hours, model_path):
    """Main training function with crash recovery."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize log file for crash recovery
    init_log_file()

    log("=" * 60)
    log("MCTS CHESS AI TRAINING - WITH SUPERVISED DATA RETENTION")
    log("=" * 60)
    log(f"Duration: {hours} hours")
    end_time = datetime.now() + timedelta(hours=hours)
    log(f"End time: {end_time}")
    log(f"Device: {device}")
    log(f"MCTS sims: {MCTS_SIMS} | Games/iter: {GAMES_PER_ITER} | Max moves: {MAX_GAME_MOVES}")
    log(f"Random/Self-play ratio: {RANDOM_OPPONENT_RATIO*100:.0f}%/{(1-RANDOM_OPPONENT_RATIO)*100:.0f}%")
    log(f"Temperature: {INITIAL_TEMPERATURE} -> {FINAL_TEMPERATURE} | Repetition penalty: {REPETITION_PENALTY}")
    log(f"Memory: {MAX_POSITIONS_IN_MEMORY:,} positions | {MAX_DATASET_FILES} files")
    log("IMPORTANT: Supervised data (expert games) mixed into EVERY training iteration!")
    log("=" * 60)

    # Setup
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)

    model = load_or_create_model(model_path, device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

    data_manager = DataManager(data_dir)

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CRITICAL: Load supervised learning data FIRST
    # This ensures we never forget the fundamentals (openings, tactics, endgames)
    log("Loading supervised learning data (expert games + positions)...")
    supervised = load_supervised_data()
    if supervised is not None:
        sup_X, _, _ = supervised
        log(f"Supervised data loaded: {len(sup_X)} positions (will be mixed into EVERY training iteration)")
    else:
        log("WARNING: No supervised data loaded! Model may forget fundamentals.")

    # Load human game data (additional learning from real play)
    human_positions = load_human_games("data/games.db", data_manager)
    if human_positions > 0:
        log(f"Added {human_positions} positions from human games to training data")

    # Training loop
    iteration = 0
    total_games = 0
    total_positions = 0
    start_time = datetime.now()

    try:
        while datetime.now() < end_time:
            iteration += 1

            mcts = TrainingMCTS(model, device, num_simulations=MCTS_SIMS)

            # Generate games with MIXED opponents for better learning
            games_this_iter = GAMES_PER_ITER
            iter_positions = 0
            iter_results = {"wins": 0, "losses": 0, "draws": 0}
            games_vs_random = 0
            games_vs_self = 0
            games_finished = 0  # Track actual checkmates

            for game_num in range(games_this_iter):
                # Mix: 50% vs random (easy wins), 50% self-play
                if random.random() < RANDOM_OPPONENT_RATIO:
                    opponent_type = "random"
                    games_vs_random += 1
                else:
                    opponent_type = "self"
                    games_vs_self += 1

                result = generate_game(mcts, data_manager, opponent_type)
                total_games += 1
                iter_positions += result["positions"]

                if result["finished"]:
                    games_finished += 1

                if result["result_type"] == "win":
                    iter_results["wins"] += 1
                elif result["result_type"] == "loss":
                    iter_results["losses"] += 1
                else:
                    iter_results["draws"] += 1

            total_positions += iter_positions

            # Train if we have enough data
            X, policy, values = data_manager.load_training_data()

            if X is not None and len(X) >= BATCH_SIZE:
                # Train for fewer epochs for speed
                for epoch in range(TRAIN_EPOCHS):
                    losses = train_epoch(model, optimizer, X, policy, values, device)
                scheduler.step()

                # Show value distribution in training data
                v_mean = np.mean(values)
                v_std = np.std(values)

                log(f"Iter {iteration}: {games_this_iter}g (R:{games_vs_random}/S:{games_vs_self}) fin={games_finished}, "
                    f"W/L/D={iter_results['wins']}/{iter_results['losses']}/{iter_results['draws']}, "
                    f"p_loss={losses['policy_loss']:.3f}, v_loss={losses['value_loss']:.3f}, "
                    f"v_mean={v_mean:.2f}, v_std={v_std:.2f}")
            else:
                log(f"Iter {iteration}: {games_this_iter} games, pos={iter_positions}, "
                    f"W/L/D={iter_results['wins']}/{iter_results['losses']}/{iter_results['draws']} (collecting...)")

            # AUTO-SAVE main model frequently (every 3 iterations)
            if iteration % AUTO_SAVE_INTERVAL == 0:
                try:
                    save_model(model, model_path)
                    # Also save data buffer to disk
                    data_manager.save_buffer()
                except Exception as e:
                    log(f"WARNING: Auto-save failed: {e}")

            # Save numbered checkpoint less frequently (every 5 iterations)
            if iteration % CHECKPOINT_INTERVAL == 0:
                try:
                    checkpoint_path = data_dir / f"checkpoint_iter{iteration}.pt"
                    save_model(model, checkpoint_path)
                    log(f"Checkpoint saved: {checkpoint_path.name}")

                    # Keep only last 10 checkpoints to save disk space
                    checkpoints = sorted(data_dir.glob("checkpoint_iter*.pt"))
                    if len(checkpoints) > 10:
                        for old_ckpt in checkpoints[:-10]:
                            old_ckpt.unlink()
                except Exception as e:
                    log(f"WARNING: Checkpoint save failed: {e}")

            # Test strength periodically
            if iteration % STRENGTH_TEST_INTERVAL == 0:
                try:
                    test_mcts = TrainingMCTS(model, device, num_simulations=15)  # Few sims for fast testing

                    # Test against random - this is our main benchmark
                    log("Running strength test vs random...")
                    test_results = test_against_random(test_mcts, num_games=10)
                    win_rate = test_results["wins"] / 10 * 100
                    log(f"STRENGTH TEST: vs Random - W/L/D = {test_results['wins']}/{test_results['losses']}/{test_results['draws']} ({win_rate:.0f}% win rate)")
                except Exception as e:
                    log(f"WARNING: Strength test failed: {e}")

            # Clear memory periodically
            if iteration % 5 == 0:
                clear_memory()

            # Log progress summary every 20 iterations
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            if iteration % 20 == 0:
                remaining_hours = hours - elapsed_hours
                log(f"PROGRESS: {elapsed_hours:.1f}h elapsed, {remaining_hours:.1f}h remaining, {total_games} games, {total_positions} positions")

    except KeyboardInterrupt:
        log("INTERRUPTED by user (Ctrl+C)")
        log("Saving model before exit...")
        data_manager.save_buffer()
        save_model(model, model_path)
        log(f"Emergency save complete: {model_path}")
        raise

    except Exception as e:
        log(f"ERROR: Training crashed: {e}")
        log("Attempting emergency save...")
        try:
            data_manager.save_buffer()
            save_model(model, model_path)
            log(f"Emergency save complete: {model_path}")
        except:
            log("Emergency save FAILED!")
        raise

    # Final save (normal completion)
    data_manager.save_buffer()
    save_model(model, model_path)

    stats = data_manager.get_stats()

    log("")
    log("=" * 60)
    log("TRAINING COMPLETE")
    log("=" * 60)
    log(f"Total iterations: {iteration}")
    log(f"Total games: {total_games}")
    log(f"Total positions generated: {total_positions}")
    log(f"Dataset files: {stats['files']}")
    log(f"Final model saved: {model_path}")
    log("=" * 60)
    log("")
    log("To play against the AI:")
    log(f"  .venv\\Scripts\\python.exe main.py --engine mcts --load_model {model_path} --sims 100")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Clean MCTS Training for Chess AI")
    parser.add_argument("--hours", type=float, default=1.0, help="Training duration in hours")
    parser.add_argument("--model", type=str, default="dual_model_mcts.pt", help="Model file path")
    args = parser.parse_args()

    run_training(args.hours, args.model)


if __name__ == "__main__":
    main()

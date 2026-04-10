#!/usr/bin/env python3
"""
Stockfish Teacher Training for Chess AI

This script uses Stockfish as a teacher to train the neural network.
Instead of self-play (which reinforces bad habits), we learn from Stockfish's
high-quality moves.

Training modes:
1. IMITATION: Model tries to predict Stockfish's moves
2. PLAY & LEARN: Model plays against Stockfish, learns from corrections

Requirements:
- Stockfish executable (download from https://stockfishchess.org/download/)
- Place stockfish.exe in the project folder or set STOCKFISH_PATH

Usage:
    .venv\\Scripts\\python.exe train_stockfish.py --hours 4 --mode imitation
"""

import argparse
import os
import sys
import random
import time
import gc
from datetime import datetime, timedelta
from pathlib import Path

print("STARTUP: Beginning imports...", flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine

print("STARTUP: All imports complete", flush=True)

sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is on the path (so imports work when run from any directory)
SCRIPT_DIR_FOR_IMPORTS = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR_FOR_IMPORTS.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_network import DualNet, POLICY_OUTPUT_SIZE, encode_move, get_legal_move_mask
from features import board_to_tensor

# ==================== Configuration ====================

# Stockfish settings - use absolute path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
STOCKFISH_PATH = str(PROJECT_DIR / "stockfish" / "stockfish.exe")
STOCKFISH_DEPTH = 12  # Depth for analysis (higher = stronger but slower)
STOCKFISH_TIME_LIMIT = 0.15  # Time limit per move in seconds

# Training settings
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
GAMES_PER_ITER = 5
MAX_GAME_MOVES = 100
TRAIN_EPOCHS = 5

# Difficulty progression
# Start with intermediate Stockfish since model already plays well
INITIAL_ELO = 1400  # Starting strength (intermediate - model already beats casual players)
FINAL_ELO = 2000  # Target strength (strong club player)
ELO_INCREASE_INTERVAL = 15  # Increase ELO every N iterations

# Checkpoint settings
CHECKPOINT_INTERVAL = 25
AUTO_SAVE_INTERVAL = 5
LOG_FILE = "data/training/stockfish_training_log.txt"

# ==================== Utilities ====================

_log_file = None

def init_log_file():
    """Initialize log file."""
    global _log_file
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a", encoding="utf-8")
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Stockfish training started: {datetime.now().isoformat()}\n")
    _log_file.write(f"{'='*60}\n")
    _log_file.flush()


def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg, flush=True)
    if _log_file is not None:
        _log_file.write(full_msg + "\n")
        _log_file.flush()


def clear_memory():
    """Force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Stockfish Interface ====================

class StockfishTeacher:
    """Interface to Stockfish chess engine for training."""

    def __init__(self, path=STOCKFISH_PATH, elo=1500):
        self.path = path
        self.elo = elo
        self.engine = None
        self._connect()

    def _connect(self):
        """Connect to Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)

            # Set skill level based on ELO
            # Stockfish skill levels: 0-20
            # Approximate mapping: ELO 800-2800 -> Skill 0-20
            skill = max(0, min(20, int((self.elo - 800) / 100)))
            self.engine.configure({"Skill Level": skill})

            log(f"Connected to Stockfish (ELO ~{self.elo}, Skill {skill})")
        except Exception as e:
            log(f"ERROR: Could not connect to Stockfish: {e}")
            log(f"Make sure stockfish.exe is in: {os.path.abspath(self.path)}")
            raise

    def set_elo(self, elo):
        """Update Stockfish strength."""
        self.elo = elo
        skill = max(0, min(20, int((elo - 800) / 100)))
        self.engine.configure({"Skill Level": skill})

    def get_best_move(self, board, time_limit=STOCKFISH_TIME_LIMIT):
        """Get Stockfish's best move for a position."""
        try:
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move
        except Exception as e:
            log(f"Warning: Stockfish error: {e}")
            return None

    def get_move_scores(self, board, depth=STOCKFISH_DEPTH):
        """Get evaluation scores for all legal moves."""
        scores = {}
        try:
            for move in board.legal_moves:
                board.push(move)
                info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info.get("score")
                if score:
                    # Negate because we pushed the move (now opponent's perspective)
                    cp = score.relative.score(mate_score=10000)
                    if cp is not None:
                        scores[move] = -cp
                board.pop()
        except Exception as e:
            log(f"Warning: Analysis error: {e}")
        return scores

    def evaluate_position(self, board, depth=STOCKFISH_DEPTH):
        """Get position evaluation in centipawns."""
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info.get("score")
            if score:
                cp = score.relative.score(mate_score=10000)
                if cp is not None:
                    return cp / 100.0  # Convert to pawn units
            return 0.0
        except:
            return 0.0

    def close(self):
        """Close the engine."""
        if self.engine:
            self.engine.quit()


# ==================== Neural Network ====================

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


# ==================== Training Data Generation ====================

def generate_training_game(model, stockfish, device, model_plays_white=True):
    """
    Play a game: model vs Stockfish.
    Collect positions where model deviated from Stockfish's choice.
    """
    board = chess.Board()
    training_data = []
    move_count = 0

    model.eval()

    while not board.is_game_over() and move_count < MAX_GAME_MOVES:
        is_model_turn = (board.turn == chess.WHITE) == model_plays_white

        if is_model_turn:
            # Model's turn - get its move
            with torch.no_grad():
                tensor = board_to_tensor(board)
                x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)
                mask = get_legal_move_mask(board).to(device).unsqueeze(0)

                policy_logits, value = model(x)
                policy_logits = policy_logits.masked_fill(mask == 0, -1e9)
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0)

                # Get model's top choice
                legal_moves = list(board.legal_moves)
                move_probs = [(m, policy_probs[encode_move(m)].item()) for m in legal_moves]
                move_probs.sort(key=lambda x: x[1], reverse=True)
                model_move = move_probs[0][0]

            # Get Stockfish's best move
            sf_move = stockfish.get_best_move(board)

            if sf_move and sf_move != model_move:
                # Model chose differently - create training sample
                policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
                policy_target[encode_move(sf_move)] = 1.0

                # Get position evaluation
                eval_score = stockfish.evaluate_position(board)
                value_target = np.tanh(eval_score / 4.0)  # Normalize to [-1, 1]
                if not model_plays_white:
                    value_target = -value_target

                training_data.append((tensor, policy_target, value_target))

            # Play model's move (to continue the game naturally)
            board.push(model_move)
        else:
            # Stockfish's turn
            sf_move = stockfish.get_best_move(board)
            if sf_move is None:
                break
            board.push(sf_move)

        move_count += 1

    # Determine result
    result = "unknown"
    if board.is_game_over():
        outcome = board.outcome()
        if outcome:
            if outcome.winner is None:
                result = "draw"
            elif (outcome.winner == chess.WHITE) == model_plays_white:
                result = "model_win"
            else:
                result = "model_loss"

    return training_data, result, move_count


_COMMON_OPENINGS = [
    # Sicilian, French, Caro-Kann, Italian, Ruy Lopez, Queen's Gambit, etc.
    "e2e4 e7e5", "e2e4 c7c5", "e2e4 e7e6", "e2e4 c7c6", "e2e4 d7d5",
    "d2d4 d7d5", "d2d4 g8f6", "d2d4 d7d5 c2c4", "d2d4 d7d5 c2c4 e7e6",
    "e2e4 e7e5 g1f3 b8c6", "e2e4 e7e5 g1f3 b8c6 f1c4",
    "e2e4 e7e5 g1f3 b8c6 f1b5", "d2d4 d7d5 c2c4 c7c6",
    "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 g1f3 b8c6",
    "d2d4 g8f6 c2c4 g7g6", "c2c4 e7e5", "g1f3 d7d5",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6",
    "",  # starting position (no opening moves)
]

def generate_imitation_data(stockfish, num_positions=200):
    """
    Generate training data by having Stockfish play itself.
    Model learns to imitate Stockfish's moves.

    Value target is from the CURRENT PLAYER's perspective:
    - Positive = good for side to move
    - Negative = bad for side to move
    This ensures the NN learns perspective-correct evaluation.
    """
    training_data = []

    for game_num in range(num_positions // 20):  # ~20 positions per game
        board = chess.Board()
        move_count = 0

        # Use real openings for realistic positions (with some randomization)
        if random.random() < 0.7:
            opening = random.choice(_COMMON_OPENINGS)
            for uci in opening.split():
                if uci:
                    try:
                        board.push_uci(uci)
                    except ValueError:
                        break
        else:
            # 30% random openings for variety
            for _ in range(random.randint(4, 10)):
                if board.is_game_over():
                    break
                legal = list(board.legal_moves)
                if legal:
                    board.push(random.choice(legal))

        # Now collect Stockfish moves
        while not board.is_game_over() and move_count < 40:
            sf_move = stockfish.get_best_move(board)
            if sf_move is None:
                break

            # Create training sample
            tensor = board_to_tensor(board)

            policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
            policy_target[encode_move(sf_move)] = 1.0

            # Stockfish eval is from White's perspective (positive = White better)
            # Convert to CURRENT PLAYER's perspective for training
            eval_score = stockfish.evaluate_position(board)
            # eval_score is already relative (from side-to-move's perspective
            # in python-chess's engine analysis), so just normalize
            value_target = np.tanh(eval_score / 4.0)

            training_data.append((tensor, policy_target, value_target))

            board.push(sf_move)
            move_count += 1

    return training_data


# ==================== Training ====================

def train_epoch(model, optimizer, training_data, device):
    """Train for one epoch."""
    model.train()

    if not training_data:
        return {"policy_loss": 0, "value_loss": 0}

    # Convert to arrays
    X = np.array([d[0] for d in training_data], dtype=np.float32)
    policy_targets = np.array([d[1] for d in training_data], dtype=np.float32)
    value_targets = np.array([d[2] for d in training_data], dtype=np.float32)

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

        # Policy loss (cross-entropy with Stockfish's moves)
        policy_loss = -torch.sum(policy_batch * torch.log_softmax(policy_logits, dim=1)) / len(batch_idx)

        # Value loss (MSE with Stockfish's evaluation)
        value_loss = torch.mean((value_pred.squeeze() - value_batch) ** 2)

        # Weight value loss 10x so the value head gets proper gradient signal
        # (policy loss ~5.0 vs value loss ~0.05 — without weighting, value head starves)
        loss = policy_loss + 10.0 * value_loss

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


# ==================== Main Training Loop ====================

def run_training(hours, model_path, mode="imitation"):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_log_file()

    log("=" * 60)
    log("STOCKFISH TEACHER TRAINING")
    log("=" * 60)
    log(f"Mode: {mode}")
    log(f"Duration: {hours} hours")
    end_time = datetime.now() + timedelta(hours=hours)
    log(f"End time: {end_time}")
    log(f"Device: {device}")
    log(f"Stockfish path: {os.path.abspath(STOCKFISH_PATH)}")
    log("=" * 60)

    # Check if Stockfish exists
    if not os.path.exists(STOCKFISH_PATH):
        log(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        log("Download from: https://stockfishchess.org/download/")
        log("Extract stockfish.exe to the project folder")
        return

    # Setup
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)

    model = load_or_create_model(model_path, device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stockfish = StockfishTeacher(STOCKFISH_PATH, elo=INITIAL_ELO)

    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Initial Stockfish ELO: {INITIAL_ELO}")

    # Training loop
    iteration = 0
    total_positions = 0
    start_time = datetime.now()
    current_elo = INITIAL_ELO

    try:
        while datetime.now() < end_time:
            iteration += 1

            # Gradually increase Stockfish strength
            if iteration % ELO_INCREASE_INTERVAL == 0 and current_elo < FINAL_ELO:
                current_elo = min(FINAL_ELO, current_elo + 100)
                stockfish.set_elo(current_elo)
                log(f"Increased Stockfish ELO to {current_elo}")

            training_data = []
            model_wins = 0
            model_losses = 0
            draws = 0

            if mode == "imitation":
                # Pure imitation learning from Stockfish self-play
                training_data = generate_imitation_data(stockfish, num_positions=200)
            else:
                # Play & Learn mode
                for game_num in range(GAMES_PER_ITER):
                    plays_white = (game_num % 2 == 0)
                    game_data, result, moves = generate_training_game(
                        model, stockfish, device, model_plays_white=plays_white
                    )
                    training_data.extend(game_data)

                    if result == "model_win":
                        model_wins += 1
                    elif result == "model_loss":
                        model_losses += 1
                    else:
                        draws += 1

            total_positions += len(training_data)

            # Train
            if training_data:
                for epoch in range(TRAIN_EPOCHS):
                    losses = train_epoch(model, optimizer, training_data, device)

                if mode == "imitation":
                    log(f"Iter {iteration}: {len(training_data)} positions, "
                        f"p_loss={losses['policy_loss']:.3f}, v_loss={losses['value_loss']:.3f}, "
                        f"SF_ELO={current_elo}")
                else:
                    log(f"Iter {iteration}: {len(training_data)} positions, "
                        f"W/L/D={model_wins}/{model_losses}/{draws}, "
                        f"p_loss={losses['policy_loss']:.3f}, v_loss={losses['value_loss']:.3f}")

            # Auto-save
            if iteration % AUTO_SAVE_INTERVAL == 0:
                save_model(model, model_path)

            # Checkpoint
            if iteration % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = data_dir / f"sf_checkpoint_iter{iteration}.pt"
                save_model(model, checkpoint_path)
                log(f"Checkpoint saved: {checkpoint_path.name}")

            # Memory cleanup
            if iteration % 5 == 0:
                clear_memory()

            # Progress
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            if iteration % 10 == 0:
                remaining = hours - elapsed
                log(f"PROGRESS: {elapsed:.1f}h elapsed, {remaining:.1f}h remaining, "
                    f"{total_positions} total positions")

    except KeyboardInterrupt:
        log("INTERRUPTED by user")
        save_model(model, model_path)
        log(f"Model saved: {model_path}")

    except Exception as e:
        log(f"ERROR: {e}")
        save_model(model, model_path)
        raise

    finally:
        stockfish.close()

    # Final save
    save_model(model, model_path)

    log("")
    log("=" * 60)
    log("TRAINING COMPLETE")
    log("=" * 60)
    log(f"Total iterations: {iteration}")
    log(f"Total positions learned: {total_positions}")
    log(f"Final Stockfish ELO: {current_elo}")
    log(f"Model saved: {model_path}")
    log("=" * 60)
    log("")
    log("To play against the AI:")
    log(f"  .venv\\Scripts\\python.exe main.py --engine mcts --load_model {model_path} --sims 100")


def main():
    parser = argparse.ArgumentParser(description="Stockfish Teacher Training")
    parser.add_argument("--hours", type=float, default=2.0, help="Training duration")
    parser.add_argument("--model", type=str, default="models/dual_model_mcts.pt", help="Model path")
    parser.add_argument("--mode", type=str, default="imitation",
                        choices=["imitation", "play"],
                        help="Training mode: 'imitation' or 'play'")
    args = parser.parse_args()

    run_training(args.hours, args.model, args.mode)


if __name__ == "__main__":
    main()

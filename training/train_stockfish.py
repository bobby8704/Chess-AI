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
import math
import os
import sys
import random
import time
import gc
from datetime import datetime, timedelta
from pathlib import Path

# Only the launching process announces itself. Teacher-pool workers re-import this
# module on spawn, and without the guard every worker prints these two lines into the
# training log on every iteration.
_IS_MAIN_PROCESS = __name__ == "__main__" or "multiprocessing" not in sys.modules

if _IS_MAIN_PROCESS:
    print("STARTUP: Beginning imports...", flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine

if _IS_MAIN_PROCESS:
    print("STARTUP: All imports complete", flush=True)

sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is on the path (so imports work when run from any directory)
SCRIPT_DIR_FOR_IMPORTS = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR_FOR_IMPORTS.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_network import DualNet, DualNetCNN, POLICY_OUTPUT_SIZE, encode_move, get_legal_move_mask
from features import board_to_tensor, board_to_tensor_2d

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
WEIGHT_DECAY = 1e-4  # L2 regularization to prevent overfitting
GAMES_PER_ITER = 5
MAX_GAME_MOVES = 100
TRAIN_EPOCHS = 3  # Fewer epochs per batch to avoid memorization

# Difficulty progression
# Start with intermediate Stockfish since model already plays well
INITIAL_ELO = 1400  # Starting strength (intermediate - model already beats casual players)
FINAL_ELO = 2000  # Target strength (strong club player)
ELO_INCREASE_INTERVAL = 15  # Increase ELO every N iterations

# Architecture (set by CLI arg)
USE_CNN = True  # True = DualNetCNN, False = DualNet MLP

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

    def get_move_and_eval(self, board, time_limit=STOCKFISH_TIME_LIMIT):
        """
        Return (best_move, eval_in_pawns) from a SINGLE search.

        Replaces calling get_best_move() and then evaluate_position() on the same board,
        which ran two full Stockfish searches per training position and accounted for
        ~35% of every iteration. `info=INFO_SCORE` makes play() report the root score it
        already computed, so the second search is pure waste.

        The score comes from a time-limited search rather than the old fixed depth-12
        analyse, so the value target shifts slightly. That is well inside the teacher's
        own noise: this teacher uses Skill Level, which is deliberately stochastic and
        reproduces its own move only ~40-50% of the time on a repeated query.
        """
        try:
            result = self.engine.play(
                board, chess.engine.Limit(time=time_limit),
                info=chess.engine.INFO_SCORE)
            score = result.info.get("score")
            cp = score.relative.score(mate_score=10000) if score else None
            return result.move, (cp / 100.0 if cp is not None else 0.0)
        except Exception as e:
            log(f"Warning: Stockfish error: {e}")
            return None, 0.0

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

CNN_NUM_FILTERS = 128
CNN_NUM_RES_BLOCKS = 10  # Deeper backbone for better pattern recognition

def load_or_create_model(model_path, device, architecture="cnn", allow_fresh=False):
    """
    Load an existing model, or create one.

    Takes the layer dimensions FROM THE CHECKPOINT rather than from the module
    constants, and treats any unexpected load failure as FATAL.

    This function previously destroyed a 13.7-hour training run in silence. It only
    compared model_type ("DualNetCNN" == "DualNetCNN"), never the dimensions, so a
    6-residual-block checkpoint was fed to a 10-block model; load_state_dict raised a
    missing-key error; the bare `except Exception` logged "Could not load model,
    starting fresh"; and the run trained from random weights all night. The log line is
    still there at data/training/stockfish_training_log.txt:5242. A trainer that is 16x
    faster and still does this is just a faster way to waste a night, so a mismatch now
    stops the run unless --fresh is passed explicitly.
    """
    ckpt = None
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if ckpt is None:
        if not allow_fresh:
            raise SystemExit(
                f"FATAL: no model at {model_path}\n"
                "Pass --fresh to start from random weights on purpose."
            )
        log(f"No existing model found, creating fresh {architecture}")
        if architecture == "cnn":
            return DualNetCNN(in_channels=13, num_filters=CNN_NUM_FILTERS,
                              num_res_blocks=CNN_NUM_RES_BLOCKS).to(device)
        return DualNet(input_dim=781).to(device)

    model_type = ckpt.get("model_type", "DualNet") if isinstance(ckpt, dict) else "DualNet"
    want_cnn = (architecture == "cnn")
    if want_cnn != (model_type == "DualNetCNN"):
        raise SystemExit(
            f"FATAL: {model_path} is a {model_type} but --arch is {architecture}.\n"
            "Refusing to silently discard it. Use the matching --arch, or --fresh."
        )

    # Build to the checkpoint's own shape, not to this module's constants.
    if want_cnn:
        filters = ckpt.get("num_filters", CNN_NUM_FILTERS)
        blocks = ckpt.get("num_res_blocks", CNN_NUM_RES_BLOCKS)
        channels = ckpt.get("in_channels", 13)
        if (filters, blocks) != (CNN_NUM_FILTERS, CNN_NUM_RES_BLOCKS):
            log(f"NOTE: checkpoint is {filters}f/{blocks}b but this script's constants "
                f"are {CNN_NUM_FILTERS}f/{CNN_NUM_RES_BLOCKS}b — continuing with the "
                f"checkpoint's shape so its weights are preserved.")
        model = DualNetCNN(in_channels=channels, num_filters=filters,
                           num_res_blocks=blocks).to(device)
    else:
        model = DualNet(input_dim=ckpt.get("input_dim", 781)).to(device)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        raise SystemExit(
            f"FATAL: {model_path} did not load into the model built from its own "
            f"metadata:\n  {e}\n"
            "This is the failure that silently wasted a 13.7h run. Fix the checkpoint "
            "or pass --fresh to start over deliberately."
        )
    log(f"Loaded existing model: {model_path} ({model_type})")
    return model


def save_model(model, path, optimizer=None, iteration=None, total_positions=None):
    """
    Save the model, and the optimizer state alongside it.

    Without optimizer state and an iteration counter a run cannot be resumed: Adam's
    moment estimates restart from zero, which throws away its warm-up every time. These
    fields are optional additions, so checkpoints stay loadable by neural_network.py
    and everything else that reads only state_dict.
    """
    payload = {
        "state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    if isinstance(model, DualNetCNN):
        payload.update({
            "model_type": "DualNetCNN",
            "in_channels": model.in_channels,
            "num_filters": model.num_filters,
            "num_res_blocks": model.num_res_blocks,
        })
    else:
        payload.update({"model_type": "DualNet", "input_dim": 781})

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if iteration is not None:
        payload["iteration"] = iteration
    if total_positions is not None:
        payload["total_positions"] = total_positions
    torch.save(payload, path)


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
                tensor = board_to_tensor_2d(board) if USE_CNN else board_to_tensor(board)
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
    # e4 openings
    "e2e4 e7e5", "e2e4 c7c5", "e2e4 e7e6", "e2e4 c7c6", "e2e4 d7d5",
    "e2e4 d7d6", "e2e4 g7g6", "e2e4 b7b6", "e2e4 g8f6",
    # e4 e5 systems
    "e2e4 e7e5 g1f3 b8c6 f1c4", "e2e4 e7e5 g1f3 b8c6 f1b5",
    "e2e4 e7e5 g1f3 b8c6 d2d4", "e2e4 e7e5 g1f3 g8f6",
    "e2e4 e7e5 f1c4", "e2e4 e7e5 d2d4",
    # Sicilian variations
    "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 g1f3 b8c6",
    "e2e4 c7c5 g1f3 e7e6", "e2e4 c7c5 b1c3",
    # French, Caro-Kann
    "e2e4 e7e6 d2d4 d7d5", "e2e4 c7c6 d2d4 d7d5",
    # d4 openings
    "d2d4 d7d5", "d2d4 g8f6", "d2d4 d7d5 c2c4", "d2d4 d7d5 c2c4 e7e6",
    "d2d4 d7d5 c2c4 c7c6", "d2d4 d7d5 g1f3 g8f6",
    "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6",
    "d2d4 d7d5 c1f4", "d2d4 f7f5",
    # English, Reti, others
    "c2c4 e7e5", "c2c4 g8f6", "g1f3 d7d5", "g1f3 g8f6",
    "b2b3 e7e5", "g2g3 d7d5", "f2f4 d7d5",
    "",  # starting position
]

def _make_random_position(max_moves=20):
    """Generate a random legal position by playing random moves.
    Creates diverse positions the model wouldn't see in normal games."""
    board = chess.Board()
    n = random.randint(4, max_moves)
    for _ in range(n):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        board.push(random.choice(legal))
    return board

def generate_imitation_data(stockfish, num_positions=200):
    """
    Generate training data by having Stockfish play from diverse positions.
    Mix of: 50% real openings, 25% random positions, 25% deep random positions.
    """
    training_data = []

    for game_num in range(num_positions // 20):  # ~20 positions per game
        board = chess.Board()
        move_count = 0

        # Diverse starting positions: heavy on openings for good development
        roll = random.random()
        if roll < 0.70:
            # 70%: real openings (35+ variations) — teaches proper development
            opening = random.choice(_COMMON_OPENINGS)
            for uci in opening.split():
                if uci:
                    try:
                        board.push_uci(uci)
                    except ValueError:
                        break
        elif roll < 0.85:
            # 15%: shallow random positions (4-12 moves — early middlegame)
            board = _make_random_position(max_moves=12)
        else:
            # 15%: deep random positions (12-25 moves — middlegame/endgame)
            board = _make_random_position(max_moves=25)

        # Now collect Stockfish moves
        while not board.is_game_over() and move_count < 40:
            # One search, not two: the move and the root score come together.
            sf_move, eval_score = stockfish.get_move_and_eval(board)
            if sf_move is None:
                break

            # Create training sample
            tensor = board_to_tensor_2d(board) if USE_CNN else board_to_tensor(board)

            policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
            policy_target[encode_move(sf_move)] = 1.0

            # eval_score is relative (from the side to move's perspective), so it just
            # needs normalising to the tanh range the value head is trained on.
            value_target = np.tanh(eval_score / 4.0)

            training_data.append((tensor, policy_target, value_target))

            board.push(sf_move)
            move_count += 1

    return training_data


# --- Parallel teacher pool -------------------------------------------------
#
# Position generation was ~94% of every iteration and ran on ONE Stockfish process with
# Threads=1 — a single core of a 20-thread machine, with the trainer idle the whole
# time. Positions WITHIN a game are strictly sequential (each depends on Stockfish's
# previous move), so the parallelism has to be across games: one game per worker, and
# more shorter games rather than fewer long ones.

_POOL_TEACHER = None


def _teacher_init(elo):
    global _POOL_TEACHER
    _POOL_TEACHER = StockfishTeacher(STOCKFISH_PATH, elo=elo)


def _teacher_game(args):
    """Generate ONE game's worth of positions. Runs in a pool worker."""
    seed, elo = args
    random.seed(seed)
    # The curriculum raises Stockfish's strength as training proceeds. The pool is
    # long-lived, so each worker retunes its own engine when the target moves rather
    # than the pool being torn down and rebuilt.
    if _POOL_TEACHER is not None and _POOL_TEACHER.elo != elo:
        _POOL_TEACHER.set_elo(elo)
    try:
        # num_positions=20 is exactly one game in generate_imitation_data's own terms
        # (it runs num_positions // 20 games), so the parallel and serial paths produce
        # identically-shaped data and the same count per iteration.
        return generate_imitation_data(_POOL_TEACHER, num_positions=20)
    except Exception:
        return []


def make_teacher_pool(workers, elo):
    """
    Create the teacher pool ONCE, for the whole run.

    Rebuilding it per iteration cost ~4-5s of the ~20s iteration in spawning eight
    Stockfish processes and re-importing this module in each — a fifth of the run spent
    on startup, plus eight "Connected to Stockfish" lines per iteration in the log.
    """
    from multiprocessing import Pool
    return Pool(workers, initializer=_teacher_init, initargs=(elo,))


def generate_imitation_data_parallel(num_positions, elo, pool):
    """
    The same data generate_imitation_data would produce, spread over a teacher pool —
    one game per task, since positions within a game are strictly sequential.

    Dispatches the SAME number of games as the serial path and does not truncate, so
    switching to the pool changes throughput and nothing else. (An earlier version
    capped the result at num_positions and quietly halved the data per iteration, which
    would have looked like a change in training dynamics.)

    Each Stockfish is ~250MB resident, so worker count is a memory decision as much as a
    speed one. Measured here: 8 teachers 29.8 pos/s, 12 -> 26.9, 16 -> 24.6 — more is
    worse past the physical core count.
    """
    n_games = max(1, num_positions // 20)
    tasks = [(random.randrange(1 << 30), elo) for _ in range(n_games)]
    out = []
    for chunk in pool.imap_unordered(_teacher_game, tasks):
        out.extend(chunk)
    return out


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

def run_training(hours, model_path, mode="imitation", teachers=1, allow_fresh=False):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Training runs at batch 128-400, which is exactly where this GPU wins — unlike
        # batch-1 inference, which stays on CPU/ONNX. TF32 is a large free speedup on
        # Ampere and later; AMP/fp16 measured SLOWER here, because GradScaler overhead
        # exceeds the tensor-core gain on a 4.6M-parameter model with 8x8 inputs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    init_log_file()

    arch = "cnn" if USE_CNN else "mlp"

    log("=" * 60)
    log("STOCKFISH TEACHER TRAINING")
    log("=" * 60)
    log(f"Mode: {mode}")
    log(f"Architecture: {arch.upper()}")
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

    model = load_or_create_model(model_path, device, architecture=arch,
                                 allow_fresh=allow_fresh)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Resume Adam's moment estimates if the checkpoint carries them; otherwise its
    # warm-up is thrown away on every restart.
    _ck = torch.load(model_path, map_location=device, weights_only=False) \
        if os.path.exists(model_path) else {}
    if isinstance(_ck, dict) and "optimizer_state" in _ck:
        try:
            optimizer.load_state_dict(_ck["optimizer_state"])
            log(f"Resumed optimizer state (from iteration {_ck.get('iteration', '?')})")
        except Exception as e:
            log(f"NOTE: optimizer state present but not loadable ({e}); starting Adam fresh")
    del _ck

    log(f"Teacher processes: {teachers}")
    stockfish = StockfishTeacher(STOCKFISH_PATH, elo=INITIAL_ELO)
    teacher_pool = make_teacher_pool(teachers, INITIAL_ELO) if teachers > 1 else None

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

            _t_gen = time.time()
            if mode == "imitation":
                # Pure imitation learning from Stockfish self-play
                if teacher_pool is not None:
                    training_data = generate_imitation_data_parallel(
                        200, current_elo, teacher_pool)
                else:
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
            gen_s = time.time() - _t_gen

            # Train
            _t_train = time.time()
            if training_data:
                for epoch in range(TRAIN_EPOCHS):
                    losses = train_epoch(model, optimizer, training_data, device)
                train_s = time.time() - _t_train

                if mode == "imitation":
                    # gen/train split is logged because it is what decides where the
                    # next optimisation should go: teacher parallelism only helps while
                    # generation dominates, and GPU training only helps once it does not.
                    log(f"Iter {iteration}: {len(training_data)} positions, "
                        f"p_loss={losses['policy_loss']:.3f}, v_loss={losses['value_loss']:.3f}, "
                        f"SF_ELO={current_elo}, gen={gen_s:.1f}s train={train_s:.1f}s")
                else:
                    log(f"Iter {iteration}: {len(training_data)} positions, "
                        f"W/L/D={model_wins}/{model_losses}/{draws}, "
                        f"p_loss={losses['policy_loss']:.3f}, v_loss={losses['value_loss']:.3f}")

            # Auto-save
            if iteration % AUTO_SAVE_INTERVAL == 0:
                save_model(model, model_path, optimizer, iteration, total_positions)

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
        save_model(model, model_path, optimizer, iteration, total_positions)
        raise

    finally:
        stockfish.close()
        if teacher_pool is not None:
            teacher_pool.terminate()
            teacher_pool.join()

    # Final save
    save_model(model, model_path, optimizer, iteration, total_positions)

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
    global USE_CNN
    parser = argparse.ArgumentParser(description="Stockfish Teacher Training")
    parser.add_argument("--hours", type=float, default=2.0, help="Training duration")
    parser.add_argument("--model", type=str, default="models/dual_model_mcts.pt", help="Model path")
    parser.add_argument("--mode", type=str, default="imitation",
                        choices=["imitation", "play"],
                        help="Training mode: 'imitation' or 'play'")
    parser.add_argument("--arch", type=str, default="cnn",
                        choices=["cnn", "mlp"],
                        help="Architecture: 'cnn' (ResNet) or 'mlp' (flat)")
    parser.add_argument("--teachers", type=int, default=8,
                        help="Parallel Stockfish teacher processes. Position generation "
                             "is ~94%% of an iteration and was single-process. Each "
                             "engine is ~250MB resident, so this is a memory decision "
                             "as much as a speed one; 1 restores the old behaviour.")
    parser.add_argument("--fresh", action="store_true",
                        help="Start from random weights ON PURPOSE. Without this, a "
                             "missing or mismatched checkpoint is a fatal error rather "
                             "than a silent restart — which once cost a 13.7h run.")
    args = parser.parse_args()

    USE_CNN = (args.arch == "cnn")
    run_training(args.hours, args.model, args.mode, args.teachers, args.fresh)


if __name__ == "__main__":
    main()

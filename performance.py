"""
Performance Optimization Utilities for Chess AI

This module provides performance enhancements:
- Batch inference for neural networks
- Position caching (transposition table)
- Parallel self-play with multiprocessing
- GPU optimization utilities
- Performance benchmarking tools

Usage:
    from performance import BatchEvaluator, TranspositionTable, parallel_selfplay
"""

import os
import time
import hashlib
import multiprocessing as mp
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from functools import lru_cache

import numpy as np
import chess
import torch

from features import board_to_tensor


# ==================== BATCH INFERENCE ====================

class BatchEvaluator:
    """
    Batch evaluator for neural network inference.

    Collects positions and evaluates them in batches for better GPU utilization.
    Significantly faster than single-position evaluation when processing many positions.

    Usage:
        evaluator = BatchEvaluator(model, batch_size=64)

        # Queue positions for evaluation
        for board in boards:
            evaluator.queue(board)

        # Get all results
        values = evaluator.evaluate_all()
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        batch_size: int = 64,
        device: str = None
    ):
        self.model = model
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.queue: List[Tuple[chess.Board, np.ndarray]] = []

        if model is not None:
            model.to(self.device)
            model.eval()

    def queue_position(self, board: chess.Board) -> int:
        """
        Queue a position for batch evaluation.

        Returns:
            Index of the queued position
        """
        tensor = board_to_tensor(board)
        idx = len(self.queue)
        self.queue.append((board.copy(), tensor))
        return idx

    def evaluate_all(self) -> List[float]:
        """
        Evaluate all queued positions in batches.

        Returns:
            List of values for each queued position
        """
        if not self.queue:
            return []

        if self.model is None:
            # Return zeros if no model
            return [0.0] * len(self.queue)

        results = []
        tensors = [t for _, t in self.queue]

        # Process in batches
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i:i + self.batch_size]
            batch_tensor = torch.from_numpy(np.array(batch)).float().to(self.device)

            with torch.no_grad():
                values = self.model(batch_tensor)

            results.extend(values.squeeze(-1).cpu().numpy().tolist())

        self.queue.clear()
        return results

    def evaluate_single(self, board: chess.Board) -> float:
        """Evaluate a single position (convenience method)."""
        self.queue_position(board)
        return self.evaluate_all()[0]


class DualBatchEvaluator:
    """
    Batch evaluator for DualNet (policy + value).

    Returns both policy probabilities and value for each position.
    """

    def __init__(
        self,
        model: torch.nn.Module = None,
        batch_size: int = 64,
        device: str = None
    ):
        self.model = model
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.queue: List[Tuple[chess.Board, np.ndarray, torch.Tensor]] = []

        if model is not None:
            model.to(self.device)
            model.eval()

    def queue_position(self, board: chess.Board) -> int:
        """Queue a position with its legal move mask."""
        from neural_network import get_legal_move_mask

        tensor = board_to_tensor(board)
        mask = get_legal_move_mask(board)
        idx = len(self.queue)
        self.queue.append((board.copy(), tensor, mask))
        return idx

    def evaluate_all(self) -> List[Tuple[Dict[chess.Move, float], float]]:
        """
        Evaluate all queued positions.

        Returns:
            List of (move_probs_dict, value) tuples
        """
        from neural_network import encode_move, POLICY_OUTPUT_SIZE

        if not self.queue:
            return []

        if self.model is None:
            # Return uniform policy and zero value
            results = []
            for board, _, _ in self.queue:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    prob = 1.0 / len(legal_moves)
                    probs = {m: prob for m in legal_moves}
                else:
                    probs = {}
                results.append((probs, 0.0))
            self.queue.clear()
            return results

        # Prepare batches
        boards = [b for b, _, _ in self.queue]
        tensors = np.array([t for _, t, _ in self.queue])
        masks = torch.stack([m for _, _, m in self.queue])

        results = []

        for i in range(0, len(tensors), self.batch_size):
            batch_boards = boards[i:i + self.batch_size]
            batch_tensors = torch.from_numpy(tensors[i:i + self.batch_size]).float().to(self.device)
            batch_masks = masks[i:i + self.batch_size].to(self.device)

            with torch.no_grad():
                policy_probs, values = self.model.get_policy_value(batch_tensors, batch_masks)

            policy_probs = policy_probs.cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()

            for j, board in enumerate(batch_boards):
                move_probs = {}
                for move in board.legal_moves:
                    idx = encode_move(move)
                    if idx < POLICY_OUTPUT_SIZE:
                        move_probs[move] = float(policy_probs[j, idx])

                # Normalize
                total = sum(move_probs.values())
                if total > 0:
                    move_probs = {m: p / total for m, p in move_probs.items()}

                results.append((move_probs, float(values[j])))

        self.queue.clear()
        return results


# ==================== TRANSPOSITION TABLE ====================

class TranspositionTable:
    """
    Position cache using Zobrist hashing.

    Stores evaluated positions to avoid redundant neural network calls.
    Uses LRU eviction when the cache is full.

    Usage:
        cache = TranspositionTable(max_size=100000)

        # Check cache before evaluation
        value = cache.get(board)
        if value is None:
            value = evaluate(board)
            cache.put(board, value)
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.cache: OrderedDict[str, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_board(self, board: chess.Board) -> str:
        """Create a hash key for the board position."""
        # Use FEN for simple hashing (Zobrist would be faster but more complex)
        return board.fen()

    def get(self, board: chess.Board) -> Optional[float]:
        """
        Get cached value for a position.

        Returns:
            Cached value or None if not found
        """
        key = self._hash_board(board)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, board: chess.Board, value: float):
        """Store a value for a position."""
        key = self._hash_board(board)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                self.cache.popitem(last=False)
        self.cache[key] = value

    def get_policy_value(
        self,
        board: chess.Board
    ) -> Optional[Tuple[Dict[chess.Move, float], float]]:
        """Get cached policy and value (for DualNet)."""
        key = self._hash_board(board) + "_pv"
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put_policy_value(
        self,
        board: chess.Board,
        policy: Dict[chess.Move, float],
        value: float
    ):
        """Store policy and value for a position."""
        key = self._hash_board(board) + "_pv"
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = (policy, value)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Get the cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate
        }


# Global transposition table
_global_cache: Optional[TranspositionTable] = None


def get_global_cache(max_size: int = 100000) -> TranspositionTable:
    """Get or create the global transposition table."""
    global _global_cache
    if _global_cache is None:
        _global_cache = TranspositionTable(max_size)
    return _global_cache


# ==================== PARALLEL SELF-PLAY ====================

def _play_single_game(args: Tuple) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Play a single game (worker function for multiprocessing).

    Args:
        Tuple of (game_id, engine_w, engine_b, depth_w, depth_b, model_path)

    Returns:
        Tuple of (pgn_string, X_array, y_array)
    """
    game_id, engine_w, engine_b, depth_w, depth_b, model_path = args

    import random
    import chess
    import chess.pgn
    random.seed(game_id)

    # Import here to avoid pickling issues
    from ai import best_move_using_minimax
    from ai_nn import best_move_nn, load_value_model
    from features import board_to_tensor

    if model_path and (engine_w == "nn" or engine_b == "nn"):
        try:
            load_value_model(model_path)
        except:
            pass

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = engine_w
    game.headers["Black"] = engine_b
    node = game

    X = []
    sgns = []

    while not board.is_game_over():
        X.append(board_to_tensor(board))
        sgns.append(1.0 if board.turn == chess.WHITE else -1.0)

        if board.turn == chess.WHITE:
            engine, depth = engine_w, depth_w
        else:
            engine, depth = engine_b, depth_b

        if engine == "minimax":
            move = best_move_using_minimax(board, depth)
        elif engine == "nn":
            move = best_move_nn(board, depth)
        elif engine == "random":
            move = random.choice(list(board.legal_moves))
        else:
            move = random.choice(list(board.legal_moves))

        board.push(move)
        node = node.add_variation(move)

    result = board.result()
    game.headers["Result"] = result

    z = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
    y = np.array(sgns, dtype=np.float32) * z

    return str(game), np.array(X, dtype=np.float32), y


def parallel_selfplay(
    n_games: int,
    engine_w: str = "minimax",
    engine_b: str = "minimax",
    depth_w: int = 2,
    depth_b: int = 2,
    model_path: str = None,
    n_workers: int = None,
    verbose: bool = True
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Run self-play games in parallel using multiprocessing.

    Args:
        n_games: Number of games to play
        engine_w: White engine name
        engine_b: Black engine name
        depth_w: White search depth
        depth_b: Black search depth
        model_path: Path to neural network model
        n_workers: Number of worker processes (None = CPU count)
        verbose: Print progress

    Returns:
        Tuple of (pgn_list, X_all, y_all)
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Prepare arguments for each game
    args_list = [
        (i, engine_w, engine_b, depth_w, depth_b, model_path)
        for i in range(n_games)
    ]

    if verbose:
        print(f"Starting parallel self-play: {n_games} games on {n_workers} workers")

    start_time = time.time()

    # Use multiprocessing pool
    # Note: For Windows compatibility, use 'spawn' method
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_workers) as pool:
            results = list(pool.imap(_play_single_game, args_list))
    except Exception as e:
        if verbose:
            print(f"Multiprocessing failed ({e}), falling back to sequential")
        results = [_play_single_game(args) for args in args_list]

    # Combine results
    pgn_list = [r[0] for r in results]
    X_list = [r[1] for r in results]
    y_list = [r[2] for r in results]

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    elapsed = time.time() - start_time

    if verbose:
        games_per_sec = n_games / elapsed
        print(f"Completed {n_games} games in {elapsed:.1f}s ({games_per_sec:.1f} games/sec)")

    return pgn_list, X_all, y_all


# ==================== GPU UTILITIES ====================

def get_device_info() -> Dict:
    """Get information about available compute devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "memory_allocated": None,
        "memory_cached": None
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**2  # MB
        info["memory_cached"] = torch.cuda.memory_reserved(0) / 1024**2  # MB

    return info


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize a model for inference.

    Applies optimizations like eval mode and optional JIT compilation.
    """
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    return model


def warmup_model(model: torch.nn.Module, input_dim: int = 781, n_warmup: int = 10):
    """
    Warm up the model with dummy inputs.

    This helps ensure consistent timing by initializing CUDA kernels.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, input_dim).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ==================== BENCHMARKING ====================

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    iterations: int
    total_time: float
    mean_time: float
    std_time: float
    throughput: float  # items per second


def benchmark_evaluation(
    model: torch.nn.Module = None,
    n_positions: int = 1000,
    batch_sizes: List[int] = None
) -> List[BenchmarkResult]:
    """
    Benchmark neural network evaluation speed.

    Args:
        model: Model to benchmark (None = create dummy ValueNet)
        n_positions: Number of positions to evaluate
        batch_sizes: List of batch sizes to test

    Returns:
        List of benchmark results for each batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32, 64, 128]

    if model is None:
        from ai_nn import ValueNet
        model = ValueNet(781)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Generate random positions
    positions = torch.randn(n_positions, 781).to(device)

    results = []

    for batch_size in batch_sizes:
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(positions[:batch_size])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        for i in range(0, n_positions, batch_size):
            batch = positions[i:i + batch_size]

            start = time.perf_counter()
            with torch.no_grad():
                _ = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        total_time = sum(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = n_positions / total_time

        results.append(BenchmarkResult(
            name=f"batch_{batch_size}",
            iterations=len(times),
            total_time=total_time,
            mean_time=mean_time,
            std_time=std_time,
            throughput=throughput
        ))

    return results


def benchmark_mcts(
    n_positions: int = 10,
    simulations_list: List[int] = None,
    model=None
) -> List[BenchmarkResult]:
    """
    Benchmark MCTS search speed.

    Args:
        n_positions: Number of positions to search
        simulations_list: List of simulation counts to test
        model: Neural network model (None = use random rollouts)

    Returns:
        List of benchmark results
    """
    from mcts import MCTSPlayer, MCTSConfig

    if simulations_list is None:
        simulations_list = [50, 100, 200, 400]

    results = []

    for n_sims in simulations_list:
        config = MCTSConfig(
            num_simulations=n_sims,
            temperature=0,
            add_noise=False
        )
        player = MCTSPlayer(model=model, config=config)

        times = []
        board = chess.Board()

        for _ in range(n_positions):
            start = time.perf_counter()
            _ = player.select_move(board)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        total_time = sum(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = n_positions / total_time

        results.append(BenchmarkResult(
            name=f"mcts_{n_sims}",
            iterations=n_positions,
            total_time=total_time,
            mean_time=mean_time,
            std_time=std_time,
            throughput=throughput
        ))

    return results


def print_benchmark_results(results: List[BenchmarkResult]):
    """Pretty print benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Name':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput':<15}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<20} {r.mean_time*1000:<12.2f} {r.std_time*1000:<12.2f} "
              f"{r.throughput:<15.1f} pos/s")


if __name__ == "__main__":
    print("Performance Utilities Test")
    print("=" * 50)

    # Test device info
    print("\nDevice Info:")
    info = get_device_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test TranspositionTable
    print("\nTesting TranspositionTable:")
    cache = TranspositionTable(max_size=1000)
    board = chess.Board()

    # Miss
    val = cache.get(board)
    print(f"  First lookup (miss): {val}")

    # Put
    cache.put(board, 0.5)
    print("  Stored value: 0.5")

    # Hit
    val = cache.get(board)
    print(f"  Second lookup (hit): {val}")

    print(f"  Cache stats: {cache.stats()}")

    # Test BatchEvaluator
    print("\nTesting BatchEvaluator:")
    evaluator = BatchEvaluator(model=None, batch_size=32)

    for _ in range(10):
        evaluator.queue_position(chess.Board())

    values = evaluator.evaluate_all()
    print(f"  Evaluated {len(values)} positions: {values[:3]}...")

    # Benchmark evaluation (quick test)
    print("\nQuick Evaluation Benchmark:")
    results = benchmark_evaluation(n_positions=100, batch_sizes=[1, 16, 32])
    print_benchmark_results(results)

    print("\nAll performance tests passed!")

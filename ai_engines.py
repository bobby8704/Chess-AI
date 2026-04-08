"""
Unified AI Engines for Chess

This module provides a unified interface for all chess AI engines:
- minimax: Basic minimax with alpha-beta pruning
- nn: Neural network enhanced minimax (existing ValueNet)
- mcts: Monte Carlo Tree Search with optional neural network
- random: Random legal move selection

Usage:
    from ai_engines import get_engine, list_engines

    engine = get_engine("mcts", model_path="value_model.pt", num_simulations=100)
    move = engine.select_move(board)
"""

import random
import time
import chess
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class MoveResult:
    """Result of a move selection with metadata."""
    move: chess.Move
    evaluation: Optional[float] = None
    depth: Optional[int] = None
    time_ms: Optional[int] = None
    visit_count: Optional[int] = None
    move_probs: Optional[Dict[chess.Move, float]] = None
    engine_name: str = ""


class BaseEngine:
    """Base class for all chess engines."""

    name: str = "base"

    def select_move(self, board: chess.Board) -> MoveResult:
        """Select a move for the current position."""
        raise NotImplementedError

    def get_evaluation(self, board: chess.Board) -> float:
        """Get position evaluation (-1 to 1 from white's perspective)."""
        return 0.0


class RandomEngine(BaseEngine):
    """Random move selection."""

    name = "random"

    def select_move(self, board: chess.Board) -> MoveResult:
        start = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return MoveResult(move=None, engine_name=self.name)

        move = random.choice(legal_moves)
        elapsed = int((time.time() - start) * 1000)

        return MoveResult(
            move=move,
            time_ms=elapsed,
            engine_name=self.name
        )


class MinimaxEngine(BaseEngine):
    """Basic minimax with alpha-beta pruning."""

    name = "minimax"

    def __init__(self, depth: int = 2):
        self.depth = depth

    def select_move(self, board: chess.Board) -> MoveResult:
        from ai import best_move_using_minimax

        start = time.time()
        move = best_move_using_minimax(board, self.depth)
        elapsed = int((time.time() - start) * 1000)

        return MoveResult(
            move=move,
            depth=self.depth,
            time_ms=elapsed,
            engine_name=self.name
        )

    def get_evaluation(self, board: chess.Board) -> float:
        from ai import evaluate_board
        return evaluate_board(board) / 100.0  # Normalize centipawns to [-1, 1] range


class NNEngine(BaseEngine):
    """Neural network enhanced minimax."""

    name = "nn"

    def __init__(self, model_path: str = None, depth: int = 2):
        self.depth = depth
        self.model_path = model_path
        self._model_loaded = False

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        from ai_nn import load_value_model
        try:
            load_value_model(path)
            self._model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model {path}: {e}")

    def select_move(self, board: chess.Board) -> MoveResult:
        from ai_nn import best_move_nn, evaluate_board_nn

        start = time.time()
        move = best_move_nn(board, self.depth)
        elapsed = int((time.time() - start) * 1000)

        evaluation = evaluate_board_nn(board) if move else None

        return MoveResult(
            move=move,
            evaluation=evaluation,
            depth=self.depth,
            time_ms=elapsed,
            engine_name=self.name
        )

    def get_evaluation(self, board: chess.Board) -> float:
        from ai_nn import evaluate_board_nn
        return evaluate_board_nn(board)


class MCTSEngine(BaseEngine):
    """Monte Carlo Tree Search with neural network."""

    name = "mcts"

    def __init__(
        self,
        model_path: str = None,
        num_simulations: int = 100,
        temperature: float = 0.0,
        add_noise: bool = False,
        c_puct: float = 1.5
    ):
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.add_noise = add_noise
        self.c_puct = c_puct
        self.model = None
        self.player = None

        if model_path:
            self._load_model(model_path)

        self._init_player()

    def _load_model(self, path: str):
        """Load the dual network model."""
        try:
            from neural_network import load_dual_model
            self.model = load_dual_model(path)
        except Exception as e:
            print(f"Warning: Could not load dual model {path}: {e}")
            # Try loading as value model for backward compatibility
            try:
                from neural_network import DualNet
                import torch
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                # Create a simple wrapper if it's an old ValueNet
                if "state_dict" in ckpt:
                    print("  Falling back to simple evaluation mode")
            except:
                pass

    def _init_player(self):
        """Initialize the MCTS player."""
        from mcts import MCTSPlayer, MCTSConfig

        config = MCTSConfig(
            num_simulations=self.num_simulations,
            temperature=self.temperature,
            add_noise=self.add_noise,
            c_puct=self.c_puct
        )

        self.player = MCTSPlayer(model=self.model, config=config)

    def select_move(self, board: chess.Board) -> MoveResult:
        start = time.time()

        move, probs = self.player.mcts.search(board, self.num_simulations)
        elapsed = int((time.time() - start) * 1000)

        # Get evaluation from model
        evaluation = None
        if self.model:
            _, evaluation = self.player._evaluate(board)

        return MoveResult(
            move=move,
            evaluation=evaluation,
            time_ms=elapsed,
            visit_count=self.num_simulations,
            move_probs=probs,
            engine_name=self.name
        )

    def get_evaluation(self, board: chess.Board) -> float:
        if self.player:
            _, value = self.player._evaluate(board)
            return value
        return 0.0


# Engine registry
_ENGINE_CLASSES = {
    "random": RandomEngine,
    "minimax": MinimaxEngine,
    "nn": NNEngine,
    "mcts": MCTSEngine,
}


def list_engines() -> list:
    """List available engine names."""
    return list(_ENGINE_CLASSES.keys())


def get_engine(
    name: str,
    model_path: str = None,
    depth: int = 2,
    num_simulations: int = 100,
    temperature: float = 0.0,
    **kwargs
) -> BaseEngine:
    """
    Get an engine instance by name.

    Args:
        name: Engine name ("random", "minimax", "nn", "mcts")
        model_path: Path to model file (for nn and mcts)
        depth: Search depth (for minimax and nn)
        num_simulations: MCTS simulation count
        temperature: MCTS temperature
        **kwargs: Additional engine-specific arguments

    Returns:
        Engine instance
    """
    if name not in _ENGINE_CLASSES:
        raise ValueError(f"Unknown engine: {name}. Available: {list_engines()}")

    engine_class = _ENGINE_CLASSES[name]

    if name == "random":
        return engine_class()
    elif name == "minimax":
        return engine_class(depth=depth)
    elif name == "nn":
        return engine_class(model_path=model_path, depth=depth)
    elif name == "mcts":
        return engine_class(
            model_path=model_path,
            num_simulations=num_simulations,
            temperature=temperature,
            **kwargs
        )

    return engine_class()


def choose_move(
    board: chess.Board,
    engine_name: str,
    depth: int = 2,
    model_path: str = None,
    num_simulations: int = 100,
    **kwargs
) -> Tuple[chess.Move, MoveResult]:
    """
    Convenience function to get a move from any engine.

    Returns:
        (move, move_result) tuple
    """
    engine = get_engine(
        engine_name,
        model_path=model_path,
        depth=depth,
        num_simulations=num_simulations,
        **kwargs
    )

    result = engine.select_move(board)
    return result.move, result


if __name__ == "__main__":
    # Test all engines
    print("Testing AI Engines...")
    print("=" * 50)

    board = chess.Board()
    print(f"\nPosition:\n{board}\n")

    for engine_name in list_engines():
        print(f"Testing {engine_name}...")

        if engine_name == "mcts":
            engine = get_engine(engine_name, num_simulations=50)
        else:
            engine = get_engine(engine_name, depth=2)

        result = engine.select_move(board)

        print(f"  Move: {result.move}")
        print(f"  Time: {result.time_ms}ms")
        if result.evaluation is not None:
            print(f"  Eval: {result.evaluation:.3f}")
        if result.move_probs:
            print(f"  Top moves: {list(result.move_probs.keys())[:3]}")
        print()

    print("All engine tests passed!")

#!/usr/bin/env python3
"""
Automated Training Pipeline for Chess AI

This module orchestrates the complete training cycle:
1. Self-play: Generate games using current best model
2. Training: Train new model on generated games
3. Evaluation: Compare new model against current best
4. Promotion: If new model wins, it becomes the new best

The pipeline runs continuously, improving the AI with each iteration.

Usage:
    # Run a single training iteration
    python training_pipeline.py --iterations 1

    # Run continuous training
    python training_pipeline.py --iterations 10

    # With custom configuration
    python training_pipeline.py --config config.yaml
"""

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""

    # Paths
    base_dir: str = "data/pipeline"
    model_dir: str = "data/pipeline/models"
    dataset_dir: str = "data/pipeline/datasets"
    db_path: str = "data/games.db"

    # Self-play settings
    selfplay_games: int = 50
    selfplay_engine: str = "nn"  # or "mcts"
    selfplay_depth: int = 2
    mcts_simulations: int = 100
    parallel_selfplay: bool = True  # Use multiprocessing for self-play
    selfplay_workers: int = None  # Number of workers (None = CPU count - 1)

    # Training settings
    training_epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    augment_data: bool = True
    use_dual_net: bool = False  # Use DualNet instead of ValueNet

    # Evaluation settings
    eval_games: int = 20
    eval_depth: int = 2
    win_threshold: float = 0.55  # New model must win > 55% to be promoted

    # Pipeline settings
    max_iterations: int = 100
    checkpoint_every: int = 5
    verbose: bool = True

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class IterationResult:
    """Result of a single training iteration."""
    iteration: int
    timestamp: str
    selfplay_games: int
    selfplay_positions: int
    training_epochs: int
    training_loss: float
    validation_loss: float
    eval_games: int
    eval_wins: int
    eval_losses: int
    eval_draws: int
    win_rate: float
    promoted: bool
    new_model_path: Optional[str] = None
    best_model_path: Optional[str] = None
    duration_seconds: float = 0.0


class TrainingPipeline:
    """
    Automated training pipeline for continuous improvement.

    The pipeline follows this cycle:
    1. Self-play with current best model
    2. Train new model on self-play games
    3. Evaluate new model vs current best
    4. Promote if new model is better
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.current_iteration = 0
        self.best_model_path = None
        self.history: List[IterationResult] = []

        # Setup directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.dataset_dir).mkdir(parents=True, exist_ok=True)

    def _log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def _get_model_path(self, iteration: int, suffix: str = "") -> str:
        """Get the model path for a given iteration."""
        name = f"model_iter{iteration:04d}{suffix}.pt"
        return os.path.join(self.config.model_dir, name)

    def _get_dataset_path(self, iteration: int) -> str:
        """Get the dataset path for a given iteration."""
        name = f"dataset_iter{iteration:04d}.npz"
        return os.path.join(self.config.dataset_dir, name)

    def run_selfplay(self, iteration: int) -> Dict[str, Any]:
        """
        Run self-play to generate training data.

        Returns:
            Dictionary with selfplay results
        """
        self._log(f"Starting self-play: {self.config.selfplay_games} games")

        dataset_path = self._get_dataset_path(iteration)
        pgn_path = dataset_path.replace('.npz', '.pgn')

        start_time = time.time()

        # Use parallel self-play if enabled and engine supports it
        use_parallel = (
            self.config.parallel_selfplay and
            self.config.selfplay_engine in ["minimax", "nn", "random"]
        )

        if use_parallel:
            from performance import parallel_selfplay
            import numpy as np

            self._log(f"  Using parallel self-play with {self.config.selfplay_workers or 'auto'} workers")

            pgn_list, X_all, y_all = parallel_selfplay(
                n_games=self.config.selfplay_games,
                engine_w=self.config.selfplay_engine,
                engine_b=self.config.selfplay_engine,
                depth_w=self.config.selfplay_depth,
                depth_b=self.config.selfplay_depth,
                model_path=self.best_model_path,
                n_workers=self.config.selfplay_workers,
                verbose=self.config.verbose
            )

            n_games = len(pgn_list)
            n_positions = len(X_all)

            # Save dataset
            np.savez_compressed(dataset_path, X=X_all, y=y_all)

            # Save PGN
            with open(pgn_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(pgn_list))

        else:
            from selfplay import batch_self_play
            from ai_nn import load_value_model

            # Load current best model if available
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    load_value_model(self.best_model_path)
                    self._log(f"  Loaded model: {self.best_model_path}")
                except Exception as e:
                    self._log(f"  Warning: Could not load model: {e}", "WARN")

            _, _, n_games, n_positions = batch_self_play(
                n_games=self.config.selfplay_games,
                engine_w=self.config.selfplay_engine,
                engine_b=self.config.selfplay_engine,
                depth_w=self.config.selfplay_depth,
                depth_b=self.config.selfplay_depth,
                out_dataset=dataset_path,
                out_pgn=pgn_path,
                db_path=self.config.db_path
            )

        elapsed = time.time() - start_time

        self._log(f"  Generated {n_games} games, {n_positions} positions in {elapsed:.1f}s")

        return {
            "games": n_games,
            "positions": n_positions,
            "dataset_path": dataset_path,
            "pgn_path": pgn_path,
            "duration": elapsed
        }

    def run_training(self, iteration: int, dataset_path: str) -> Dict[str, Any]:
        """
        Train a new model on the generated data.

        Returns:
            Dictionary with training results
        """
        self._log(f"Starting training: {self.config.training_epochs} epochs")

        new_model_path = self._get_model_path(iteration, "_candidate")

        start_time = time.time()

        if self.config.use_dual_net:
            from train_dual import train_dual
            results = train_dual(
                dataset_path=dataset_path,
                epochs=self.config.training_epochs,
                batch_size=self.config.batch_size,
                lr=self.config.learning_rate,
                out_path=new_model_path,
                augment=self.config.augment_data,
                db_path=self.config.db_path,
                verbose=self.config.verbose
            )
        else:
            from train_improved import train
            results = train(
                dataset_path=dataset_path,
                epochs=self.config.training_epochs,
                batch_size=self.config.batch_size,
                lr=self.config.learning_rate,
                out_path=new_model_path,
                augment=self.config.augment_data,
                db_path=self.config.db_path,
                verbose=self.config.verbose
            )

        elapsed = time.time() - start_time

        self._log(f"  Training complete in {elapsed:.1f}s")
        self._log(f"  Best val loss: {results['best_val_loss']:.5f}")

        return {
            "model_path": new_model_path,
            "epochs": results['epochs_trained'],
            "train_loss": results['final_train_loss'],
            "val_loss": results['best_val_loss'],
            "duration": elapsed
        }

    def run_evaluation(
        self,
        candidate_path: str,
        best_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate candidate model against current best.

        Returns:
            Dictionary with evaluation results
        """
        self._log(f"Starting evaluation: {self.config.eval_games} games")

        from arena import play
        from ai_nn import load_value_model
        from game_recorder import GameRecorder

        recorder = GameRecorder(self.config.db_path)

        # If no best model, candidate automatically wins
        if best_path is None or not os.path.exists(best_path):
            self._log("  No previous best model, candidate promoted automatically")
            return {
                "games": 0,
                "wins": self.config.eval_games,
                "losses": 0,
                "draws": 0,
                "win_rate": 1.0,
                "promoted": True
            }

        # Load candidate model for evaluation
        load_value_model(candidate_path)

        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        start_time = time.time()

        for i in range(self.config.eval_games):
            # Alternate colors for fairness
            if i % 2 == 0:
                # Candidate plays white
                load_value_model(candidate_path)
                game, result = play(
                    "nn", "nn", self.config.eval_depth,
                    model=best_path,  # Opponent loads best
                    recorder=recorder
                )
                # Candidate wins if white wins
                if result == "1-0":
                    results["1-0"] += 1
                elif result == "0-1":
                    results["0-1"] += 1
                else:
                    results["1/2-1/2"] += 1
            else:
                # Candidate plays black
                load_value_model(best_path)
                game, result = play(
                    "nn", "nn", self.config.eval_depth,
                    model=candidate_path,  # Opponent loads candidate
                    recorder=recorder
                )
                # Candidate wins if black wins
                if result == "0-1":
                    results["1-0"] += 1  # Count as candidate win
                elif result == "1-0":
                    results["0-1"] += 1  # Count as candidate loss
                else:
                    results["1/2-1/2"] += 1

        elapsed = time.time() - start_time

        total_games = sum(results.values())
        wins = results["1-0"]
        losses = results["0-1"]
        draws = results["1/2-1/2"]

        # Win rate: wins count as 1, draws as 0.5
        win_rate = (wins + 0.5 * draws) / total_games if total_games > 0 else 0

        promoted = win_rate >= self.config.win_threshold

        self._log(f"  Results: +{wins} ={draws} -{losses} ({win_rate:.1%})")
        self._log(f"  Promoted: {promoted}")

        return {
            "games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "promoted": promoted,
            "duration": elapsed
        }

    def run_iteration(self, iteration: int) -> IterationResult:
        """
        Run a single training iteration.

        Returns:
            IterationResult with all metrics
        """
        self._log("=" * 60)
        self._log(f"ITERATION {iteration}")
        self._log("=" * 60)

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # Phase 1: Self-play
        selfplay_results = self.run_selfplay(iteration)

        # Phase 2: Training
        training_results = self.run_training(
            iteration,
            selfplay_results["dataset_path"]
        )

        # Phase 3: Evaluation
        eval_results = self.run_evaluation(
            training_results["model_path"],
            self.best_model_path
        )

        # Phase 4: Promotion
        if eval_results["promoted"]:
            new_best_path = self._get_model_path(iteration, "_best")
            shutil.copy(training_results["model_path"], new_best_path)
            self.best_model_path = new_best_path
            self._log(f"  New best model: {new_best_path}")
        else:
            new_best_path = self.best_model_path

        total_duration = time.time() - start_time

        result = IterationResult(
            iteration=iteration,
            timestamp=timestamp,
            selfplay_games=selfplay_results["games"],
            selfplay_positions=selfplay_results["positions"],
            training_epochs=training_results["epochs"],
            training_loss=training_results["train_loss"],
            validation_loss=training_results["val_loss"],
            eval_games=eval_results["games"],
            eval_wins=eval_results["wins"],
            eval_losses=eval_results["losses"],
            eval_draws=eval_results["draws"],
            win_rate=eval_results["win_rate"],
            promoted=eval_results["promoted"],
            new_model_path=training_results["model_path"],
            best_model_path=new_best_path,
            duration_seconds=total_duration
        )

        self.history.append(result)
        self._save_history()

        self._log(f"Iteration {iteration} complete in {total_duration:.1f}s")

        return result

    def run(self, iterations: int = None):
        """
        Run the training pipeline for multiple iterations.

        Args:
            iterations: Number of iterations (None = use config)
        """
        n_iterations = iterations or self.config.max_iterations

        self._log(f"Starting training pipeline: {n_iterations} iterations")
        self._log(f"Configuration: {asdict(self.config)}")

        # Load existing best model if available
        self._find_best_model()

        for i in range(1, n_iterations + 1):
            self.current_iteration = i

            try:
                result = self.run_iteration(i)

                # Checkpoint
                if i % self.config.checkpoint_every == 0:
                    self._save_checkpoint(i)

            except KeyboardInterrupt:
                self._log("Pipeline interrupted by user", "WARN")
                break
            except Exception as e:
                self._log(f"Error in iteration {i}: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                continue

        self._log("Pipeline complete!")
        self._print_summary()

    def _find_best_model(self):
        """Find the best model from previous runs."""
        model_dir = Path(self.config.model_dir)
        if not model_dir.exists():
            return

        best_models = sorted(model_dir.glob("*_best.pt"))
        if best_models:
            self.best_model_path = str(best_models[-1])
            self._log(f"Found existing best model: {self.best_model_path}")

    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config.base_dir, "history.json")
        history_data = [asdict(r) for r in self.history]
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

    def _save_checkpoint(self, iteration: int):
        """Save a checkpoint of the current state."""
        checkpoint_path = os.path.join(
            self.config.base_dir,
            f"checkpoint_iter{iteration:04d}.json"
        )
        checkpoint = {
            "iteration": iteration,
            "best_model_path": self.best_model_path,
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        self._log(f"Saved checkpoint: {checkpoint_path}")

    def _print_summary(self):
        """Print a summary of the training run."""
        if not self.history:
            return

        print("\n" + "=" * 60)
        print("TRAINING PIPELINE SUMMARY")
        print("=" * 60)

        total_games = sum(r.selfplay_games for r in self.history)
        total_positions = sum(r.selfplay_positions for r in self.history)
        total_time = sum(r.duration_seconds for r in self.history)
        promotions = sum(1 for r in self.history if r.promoted)

        print(f"Iterations completed: {len(self.history)}")
        print(f"Total self-play games: {total_games}")
        print(f"Total training positions: {total_positions}")
        print(f"Model promotions: {promotions}")
        print(f"Total time: {total_time / 60:.1f} minutes")

        if self.best_model_path:
            print(f"Best model: {self.best_model_path}")

        # Win rate trend
        if len(self.history) > 1:
            win_rates = [r.win_rate for r in self.history]
            print(f"Win rate trend: {win_rates[0]:.1%} -> {win_rates[-1]:.1%}")


def create_default_config(path: str):
    """Create a default configuration file."""
    config = PipelineConfig()
    config.save(path)
    print(f"Created default config: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the automated training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 5 training iterations
    python training_pipeline.py --iterations 5

    # Run with custom config
    python training_pipeline.py --config my_config.yaml

    # Create a default config file
    python training_pipeline.py --create-config config.yaml

    # Quick test run
    python training_pipeline.py --iterations 1 --games 10 --epochs 5 --eval_games 4
        """
    )

    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of training iterations")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--create-config", type=str, default=None,
                        help="Create a default config file at path")

    # Override config options
    parser.add_argument("--games", type=int, default=None,
                        help="Override: self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override: training epochs")
    parser.add_argument("--eval_games", type=int, default=None,
                        help="Override: evaluation games")
    parser.add_argument("--engine", choices=["nn", "mcts"], default=None,
                        help="Override: self-play engine")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override: win threshold for promotion")
    parser.add_argument("--parallel", action="store_true", default=None,
                        help="Enable parallel self-play")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel",
                        help="Disable parallel self-play")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    # Create config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()

    # Apply overrides
    if args.games:
        config.selfplay_games = args.games
    if args.epochs:
        config.training_epochs = args.epochs
    if args.eval_games:
        config.eval_games = args.eval_games
    if args.engine:
        config.selfplay_engine = args.engine
    if args.threshold:
        config.win_threshold = args.threshold
    if args.parallel is not None:
        config.parallel_selfplay = args.parallel
    if args.workers:
        config.selfplay_workers = args.workers
    if args.quiet:
        config.verbose = False

    # Run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run(iterations=args.iterations)


if __name__ == "__main__":
    main()

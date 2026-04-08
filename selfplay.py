
import argparse
import random
import time
import numpy as np
import chess
import chess.pgn

from features import board_to_tensor
from ai import best_move_using_minimax
from ai_nn import best_move_nn, load_value_model
from game_recorder import (
    GameRecorder, PlayerType, GameResult,
    player_type_from_engine, result_from_string
)

# Global MCTS player cache
_mcts_player = None


def choose_move(board: chess.Board, engine: str, depth: int, num_simulations: int = 100):
    global _mcts_player

    if engine == "minimax":
        return best_move_using_minimax(board, depth)
    elif engine == "nn":
        return best_move_nn(board, depth)
    elif engine == "random":
        return random.choice(list(board.legal_moves))
    elif engine == "mcts":
        # Lazy initialize MCTS player
        if _mcts_player is None:
            from mcts import MCTSPlayer, MCTSConfig
            config = MCTSConfig(
                num_simulations=num_simulations,
                temperature=1.0,  # Temperature for training diversity
                add_noise=True    # Exploration noise for training
            )
            _mcts_player = MCTSPlayer(model=None, config=config)
        _mcts_player.config.num_simulations = num_simulations
        return _mcts_player.select_move(board)
    else:
        raise ValueError(f"Unknown engine '{engine}'")

def play_game(
    engine_w: str = "minimax",
    engine_b: str = "minimax",
    depth_w: int = 2,
    depth_b: int = 2,
    seed: int | None = None,
    recorder: GameRecorder | None = None,
    white_model_id: int | None = None,
    black_model_id: int | None = None
):
    if seed is not None:
        random.seed(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play"
    game.headers["White"] = engine_w
    game.headers["Black"] = engine_b
    node = game

    # Start recording if recorder provided
    game_id = None
    if recorder:
        game_id = recorder.start_game(
            white_player=engine_w,
            black_player=engine_b,
            white_type=player_type_from_engine(engine_w),
            black_type=player_type_from_engine(engine_b),
            white_model_id=white_model_id,
            black_model_id=black_model_id,
            white_depth=depth_w,
            black_depth=depth_b,
            event="Self-Play"
        )

    # For value learning dataset
    X = []
    # Whose perspective each X corresponds to (1 for white-to-move, -1 for black-to-move)
    sgns = []

    while not board.is_game_over():
        X.append(board_to_tensor(board))
        sgns.append(1.0 if board.turn == chess.WHITE else -1.0)

        # Determine current player's settings
        if board.turn == chess.WHITE:
            engine, depth = engine_w, depth_w
        else:
            engine, depth = engine_b, depth_b

        # Time the move
        start_time = time.time()
        move = choose_move(board, engine, depth)
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Record the move before pushing
        if recorder and game_id:
            recorder.record_move(
                game_id=game_id,
                board=board,
                move=move,
                depth=depth,
                time_ms=elapsed_ms
            )

        board.push(move)
        node = node.add_variation(move)

    result = board.result()  # "1-0", "0-1", "1/2-1/2"
    game.headers["Result"] = result

    # End game recording
    if recorder and game_id:
        termination = None
        if board.is_checkmate():
            termination = "CHECKMATE"
        elif board.is_stalemate():
            termination = "STALEMATE"
        elif board.is_insufficient_material():
            termination = "INSUFFICIENT_MATERIAL"
        elif board.is_fifty_moves():
            termination = "FIFTY_MOVES"
        elif board.is_repetition():
            termination = "REPETITION"

        recorder.end_game(
            game_id=game_id,
            result=result_from_string(result),
            termination=termination,
            pgn=str(game)
        )

    z = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
    # For each position, target from side-to-move perspective is sgn * z
    y = np.array(sgns, dtype=np.float32) * z
    return game, np.array(X, dtype=np.float32), y

def batch_self_play(
    n_games: int = 50,
    engine_w: str = "minimax",
    engine_b: str = "minimax",
    depth_w: int = 2,
    depth_b: int = 2,
    out_dataset: str = "data/datasets/dataset.npz",
    out_pgn: str = "data/datasets/selfplay.pgn",
    db_path: str = "data/games.db",
    white_model_id: int | None = None,
    black_model_id: int | None = None
):
    # Initialize game recorder
    recorder = GameRecorder(db_path)

    X_list, y_list = [], []
    pgn_games = []
    for i in range(n_games):
        g, X, y = play_game(
            engine_w, engine_b, depth_w, depth_b,
            seed=i,
            recorder=recorder,
            white_model_id=white_model_id,
            black_model_id=black_model_id
        )
        X_list.append(X)
        y_list.append(y)
        pgn_games.append(str(g))

        # Progress indicator
        if (i + 1) % 10 == 0 or i == n_games - 1:
            print(f"  Completed {i + 1}/{n_games} games...")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    np.savez_compressed(out_dataset, X=X_all, y=y_all)
    with open(out_pgn, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pgn_games))

    # Print database statistics
    stats = recorder.get_statistics()
    print(f"\nDatabase stats: {stats['total_games']} total games, {stats['total_moves']} total moves")

    return out_dataset, out_pgn, len(pgn_games), len(X_all)

def main():
    ap = argparse.ArgumentParser(description="Run self-play to create a dataset and PGN.")
    ap.add_argument("--n_games", type=int, default=50)
    ap.add_argument("--engine_w", choices=["minimax", "nn", "random", "mcts"], default="minimax")
    ap.add_argument("--engine_b", choices=["minimax", "nn", "random", "mcts"], default="minimax")
    ap.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    ap.add_argument("--depth_w", type=int, default=2)
    ap.add_argument("--depth_b", type=int, default=2)
    ap.add_argument("--out_dataset", default="data/datasets/dataset.npz")
    ap.add_argument("--out_pgn", default="data/datasets/selfplay.pgn")
    ap.add_argument("--load_model", default=None, help="Path to value_model.pt if using engine 'nn'")
    ap.add_argument("--db_path", default="data/games.db", help="Path to SQLite game database")
    ap.add_argument("--white_model_id", type=int, default=None, help="Model ID for white player (if AI)")
    ap.add_argument("--black_model_id", type=int, default=None, help="Model ID for black player (if AI)")
    args = ap.parse_args()

    if args.engine_w == "nn" or args.engine_b == "nn":
        if args.load_model is None:
            print("Note: engine 'nn' selected but no --load_model provided. Will default to neutral evals.")
        else:
            load_value_model(args.load_model)

    print(f"Starting self-play: {args.n_games} games")
    print(f"  White: {args.engine_w} (depth={args.depth_w})")
    print(f"  Black: {args.engine_b} (depth={args.depth_b})")
    print(f"  Recording to: {args.db_path}")
    print()

    ds, pgn, n_games, n_pos = batch_self_play(
        n_games=args.n_games,
        engine_w=args.engine_w,
        engine_b=args.engine_b,
        depth_w=args.depth_w,
        depth_b=args.depth_b,
        out_dataset=args.out_dataset,
        out_pgn=args.out_pgn,
        db_path=args.db_path,
        white_model_id=args.white_model_id,
        black_model_id=args.black_model_id
    )
    print(f"\nWrote {n_games} games and {n_pos} positions.")
    print(f"Dataset: {ds}")
    print(f"PGN:     {pgn}")

if __name__ == "__main__":
    main()

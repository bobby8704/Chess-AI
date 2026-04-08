
import argparse
import time
import chess
import chess.pgn
from ai import best_move_using_minimax
from ai_nn import best_move_nn, load_value_model
from game_recorder import (
    GameRecorder, PlayerType, GameResult,
    player_type_from_engine, result_from_string
)


def choose(board, name, depth):
    if name == "minimax":
        return best_move_using_minimax(board, depth)
    elif name == "nn":
        return best_move_nn(board, depth)
    elif name == "random":
        import random
        return random.choice(list(board.legal_moves))
    else:
        raise ValueError(f"Unknown engine: {name}")


def play(
    engine_w: str,
    engine_b: str,
    depth: int,
    model: str = None,
    recorder: GameRecorder = None,
    white_model_id: int = None,
    black_model_id: int = None
):
    if model:
        load_value_model(model)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Arena"
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
            white_depth=depth,
            black_depth=depth,
            event="Arena"
        )

    while not board.is_game_over():
        engine = engine_w if board.turn == chess.WHITE else engine_b

        # Time the move
        start_time = time.time()
        move = choose(board, engine, depth)
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

    result = board.result()
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

    return game, result


def main():
    ap = argparse.ArgumentParser(description="Play head-to-head AI vs AI arena matches.")
    ap.add_argument("--white", choices=["minimax", "nn", "random"], default="minimax")
    ap.add_argument("--black", choices=["minimax", "nn", "random"], default="minimax")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--model", default=None, help="value_model.pt if --white or --black is nn")
    ap.add_argument("--out_pgn", default="data/datasets/arena.pgn")
    ap.add_argument("--db_path", default="data/games.db", help="Path to SQLite game database")
    ap.add_argument("--white_model_id", type=int, default=None, help="Model ID for white player")
    ap.add_argument("--black_model_id", type=int, default=None, help="Model ID for black player")
    args = ap.parse_args()

    # Initialize recorder
    recorder = GameRecorder(args.db_path)

    print(f"Arena: {args.games} games")
    print(f"  White: {args.white}")
    print(f"  Black: {args.black}")
    print(f"  Depth: {args.depth}")
    print(f"  Recording to: {args.db_path}")
    print()

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    games = []

    for i in range(args.games):
        # Alternate colors for fairness
        if i % 2 == 0:
            w, b = args.white, args.black
            w_model, b_model = args.white_model_id, args.black_model_id
        else:
            w, b = args.black, args.white
            w_model, b_model = args.black_model_id, args.white_model_id

        g, r = play(
            w, b, args.depth, args.model,
            recorder=recorder,
            white_model_id=w_model,
            black_model_id=b_model
        )
        results[r] += 1
        games.append(str(g))

        # Progress
        print(f"  Game {i + 1}/{args.games}: {w} vs {b} -> {r}")

    with open(args.out_pgn, "w", encoding="utf-8") as f:
        f.write("\n\n".join(games))

    print()
    print("=" * 40)
    print("Results:")
    print(f"  White wins: {results['1-0']}")
    print(f"  Black wins: {results['0-1']}")
    print(f"  Draws:      {results['1/2-1/2']}")
    print(f"Saved PGN to {args.out_pgn}")

    # Print database stats
    stats = recorder.get_statistics()
    print(f"\nDatabase: {stats['total_games']} total games, {stats['total_moves']} total moves")


if __name__ == "__main__":
    main()

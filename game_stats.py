#!/usr/bin/env python3
"""
Game Statistics CLI Tool

View and analyze recorded games from the database.

Usage:
    python game_stats.py                    # Show overall statistics
    python game_stats.py --recent 10        # Show 10 most recent games
    python game_stats.py --model 1          # Show stats for model ID 1
    python game_stats.py --export out.pgn   # Export games to PGN
    python game_stats.py --game 5           # Show details for game ID 5
    python game_stats.py --list-models      # List all registered models
"""

import argparse
import json
from datetime import datetime
from game_recorder import (
    GameRecorder, PlayerType, GameResult,
    ModelRecord, GameRecord
)


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_separator(char: str = "=", width: int = 60):
    """Print a separator line."""
    print(char * width)


def show_overall_stats(recorder: GameRecorder):
    """Display overall database statistics."""
    stats = recorder.get_statistics()

    print_separator()
    print("CHESS AI GAME DATABASE STATISTICS")
    print_separator()
    print()

    print(f"Total Games:       {stats['total_games']}")
    print(f"Completed Games:   {stats['completed_games']}")
    print(f"Total Moves:       {stats['total_moves']}")
    print(f"Unique Positions:  {stats['unique_positions']}")
    print(f"Average Game Length: {stats['avg_game_length']} moves")
    print(f"Total Models:      {stats['total_models']}")
    print()

    if stats['results']:
        print("Results Breakdown:")
        total = sum(stats['results'].values())
        for result, count in sorted(stats['results'].items()):
            pct = (count / total * 100) if total > 0 else 0
            result_name = {
                "1-0": "White Wins",
                "0-1": "Black Wins",
                "1/2-1/2": "Draws"
            }.get(result, result)
            print(f"  {result_name:12}: {count:5} ({pct:5.1f}%)")
        print()

    if stats['games_by_type']:
        print("Games by Player Type:")
        for entry in stats['games_by_type']:
            print(f"  {entry['white']:12} vs {entry['black']:12}: {entry['count']} games")
        print()


def show_recent_games(recorder: GameRecorder, limit: int = 10, player_type: PlayerType = None):
    """Display recent games."""
    games = recorder.get_recent_games(limit=limit, player_type=player_type)

    print_separator()
    print(f"RECENT GAMES (Last {limit})")
    print_separator()
    print()

    if not games:
        print("No games found.")
        return

    print(f"{'ID':>5} | {'White':^15} | {'Black':^15} | {'Result':^8} | {'Moves':>5} | {'Date'}")
    print("-" * 75)

    for game in games:
        if game:
            print(f"{game.game_id:>5} | {game.white_player:^15} | {game.black_player:^15} | "
                  f"{game.result.value:^8} | {game.total_moves:>5} | {format_datetime(game.start_time)}")
    print()


def show_game_details(recorder: GameRecorder, game_id: int):
    """Display detailed information about a specific game."""
    game = recorder.get_game(game_id)

    if not game:
        print(f"Game ID {game_id} not found.")
        return

    print_separator()
    print(f"GAME #{game_id} DETAILS")
    print_separator()
    print()

    print(f"Event:       {game.event}")
    print(f"White:       {game.white_player} ({game.white_type.value})")
    print(f"Black:       {game.black_player} ({game.black_type.value})")
    print(f"Result:      {game.result.value}")
    print(f"Termination: {game.termination or 'N/A'}")
    print(f"Total Moves: {game.total_moves}")
    print(f"Started:     {format_datetime(game.start_time)}")
    print(f"Ended:       {format_datetime(game.end_time)}")

    if game.white_depth:
        print(f"White Depth: {game.white_depth}")
    if game.black_depth:
        print(f"Black Depth: {game.black_depth}")

    print()

    # Show moves
    moves = recorder.get_game_moves(game_id)
    if moves:
        print("Moves:")
        print("-" * 60)

        move_strs = []
        for i, move in enumerate(moves):
            if i % 2 == 0:  # White's move
                move_strs.append(f"{move.move_number}. {move.san}")
            else:  # Black's move
                move_strs[-1] += f" {move.san}"

        # Print moves in rows of 5
        for i in range(0, len(move_strs), 5):
            row = move_strs[i:i+5]
            print("  " + "  ".join(f"{m:15}" for m in row))
        print()

        # Show timing stats
        times_white = [m.time_ms for m in moves if m.ply % 2 == 0 and m.time_ms]
        times_black = [m.time_ms for m in moves if m.ply % 2 == 1 and m.time_ms]

        if times_white:
            avg_white = sum(times_white) / len(times_white)
            print(f"White avg time/move: {avg_white:.0f}ms")
        if times_black:
            avg_black = sum(times_black) / len(times_black)
            print(f"Black avg time/move: {avg_black:.0f}ms")
        print()

    # Show PGN if available
    if game.pgn:
        print("PGN:")
        print("-" * 60)
        print(game.pgn)
        print()


def show_model_stats(recorder: GameRecorder, model_id: int):
    """Display statistics for a specific model."""
    model = recorder.get_model(model_id)

    if not model:
        print(f"Model ID {model_id} not found.")
        return

    print_separator()
    print(f"MODEL #{model_id} STATISTICS")
    print_separator()
    print()

    print(f"Name:         {model.name}")
    print(f"Version:      {model.version}")
    print(f"Architecture: {model.architecture or 'N/A'}")
    print(f"File Path:    {model.file_path or 'N/A'}")
    print(f"Created:      {format_datetime(model.created_at)}")

    if model.training_epochs:
        print(f"Training Epochs:    {model.training_epochs}")
    if model.training_games:
        print(f"Training Games:     {model.training_games}")
    if model.training_positions:
        print(f"Training Positions: {model.training_positions}")
    if model.training_loss:
        print(f"Training Loss:      {model.training_loss:.6f}")
    if model.elo_estimate:
        print(f"ELO Estimate:       {model.elo_estimate}")

    print()

    # Performance stats
    perf = recorder.get_model_performance(model_id)

    print("Performance:")
    print("-" * 40)
    print(f"Total Games:     {perf['total_games']}")
    print(f"Total Wins:      {perf['total_wins']}")
    print(f"Overall Win Rate: {perf['overall_win_rate']:.1f}%")
    print()

    if perf['as_white']:
        print("As White:")
        for result, count in perf['as_white'].items():
            print(f"  {result}: {count}")
        print(f"  Win Rate: {perf['white_win_rate']:.1f}%")
        print()

    if perf['as_black']:
        print("As Black:")
        for result, count in perf['as_black'].items():
            print(f"  {result}: {count}")
        print(f"  Win Rate: {perf['black_win_rate']:.1f}%")
        print()


def list_models(recorder: GameRecorder):
    """List all registered models."""
    conn = recorder._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    print_separator()
    print("REGISTERED MODELS")
    print_separator()
    print()

    if not rows:
        print("No models registered.")
        return

    print(f"{'ID':>4} | {'Name':^20} | {'Version':^10} | {'ELO':>6} | {'Created'}")
    print("-" * 70)

    for row in rows:
        elo = row['elo_estimate'] or '-'
        print(f"{row['model_id']:>4} | {row['name']:^20} | {row['version']:^10} | "
              f"{str(elo):>6} | {row['created_at']}")
    print()


def export_games(recorder: GameRecorder, output_path: str, model_id: int = None, limit: int = None):
    """Export games to PGN file."""
    count = recorder.export_games_to_pgn(
        output_path=output_path,
        model_id=model_id,
        limit=limit
    )

    print(f"Exported {count} games to {output_path}")


def register_current_model(recorder: GameRecorder, model_path: str):
    """Register the current value_model.pt if not already registered."""
    import os

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # Check if already registered by hash
    import hashlib
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()

    conn = recorder._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT model_id FROM models WHERE file_hash = ?", (file_hash,))
    existing = cursor.fetchone()
    conn.close()

    if existing:
        print(f"Model already registered with ID: {existing['model_id']}")
        return existing['model_id']

    # Register new model
    model_id = recorder.register_model(ModelRecord(
        name="ValueNet",
        version="auto",
        file_path=model_path,
        architecture="ValueNet_781_512_256_1"
    ))

    print(f"Registered model with ID: {model_id}")
    return model_id


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze chess game statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python game_stats.py                       # Show overall stats
    python game_stats.py --recent 20           # Show last 20 games
    python game_stats.py --game 5              # Details for game #5
    python game_stats.py --model 1             # Stats for model #1
    python game_stats.py --list-models         # List all models
    python game_stats.py --export games.pgn    # Export to PGN
    python game_stats.py --register-model value_model.pt
        """
    )

    parser.add_argument("--db", default="data/games.db",
                        help="Path to SQLite database (default: data/games.db)")
    parser.add_argument("--recent", type=int, metavar="N",
                        help="Show N most recent games")
    parser.add_argument("--game", type=int, metavar="ID",
                        help="Show details for a specific game ID")
    parser.add_argument("--model", type=int, metavar="ID",
                        help="Show statistics for a specific model ID")
    parser.add_argument("--list-models", action="store_true",
                        help="List all registered models")
    parser.add_argument("--export", metavar="FILE",
                        help="Export games to PGN file")
    parser.add_argument("--export-limit", type=int, metavar="N",
                        help="Limit number of games to export")
    parser.add_argument("--register-model", metavar="PATH",
                        help="Register a model file in the database")
    parser.add_argument("--human-games", action="store_true",
                        help="Filter to show only human vs AI games")
    parser.add_argument("--json", action="store_true",
                        help="Output statistics as JSON")

    args = parser.parse_args()
    recorder = GameRecorder(args.db)

    # Handle JSON output for stats
    if args.json and not any([args.recent, args.game, args.model, args.list_models, args.export]):
        stats = recorder.get_statistics()
        print(json.dumps(stats, indent=2))
        return

    # Handle specific commands
    if args.register_model:
        register_current_model(recorder, args.register_model)
    elif args.list_models:
        list_models(recorder)
    elif args.export:
        export_games(
            recorder, args.export,
            model_id=args.model,
            limit=args.export_limit
        )
    elif args.game:
        show_game_details(recorder, args.game)
    elif args.model:
        show_model_stats(recorder, args.model)
    elif args.recent:
        player_type = PlayerType.HUMAN if args.human_games else None
        show_recent_games(recorder, args.recent, player_type)
    else:
        # Default: show overall statistics
        show_overall_stats(recorder)


if __name__ == "__main__":
    main()

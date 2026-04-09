"""
Game Recording System for Chess AI Training

This module provides a comprehensive system for recording all chess games
(AI vs AI, Human vs AI) with detailed metadata for reinforcement learning.

Features:
- SQLite database storage for structured queries
- Model versioning and lineage tracking
- Per-move statistics (evaluation, time, depth)
- Game metadata (players, result, timestamps)
- Query interface for training data extraction
- Statistics and analysis utilities
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import chess
import chess.pgn


class PlayerType(Enum):
    """Type of player in a game."""
    HUMAN = "human"
    AI_MINIMAX = "ai_minimax"
    AI_NN = "ai_nn"
    AI_RANDOM = "ai_random"


class GameResult(Enum):
    """Possible game outcomes."""
    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    ONGOING = "*"


@dataclass
class MoveRecord:
    """Record of a single move in a game."""
    move_number: int
    ply: int  # Half-move number (0-indexed)
    uci: str  # Move in UCI format (e.g., "e2e4")
    san: str  # Move in SAN format (e.g., "e4")
    fen_before: str  # Position before the move
    fen_after: str  # Position after the move
    evaluation: Optional[float] = None  # Position evaluation
    depth: Optional[int] = None  # Search depth used
    time_ms: Optional[int] = None  # Time taken for move in milliseconds
    is_capture: bool = False
    is_check: bool = False
    is_checkmate: bool = False


@dataclass
class GameRecord:
    """Complete record of a chess game."""
    game_id: Optional[int] = None
    white_player: str = ""
    black_player: str = ""
    white_type: PlayerType = PlayerType.HUMAN
    black_type: PlayerType = PlayerType.HUMAN
    white_model_id: Optional[int] = None  # Reference to model version
    black_model_id: Optional[int] = None
    white_depth: Optional[int] = None
    black_depth: Optional[int] = None
    result: GameResult = GameResult.ONGOING
    termination: Optional[str] = None  # e.g., "CHECKMATE", "STALEMATE"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_moves: int = 0
    pgn: str = ""
    opening_eco: Optional[str] = None  # ECO code if identified
    opening_name: Optional[str] = None
    event: str = "Training"
    notes: Optional[str] = None


@dataclass
class ModelRecord:
    """Record of a trained model version."""
    model_id: Optional[int] = None
    name: str = ""
    version: str = ""
    file_path: str = ""
    file_hash: str = ""  # SHA256 hash for integrity
    parent_model_id: Optional[int] = None  # For tracking lineage
    architecture: str = ""  # e.g., "ValueNet_781_512_256_1"
    training_games: int = 0
    training_positions: int = 0
    training_epochs: int = 0
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    elo_estimate: Optional[int] = None
    created_at: Optional[datetime] = None
    notes: Optional[str] = None


class GameRecorder:
    """
    Comprehensive game recording system for chess AI training.

    Usage:
        recorder = GameRecorder("data/games.db")

        # Start recording a game
        game_id = recorder.start_game(
            white_player="human",
            black_player="nn_v1",
            white_type=PlayerType.HUMAN,
            black_type=PlayerType.AI_NN,
            black_model_id=1,
            black_depth=3
        )

        # Record each move
        recorder.record_move(game_id, move_record)

        # End the game
        recorder.end_game(game_id, result=GameResult.WHITE_WINS, termination="CHECKMATE")
    """

    def __init__(self, db_path: str = "data/games.db"):
        """Initialize the game recorder with a SQLite database."""
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Models table - tracks all model versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                file_path TEXT,
                file_hash TEXT,
                parent_model_id INTEGER,
                architecture TEXT,
                training_games INTEGER DEFAULT 0,
                training_positions INTEGER DEFAULT 0,
                training_epochs INTEGER DEFAULT 0,
                training_loss REAL,
                validation_loss REAL,
                elo_estimate INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (parent_model_id) REFERENCES models(model_id)
            )
        """)

        # Games table - stores game metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                white_player TEXT NOT NULL,
                black_player TEXT NOT NULL,
                white_type TEXT NOT NULL,
                black_type TEXT NOT NULL,
                white_model_id INTEGER,
                black_model_id INTEGER,
                white_depth INTEGER,
                black_depth INTEGER,
                result TEXT DEFAULT '*',
                termination TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_moves INTEGER DEFAULT 0,
                pgn TEXT,
                opening_eco TEXT,
                opening_name TEXT,
                event TEXT DEFAULT 'Training',
                notes TEXT,
                FOREIGN KEY (white_model_id) REFERENCES models(model_id),
                FOREIGN KEY (black_model_id) REFERENCES models(model_id)
            )
        """)

        # Moves table - stores individual moves with evaluations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moves (
                move_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                move_number INTEGER NOT NULL,
                ply INTEGER NOT NULL,
                uci TEXT NOT NULL,
                san TEXT NOT NULL,
                fen_before TEXT NOT NULL,
                fen_after TEXT NOT NULL,
                evaluation REAL,
                depth INTEGER,
                time_ms INTEGER,
                is_capture BOOLEAN DEFAULT 0,
                is_check BOOLEAN DEFAULT 0,
                is_checkmate BOOLEAN DEFAULT 0,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            )
        """)

        # Training runs table - tracks training sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                dataset_path TEXT,
                num_games INTEGER,
                num_positions INTEGER,
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                final_loss REAL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_result ON games(result)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_white_type ON games(white_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_black_type ON games(black_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_start_time ON games(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)")

        conn.commit()
        conn.close()

    # ==================== MODEL MANAGEMENT ====================

    def register_model(self, record: ModelRecord) -> int:
        """Register a new model version in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Calculate file hash if file exists
        if record.file_path and os.path.exists(record.file_path):
            record.file_hash = self._compute_file_hash(record.file_path)

        cursor.execute("""
            INSERT INTO models (
                name, version, file_path, file_hash, parent_model_id,
                architecture, training_games, training_positions, training_epochs,
                training_loss, validation_loss, elo_estimate, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.name, record.version, record.file_path, record.file_hash,
            record.parent_model_id, record.architecture, record.training_games,
            record.training_positions, record.training_epochs, record.training_loss,
            record.validation_loss, record.elo_estimate, record.notes
        ))

        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return model_id

    def get_model(self, model_id: int) -> Optional[ModelRecord]:
        """Retrieve a model record by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return ModelRecord(
                model_id=row["model_id"],
                name=row["name"],
                version=row["version"],
                file_path=row["file_path"],
                file_hash=row["file_hash"],
                parent_model_id=row["parent_model_id"],
                architecture=row["architecture"],
                training_games=row["training_games"],
                training_positions=row["training_positions"],
                training_epochs=row["training_epochs"],
                training_loss=row["training_loss"],
                validation_loss=row["validation_loss"],
                elo_estimate=row["elo_estimate"],
                created_at=row["created_at"],
                notes=row["notes"]
            )
        return None

    def get_latest_model(self, name: str = None) -> Optional[ModelRecord]:
        """Get the most recently created model, optionally filtered by name."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if name:
            cursor.execute(
                "SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,)
            )
        else:
            cursor.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 1")

        row = cursor.fetchone()
        conn.close()

        if row:
            return self.get_model(row["model_id"])
        return None

    def update_model_elo(self, model_id: int, elo: int):
        """Update the ELO estimate for a model."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE models SET elo_estimate = ? WHERE model_id = ?",
            (elo, model_id)
        )
        conn.commit()
        conn.close()

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ==================== GAME RECORDING ====================

    def start_game(
        self,
        white_player: str,
        black_player: str,
        white_type: PlayerType,
        black_type: PlayerType,
        white_model_id: Optional[int] = None,
        black_model_id: Optional[int] = None,
        white_depth: Optional[int] = None,
        black_depth: Optional[int] = None,
        event: str = "Training"
    ) -> int:
        """Start recording a new game. Returns the game_id."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO games (
                white_player, black_player, white_type, black_type,
                white_model_id, black_model_id, white_depth, black_depth,
                event, start_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            white_player, black_player, white_type.value, black_type.value,
            white_model_id, black_model_id, white_depth, black_depth,
            event, datetime.now()
        ))

        game_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return game_id

    def record_move(
        self,
        game_id: int,
        board: chess.Board,
        move: chess.Move,
        evaluation: Optional[float] = None,
        depth: Optional[int] = None,
        time_ms: Optional[int] = None
    ):
        """Record a single move for a game."""
        conn = self._get_connection()
        cursor = conn.cursor()

        fen_before = board.fen()
        san = board.san(move)
        is_capture = board.is_capture(move)

        # Apply the move to get fen_after and check status
        board_copy = board.copy()
        board_copy.push(move)
        fen_after = board_copy.fen()
        is_check = board_copy.is_check()
        is_checkmate = board_copy.is_checkmate()

        # Calculate move number and ply
        ply = board.ply()
        move_number = (ply // 2) + 1

        cursor.execute("""
            INSERT INTO moves (
                game_id, move_number, ply, uci, san, fen_before, fen_after,
                evaluation, depth, time_ms, is_capture, is_check, is_checkmate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, move_number, ply, move.uci(), san, fen_before, fen_after,
            evaluation, depth, time_ms, is_capture, is_check, is_checkmate
        ))

        conn.commit()
        conn.close()

    def end_game(
        self,
        game_id: int,
        result: GameResult,
        termination: Optional[str] = None,
        pgn: Optional[str] = None
    ):
        """Finalize a game record with the result."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Count total moves
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM moves WHERE game_id = ?",
            (game_id,)
        )
        total_moves = cursor.fetchone()["cnt"]

        cursor.execute("""
            UPDATE games SET
                result = ?,
                termination = ?,
                end_time = ?,
                total_moves = ?,
                pgn = ?
            WHERE game_id = ?
        """, (
            result.value, termination, datetime.now(), total_moves, pgn, game_id
        ))

        conn.commit()
        conn.close()

    def get_game(self, game_id: int) -> Optional[GameRecord]:
        """Retrieve a game record by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return GameRecord(
                game_id=row["game_id"],
                white_player=row["white_player"],
                black_player=row["black_player"],
                white_type=PlayerType(row["white_type"]),
                black_type=PlayerType(row["black_type"]),
                white_model_id=row["white_model_id"],
                black_model_id=row["black_model_id"],
                white_depth=row["white_depth"],
                black_depth=row["black_depth"],
                result=GameResult(row["result"]),
                termination=row["termination"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                total_moves=row["total_moves"],
                pgn=row["pgn"],
                opening_eco=row["opening_eco"],
                opening_name=row["opening_name"],
                event=row["event"],
                notes=row["notes"]
            )
        return None

    def get_game_moves(self, game_id: int) -> List[MoveRecord]:
        """Get all moves for a game in order."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM moves WHERE game_id = ? ORDER BY ply",
            (game_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            MoveRecord(
                move_number=row["move_number"],
                ply=row["ply"],
                uci=row["uci"],
                san=row["san"],
                fen_before=row["fen_before"],
                fen_after=row["fen_after"],
                evaluation=row["evaluation"],
                depth=row["depth"],
                time_ms=row["time_ms"],
                is_capture=bool(row["is_capture"]),
                is_check=bool(row["is_check"]),
                is_checkmate=bool(row["is_checkmate"])
            )
            for row in rows
        ]

    # ==================== TRAINING SUPPORT ====================

    def record_training_run(
        self,
        model_id: int,
        dataset_path: str,
        num_games: int,
        num_positions: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        final_loss: Optional[float] = None,
        notes: Optional[str] = None
    ) -> int:
        """Record a training run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO training_runs (
                model_id, dataset_path, num_games, num_positions,
                epochs, batch_size, learning_rate, final_loss, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, dataset_path, num_games, num_positions,
            epochs, batch_size, learning_rate, final_loss, notes
        ))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id

    def complete_training_run(self, run_id: int, final_loss: float):
        """Mark a training run as complete with final loss."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE training_runs SET
                final_loss = ?,
                completed_at = ?
            WHERE run_id = ?
        """, (final_loss, datetime.now(), run_id))
        conn.commit()
        conn.close()

    def get_training_positions(
        self,
        model_id: Optional[int] = None,
        player_type: Optional[PlayerType] = None,
        result: Optional[GameResult] = None,
        limit: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract training positions (FEN, target) from recorded games.

        Returns list of (fen, target) where target is:
            +1.0 for white win, -1.0 for black win, 0.0 for draw
            Adjusted for side-to-move perspective
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build query with optional filters
        query = """
            SELECT m.fen_before, g.result, m.ply
            FROM moves m
            JOIN games g ON m.game_id = g.game_id
            WHERE g.result != '*'
        """
        params = []

        if model_id:
            query += " AND (g.white_model_id = ? OR g.black_model_id = ?)"
            params.extend([model_id, model_id])

        if player_type:
            query += " AND (g.white_type = ? OR g.black_type = ?)"
            params.extend([player_type.value, player_type.value])

        if result:
            query += " AND g.result = ?"
            params.append(result.value)

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            fen = row["fen_before"]
            game_result = row["result"]
            ply = row["ply"]

            # Determine target value
            if game_result == "1-0":
                z = 1.0
            elif game_result == "0-1":
                z = -1.0
            else:
                z = 0.0

            # Adjust for side-to-move (ply % 2 == 0 means white to move)
            if ply % 2 == 1:  # Black to move
                z = -z

            positions.append((fen, z))

        return positions

    # ==================== STATISTICS & QUERIES ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Total games
        cursor.execute("SELECT COUNT(*) as cnt FROM games")
        stats["total_games"] = cursor.fetchone()["cnt"]

        # Completed games
        cursor.execute("SELECT COUNT(*) as cnt FROM games WHERE result != '*'")
        stats["completed_games"] = cursor.fetchone()["cnt"]

        # Total moves
        cursor.execute("SELECT COUNT(*) as cnt FROM moves")
        stats["total_moves"] = cursor.fetchone()["cnt"]

        # Total positions (unique FENs)
        cursor.execute("SELECT COUNT(DISTINCT fen_before) as cnt FROM moves")
        stats["unique_positions"] = cursor.fetchone()["cnt"]

        # Results breakdown
        cursor.execute("""
            SELECT result, COUNT(*) as cnt
            FROM games
            WHERE result != '*'
            GROUP BY result
        """)
        stats["results"] = {row["result"]: row["cnt"] for row in cursor.fetchall()}

        # Games by player type
        cursor.execute("""
            SELECT white_type, black_type, COUNT(*) as cnt
            FROM games
            GROUP BY white_type, black_type
        """)
        stats["games_by_type"] = [
            {"white": row["white_type"], "black": row["black_type"], "count": row["cnt"]}
            for row in cursor.fetchall()
        ]

        # Model count
        cursor.execute("SELECT COUNT(*) as cnt FROM models")
        stats["total_models"] = cursor.fetchone()["cnt"]

        # Average game length
        cursor.execute("""
            SELECT AVG(total_moves) as avg_moves
            FROM games
            WHERE result != '*'
        """)
        avg = cursor.fetchone()["avg_moves"]
        stats["avg_game_length"] = round(avg, 1) if avg else 0

        conn.close()
        return stats

    def get_model_performance(self, model_id: int) -> Dict[str, Any]:
        """Get performance statistics for a specific model."""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {"model_id": model_id, "as_white": {}, "as_black": {}}

        # Performance as white
        cursor.execute("""
            SELECT result, COUNT(*) as cnt
            FROM games
            WHERE white_model_id = ? AND result != '*'
            GROUP BY result
        """, (model_id,))
        stats["as_white"] = {row["result"]: row["cnt"] for row in cursor.fetchall()}

        # Performance as black
        cursor.execute("""
            SELECT result, COUNT(*) as cnt
            FROM games
            WHERE black_model_id = ? AND result != '*'
            GROUP BY result
        """, (model_id,))
        stats["as_black"] = {row["result"]: row["cnt"] for row in cursor.fetchall()}

        # Calculate win rates
        white_games = sum(stats["as_white"].values()) if stats["as_white"] else 0
        black_games = sum(stats["as_black"].values()) if stats["as_black"] else 0

        white_wins = stats["as_white"].get("1-0", 0)
        black_wins = stats["as_black"].get("0-1", 0)

        stats["white_win_rate"] = (white_wins / white_games * 100) if white_games > 0 else 0
        stats["black_win_rate"] = (black_wins / black_games * 100) if black_games > 0 else 0
        stats["total_games"] = white_games + black_games
        stats["total_wins"] = white_wins + black_wins
        stats["overall_win_rate"] = (
            (white_wins + black_wins) / (white_games + black_games) * 100
            if (white_games + black_games) > 0 else 0
        )

        conn.close()
        return stats

    def get_recent_games(
        self,
        limit: int = 10,
        player_type: Optional[PlayerType] = None
    ) -> List[GameRecord]:
        """Get the most recent games."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT game_id FROM games"
        params = []

        if player_type:
            query += " WHERE white_type = ? OR black_type = ?"
            params.extend([player_type.value, player_type.value])

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        game_ids = [row["game_id"] for row in cursor.fetchall()]
        conn.close()

        return [self.get_game(gid) for gid in game_ids]

    def export_games_to_pgn(
        self,
        output_path: str,
        model_id: Optional[int] = None,
        player_type: Optional[PlayerType] = None,
        limit: Optional[int] = None
    ) -> int:
        """Export games to a PGN file. Returns number of games exported."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT game_id, pgn FROM games WHERE pgn IS NOT NULL AND pgn != ''"
        params = []

        if model_id:
            query += " AND (white_model_id = ? OR black_model_id = ?)"
            params.extend([model_id, model_id])

        if player_type:
            query += " AND (white_type = ? OR black_type = ?)"
            params.extend([player_type.value, player_type.value])

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        with open(output_path, "w", encoding="utf-8") as f:
            pgns = [row["pgn"] for row in rows if row["pgn"]]
            f.write("\n\n".join(pgns))

        return len(rows)


# ==================== CONVENIENCE FUNCTIONS ====================

# Global recorder instance (lazy initialization)
_global_recorder: Optional[GameRecorder] = None


def get_recorder(db_path: str = "data/games.db") -> GameRecorder:
    """Get or create the global GameRecorder instance."""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = GameRecorder(db_path)
    return _global_recorder


def player_type_from_engine(engine_name: str) -> PlayerType:
    """Convert engine name string to PlayerType enum."""
    mapping = {
        "human": PlayerType.HUMAN,
        "minimax": PlayerType.AI_MINIMAX,
        "nn": PlayerType.AI_NN,
        "random": PlayerType.AI_RANDOM,
    }
    return mapping.get(engine_name.lower(), PlayerType.HUMAN)


def result_from_string(result_str: str) -> GameResult:
    """Convert result string to GameResult enum."""
    mapping = {
        "1-0": GameResult.WHITE_WINS,
        "0-1": GameResult.BLACK_WINS,
        "1/2-1/2": GameResult.DRAW,
        "*": GameResult.ONGOING,
    }
    return mapping.get(result_str, GameResult.ONGOING)


if __name__ == "__main__":
    # Quick test/demo
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_games.db")
        recorder = GameRecorder(db_path)

        # Register a model
        model_id = recorder.register_model(ModelRecord(
            name="ValueNet",
            version="1.0",
            architecture="ValueNet_781_512_256_1",
            training_epochs=50,
            notes="Initial test model"
        ))
        print(f"Registered model with ID: {model_id}")

        # Start a game
        game_id = recorder.start_game(
            white_player="human_player",
            black_player="nn_v1",
            white_type=PlayerType.HUMAN,
            black_type=PlayerType.AI_NN,
            black_model_id=model_id,
            black_depth=3,
            event="Test Game"
        )
        print(f"Started game with ID: {game_id}")

        # Simulate some moves
        board = chess.Board()
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        for uci in moves:
            move = chess.Move.from_uci(uci)
            recorder.record_move(game_id, board, move, evaluation=0.1, depth=3)
            board.push(move)

        # End the game
        recorder.end_game(game_id, GameResult.DRAW, termination="AGREEMENT")

        # Get statistics
        stats = recorder.get_statistics()
        print(f"\nStatistics: {json.dumps(stats, indent=2)}")

        # Get game moves
        moves = recorder.get_game_moves(game_id)
        print(f"\nRecorded {len(moves)} moves")
        for m in moves:
            print(f"  {m.move_number}. {m.san} (eval: {m.evaluation})")

        print("\nGame recorder test completed successfully!")

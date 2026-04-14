"""
Flask Web Server for Chess AI

Serves a web interface where users can play against the chess AI.
Uses the MCTS engine with the trained dual neural network.
"""

import uuid
import time
import chess
from flask import Flask, jsonify, request, render_template, send_from_directory

from neural_network import load_dual_model
from mcts import MCTSPlayer, MCTSConfig
from evaluation import evaluate as hc_evaluate

# ---------------------------------------------------------------------------
# App & AI initialisation
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Load the trained model once at startup
MODEL_PATH = "models/dual_model_mcts.pt"

# Difficulty presets: name -> simulations
DIFFICULTY_PRESETS = {
    "easy": 300,
    "medium": 800,
    "hard": 1600,
}

print("Loading chess AI model...")
try:
    dual_model = load_dual_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not load model ({e}). AI will use uniform policy.")
    dual_model = None


def _make_player(num_sims: int) -> MCTSPlayer:
    """Create a fresh MCTS player (new search tree each time to avoid memory bloat)."""
    config = MCTSConfig(
        num_simulations=num_sims,
        temperature=0,
        add_noise=False,
    )
    return MCTSPlayer(model=dual_model, config=config)

# ---------------------------------------------------------------------------
# In-memory game sessions
# ---------------------------------------------------------------------------

games: dict[str, chess.Board] = {}
game_histories: dict[str, list[dict]] = {}
game_difficulty: dict[str, str] = {}
completed_games: list[dict] = []  # Archive of finished games


def _get_evaluation(board: chess.Board) -> float:
    """Get position evaluation from White's perspective (-1 to +1)."""
    val = hc_evaluate(board)
    # hc_evaluate returns from current player's perspective
    # Convert to White's perspective for the UI
    if board.turn == chess.BLACK:
        val = -val
    return round(val, 3)


def board_to_json(board: chess.Board, game_id: str, ai_time_ms: int = 0) -> dict:
    """Serialise the current board state for the frontend."""
    status = "playing"
    result = None

    if board.is_game_over():
        result = board.result()
        outcome = board.outcome()
        if board.is_checkmate():
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            status = f"checkmate — {winner} wins"
        elif board.is_stalemate():
            status = "draw — stalemate"
        elif board.is_insufficient_material():
            status = "draw — insufficient material"
        elif board.is_fifty_moves():
            status = "draw — fifty-move rule"
        elif board.is_repetition():
            status = "draw — repetition"
        else:
            status = "game over"
    elif board.is_check():
        status = "check"

    return {
        "game_id": game_id,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "is_game_over": board.is_game_over(),
        "status": status,
        "result": result,
        "move_number": board.fullmove_number,
        "history": game_histories.get(game_id, []),
        "legal_moves": [m.uci() for m in board.legal_moves],
        "evaluation": _get_evaluation(board),
        "ai_time_ms": ai_time_ms,
        "difficulty": game_difficulty.get(game_id, "hard"),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


_PIECE_FILE_MAP = {
    "wP": "w_pawn.png",   "wR": "w_rook.png",  "wN": "w_knight.png",
    "wB": "w_bishop.png", "wQ": "w_queen.png",  "wK": "w_king.png",
    "bP": "b_pawn.png",   "bR": "b_rook.png",   "bN": "b_knight.png",
    "bB": "b_bishop.png", "bQ": "b_queen.png",   "bK": "b_king.png",
}

@app.route("/img/pieces/<piece>.png")
def piece_image(piece):
    filename = _PIECE_FILE_MAP.get(piece)
    if filename is None:
        return "Not found", 404
    return send_from_directory("data/imgs", filename)


@app.route("/api/new-game", methods=["POST"])
def new_game():
    """Start a new game. Accepts optional {"difficulty": "easy|medium|hard"}."""
    data = request.get_json(silent=True) or {}
    difficulty = data.get("difficulty", "hard")
    if difficulty not in DIFFICULTY_PRESETS:
        difficulty = "hard"

    # Archive any existing games that haven't been archived yet
    for old_id, old_board in list(games.items()):
        if old_id not in [g["game_id"] for g in completed_games]:
            _archive_game(old_id, old_board)

    game_id = uuid.uuid4().hex[:12]
    board = chess.Board()
    games[game_id] = board
    game_histories[game_id] = []
    game_difficulty[game_id] = difficulty
    return jsonify(board_to_json(board, game_id))


@app.route("/api/move", methods=["POST"])
def make_move():
    """
    Accept a human move, apply it, then compute and apply the AI reply.

    Expects JSON: {"game_id": "...", "move": "e2e4"}
    """
    data = request.get_json(force=True)
    game_id = data.get("game_id")
    uci_str = data.get("move", "")

    board = games.get(game_id)
    if board is None:
        return jsonify({"error": "Game not found"}), 404

    if board.is_game_over():
        return jsonify({"error": "Game is already over", **board_to_json(board, game_id)}), 400

    # --- Validate & apply human move ---
    try:
        human_move = chess.Move.from_uci(uci_str)
    except ValueError:
        return jsonify({"error": f"Invalid UCI string: {uci_str}"}), 400

    if human_move not in board.legal_moves:
        for promo in ["q", "r", "b", "n"]:
            promo_move = chess.Move.from_uci(uci_str[:4] + promo)
            if promo_move in board.legal_moves:
                human_move = promo_move
                break
        else:
            return jsonify({"error": f"Illegal move: {uci_str}"}), 400

    san_human = board.san(human_move)
    board.push(human_move)
    game_histories[game_id].append({
        "color": "white",
        "san": san_human,
        "uci": human_move.uci(),
    })

    if board.is_game_over():
        _archive_game(game_id, board)
        return jsonify(board_to_json(board, game_id))

    # --- Check if AI should resign (hopelessly behind) ---
    from mcts import calculate_material
    material = calculate_material(board)
    # material is from White's perspective; AI is Black, so negative = AI losing
    ai_material_deficit = material  # positive means White (human) is ahead
    if ai_material_deficit >= 15.0 and board.fullmove_number >= 20:
        # AI is down 15+ pawns worth (e.g., 2 queens) after move 20 — resign
        game_histories[game_id].append({
            "color": "black",
            "san": "resigns",
            "uci": "",
        })
        resp = board_to_json(board, game_id)
        resp["is_game_over"] = True
        resp["status"] = "Black resigns — White wins"
        resp["result"] = "1-0"
        _archive_game(game_id, board)
        return jsonify(resp)

    # --- AI reply (fresh player per move to avoid memory bloat from stale trees) ---
    difficulty = game_difficulty.get(game_id, "hard")
    num_sims = DIFFICULTY_PRESETS.get(difficulty, 300)

    # Boost sims in lone-king endgames to find checkmate faster
    opponent_pieces = sum(1 for sq in chess.SQUARES
                         if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
                         and board.piece_at(sq).piece_type != chess.KING)
    if opponent_pieces == 0:
        num_sims = max(num_sims, 800)  # Lone king: search deeper for mate

    player = _make_player(num_sims)

    start = time.time()
    ai_move = player.select_move(board)
    elapsed_ms = int((time.time() - start) * 1000)

    if ai_move is None:
        _archive_game(game_id, board)
        return jsonify(board_to_json(board, game_id, elapsed_ms))

    san_ai = board.san(ai_move)
    board.push(ai_move)
    game_histories[game_id].append({
        "color": "black",
        "san": san_ai,
        "uci": ai_move.uci(),
        "time_ms": elapsed_ms,
    })

    if board.is_game_over():
        _archive_game(game_id, board)

    return jsonify(board_to_json(board, game_id, elapsed_ms))


@app.route("/api/state", methods=["GET"])
def get_state():
    game_id = request.args.get("game_id")
    board = games.get(game_id)
    if board is None:
        return jsonify({"error": "Game not found"}), 404
    return jsonify(board_to_json(board, game_id))


def _archive_game(game_id: str, board: chess.Board):
    """Save a completed game to the archive."""
    if any(g["game_id"] == game_id for g in completed_games):
        return  # Already archived
    result = board.result() if board.is_game_over() else "*"
    outcome = board.outcome()
    completed_games.append({
        "game_id": game_id,
        "result": result,
        "termination": outcome.termination.name if outcome else None,
        "difficulty": game_difficulty.get(game_id, "hard"),
        "history": game_histories.get(game_id, []),
        "fen": board.fen(),
        "total_moves": len(game_histories.get(game_id, [])),
    })
    print(f"Game archived: {game_id} result={result}")


@app.route("/api/games", methods=["GET"])
def list_games():
    """List all completed games."""
    return jsonify({"games": completed_games})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Chess AI Web Server ===")
    print("Open http://localhost:5000 in your browser to play!\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

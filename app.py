"""
Flask Web Server for Chess AI

Serves a web interface where users can play against the chess AI.
Uses the MCTS engine with the trained dual neural network.
"""

import uuid
import time
import os
import chess
from flask import Flask, jsonify, request, render_template, send_from_directory

from neural_network import load_dual_model
from mcts import MCTSPlayer, MCTSConfig

# ---------------------------------------------------------------------------
# App & AI initialisation
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Load the trained model once at startup
MODEL_PATH = "models/dual_model_mcts.pt"
DEFAULT_SIMS = 300  # simulations per move (balance speed vs strength)

print("Loading chess AI model...")
try:
    dual_model = load_dual_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not load model ({e}). AI will use uniform policy.")
    dual_model = None

mcts_config = MCTSConfig(
    num_simulations=DEFAULT_SIMS,
    temperature=0,  # Greedy — always pick the best move
    add_noise=False,
)
mcts_player = MCTSPlayer(model=dual_model, config=mcts_config)

# ---------------------------------------------------------------------------
# In-memory game sessions  {game_id: chess.Board}
# ---------------------------------------------------------------------------

games: dict[str, chess.Board] = {}
game_histories: dict[str, list[dict]] = {}


def board_to_json(board: chess.Board, game_id: str) -> dict:
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
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# Map chessboard.js piece codes (e.g. "wP") to our local image filenames
_PIECE_FILE_MAP = {
    "wP": "w_pawn.png",   "wR": "w_rook.png",  "wN": "w_knight.png",
    "wB": "w_bishop.png", "wQ": "w_queen.png",  "wK": "w_king.png",
    "bP": "b_pawn.png",   "bR": "b_rook.png",   "bN": "b_knight.png",
    "bB": "b_bishop.png", "bQ": "b_queen.png",   "bK": "b_king.png",
}

@app.route("/img/pieces/<piece>.png")
def piece_image(piece):
    """Serve chess piece images, mapping chessboard.js names to local files."""
    filename = _PIECE_FILE_MAP.get(piece)
    if filename is None:
        return "Not found", 404
    return send_from_directory("data/imgs", filename)


@app.route("/api/new-game", methods=["POST"])
def new_game():
    """Start a new game. Returns the initial board state."""
    game_id = uuid.uuid4().hex[:12]
    board = chess.Board()
    games[game_id] = board
    game_histories[game_id] = []
    return jsonify(board_to_json(board, game_id))


@app.route("/api/move", methods=["POST"])
def make_move():
    """
    Accept a human move, apply it, then compute and apply the AI reply.

    Expects JSON: {"game_id": "...", "move": "e2e4"}
    Returns the updated board state (after both moves).
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
        # Try with promotion variants
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

    # --- Check if game ended after human move ---
    if board.is_game_over():
        return jsonify(board_to_json(board, game_id))

    # --- AI reply ---
    start = time.time()
    ai_move = mcts_player.select_move(board)
    elapsed_ms = int((time.time() - start) * 1000)

    if ai_move is None:
        return jsonify(board_to_json(board, game_id))

    san_ai = board.san(ai_move)
    board.push(ai_move)
    game_histories[game_id].append({
        "color": "black",
        "san": san_ai,
        "uci": ai_move.uci(),
        "time_ms": elapsed_ms,
    })

    return jsonify(board_to_json(board, game_id))


@app.route("/api/state", methods=["GET"])
def get_state():
    """Get the current state of a game."""
    game_id = request.args.get("game_id")
    board = games.get(game_id)
    if board is None:
        return jsonify({"error": "Game not found"}), 404
    return jsonify(board_to_json(board, game_id))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Chess AI Web Server ===")
    print("Open http://localhost:5000 in your browser to play!\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

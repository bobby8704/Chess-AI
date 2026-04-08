import pygame
import time
import sys
import argparse
import chess
import chess.pgn

from data.classes.Board import Board
from ai_nn import load_value_model, best_move_nn
from ai import best_move_using_minimax
from game_recorder import (
    GameRecorder, PlayerType, GameResult, result_from_string
)


def sync_pygame_from_chess(pygame_board, chess_board):
    """
    Sync the Pygame board state to match python-chess board.
    This ensures both boards are always in sync, preventing illegal states.
    """
    # Clear all pieces and highlights from pygame board
    for square in pygame_board.squares:
        square.occupying_piece = None
        square.highlight = False

    # Import piece classes
    from data.classes.pieces.Pawn import Pawn
    from data.classes.pieces.Rook import Rook
    from data.classes.pieces.Knight import Knight
    from data.classes.pieces.Bishop import Bishop
    from data.classes.pieces.Queen import Queen
    from data.classes.pieces.King import King

    piece_map = {
        chess.PAWN: Pawn,
        chess.ROOK: Rook,
        chess.KNIGHT: Knight,
        chess.BISHOP: Bishop,
        chess.QUEEN: Queen,
        chess.KING: King,
    }

    # Place pieces from chess_board onto pygame board
    for square_idx in chess.SQUARES:
        piece = chess_board.piece_at(square_idx)
        if piece:
            file = chess.square_file(square_idx)
            rank = 7 - chess.square_rank(square_idx)  # Flip rank for pygame coords
            pos = (file, rank)
            color = 'white' if piece.color == chess.WHITE else 'black'
            piece_class = piece_map[piece.piece_type]
            pygame_square = pygame_board.get_square_from_pos(pos)
            pygame_square.occupying_piece = piece_class(pos, color, pygame_board)

    # Set the turn
    pygame_board.turn = 'white' if chess_board.turn == chess.WHITE else 'black'
    pygame_board.selected_piece = None
    # Ensure chess_board reference is maintained
    pygame_board.chess_board = chess_board


# Parse command line arguments
parser = argparse.ArgumentParser(description="Chess AI Game")
parser.add_argument("--load_model", type=str, default=None, help="Path to model file")
parser.add_argument("--engine", type=str, default="mcts", choices=["nn", "mcts", "minimax"],
                    help="AI engine to use")
parser.add_argument("--depth", type=int, default=2, help="Search depth for nn/minimax")
parser.add_argument("--sims", type=int, default=100, help="MCTS simulations")
args, _ = parser.parse_known_args()

pygame.init()

WINDOW_SIZE = (600, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Chess AI")

# Initialize game state
board = Board(*WINDOW_SIZE)
chess_board = chess.Board()  # keep python-chess in sync
board.chess_board = chess_board  # Link for accurate move highlighting

# Initialize game recorder
recorder = GameRecorder("data/games.db")
game_id = None  # Will be set when game starts
AI_DEPTH = args.depth
AI_ENGINE = args.engine
MCTS_SIMS = args.sims

# AI engine globals
mcts_player = None
model_loaded = False

# Load model based on engine type
if AI_ENGINE == "mcts":
    model_path = args.load_model or "models/dual_model_mcts.pt"
    try:
        from neural_network import load_dual_model
        from mcts import MCTSPlayer, MCTSConfig

        dual_model = load_dual_model(model_path)
        config = MCTSConfig(
            num_simulations=MCTS_SIMS,
            temperature=0.1,  # Low temperature for strong play
            add_noise=False
        )
        mcts_player = MCTSPlayer(model=dual_model, config=config)
        model_loaded = True
        print(f"Loaded MCTS model: {model_path}")
    except Exception as e:
        print(f"Note: couldn't load MCTS model — using uniform policy. Details: {e}")
        from mcts import MCTSPlayer, MCTSConfig
        config = MCTSConfig(num_simulations=MCTS_SIMS, temperature=0.1, add_noise=False)
        mcts_player = MCTSPlayer(model=None, config=config)
else:
    # Load value model for nn/minimax engine
    model_path = args.load_model or "value_model.pt"
    try:
        load_value_model(model_path)
        model_loaded = True
        print(f"Loaded {model_path}")
    except Exception as e:
        print(f"Note: couldn't load {model_path} — NN will evaluate neutral. Details:", e)


def get_ai_move(chess_board):
    """Get AI move based on selected engine."""
    if AI_ENGINE == "mcts":
        if mcts_player:
            return mcts_player.select_move(chess_board)
        return None
    elif AI_ENGINE == "minimax":
        return best_move_using_minimax(chess_board, AI_DEPTH)
    else:  # nn
        return best_move_nn(chess_board, depth=AI_DEPTH)


def start_new_game():
    """Initialize a new game and start recording."""
    global game_id, chess_board, board

    chess_board = chess.Board()
    board = Board(*WINDOW_SIZE)
    board.chess_board = chess_board

    # Determine AI player type
    if AI_ENGINE == "mcts":
        ai_type = PlayerType.AI_NN  # MCTS uses neural network
        ai_name = f"AI_MCTS_{MCTS_SIMS}sims"
    elif AI_ENGINE == "minimax":
        ai_type = PlayerType.AI_MINIMAX
        ai_name = f"AI_Minimax_d{AI_DEPTH}"
    else:
        ai_type = PlayerType.AI_NN
        ai_name = f"AI_NN_d{AI_DEPTH}"

    game_id = recorder.start_game(
        white_player="Human",
        black_player=ai_name,
        white_type=PlayerType.HUMAN,
        black_type=ai_type,
        black_depth=AI_DEPTH,
        event="Human vs AI"
    )
    print(f"Started new game (ID: {game_id}) - Engine: {AI_ENGINE}")
    return game_id


def end_current_game(result_str: str, termination: str = None):
    """End the current game and save to database."""
    global game_id

    if game_id is None:
        return

    # Create PGN
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Human vs AI"
    pgn_game.headers["White"] = "Human"
    pgn_game.headers["Black"] = "AI_NN"
    pgn_game.headers["Result"] = result_str

    # Replay moves to build PGN
    temp_board = chess.Board()
    node = pgn_game
    moves = recorder.get_game_moves(game_id)
    for move_record in moves:
        move = chess.Move.from_uci(move_record.uci)
        node = node.add_variation(move)
        temp_board.push(move)

    recorder.end_game(
        game_id=game_id,
        result=result_from_string(result_str),
        termination=termination,
        pgn=str(pgn_game)
    )

    # Print stats
    stats = recorder.get_statistics()
    print(f"Game saved! Total games in database: {stats['total_games']}")
    game_id = None


def draw_board():
    """Draw the current board state."""
    screen.fill('white')
    board.draw(screen)
    pygame.display.flip()


def wait_with_events(seconds):
    """Wait for specified seconds while processing pygame events."""
    start = time.time()
    while time.time() - start < seconds:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                if game_id is not None:
                    end_current_game("*", "ABANDONED")
                pygame.quit()
                sys.exit()
        pygame.time.delay(50)


# Start the first game
start_new_game()


if __name__ == '__main__':
    clock = pygame.time.Clock()
    running = True

    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if game_id is not None:
                    end_current_game("*", "ABANDONED")
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                result = board.handle_click(mx, my)

                if not result:
                    # Just a selection, redraw and continue
                    draw_board()
                    continue

                # Human made a move
                fr, to = result
                uci = board.get_square_from_pos(fr).get_coord() + board.get_square_from_pos(to).get_coord()

                # Check for pawn promotion
                piece = board.get_square_from_pos(to).occupying_piece
                if piece and piece.notation == 'Q' and hasattr(piece, 'just_promoted') and piece.just_promoted:
                    uci += 'q'

                human_move = chess.Move.from_uci(uci)

                # Validate move is legal in python-chess
                if human_move not in chess_board.legal_moves:
                    # Try with promotion variants
                    promoted = False
                    for promo in ['q', 'r', 'b', 'n']:
                        promo_move = chess.Move.from_uci(uci[:4] + promo)
                        if promo_move in chess_board.legal_moves:
                            human_move = promo_move
                            promoted = True
                            break
                    if not promoted:
                        print(f"Illegal move attempted: {uci}")
                        sync_pygame_from_chess(board, chess_board)
                        draw_board()
                        continue

                # Record and execute human move
                if game_id is not None:
                    recorder.record_move(
                        game_id=game_id,
                        board=chess_board,
                        move=human_move,
                        depth=None,
                        time_ms=None
                    )
                chess_board.push(human_move)
                sync_pygame_from_chess(board, chess_board)
                draw_board()
                print(f"Human played: {human_move}")

                # Check if game ended after human move
                if chess_board.is_game_over():
                    outcome = chess_board.outcome()
                    result_str = chess_board.result()
                    termination = outcome.termination.name if outcome else None
                    print(f"Game over: {result_str} ({termination})")
                    end_current_game(result_str, termination)
                    wait_with_events(3)
                    running = False
                    continue

                # Show thinking message
                pygame.display.set_caption("Chess AI - Thinking...")

                # AI's turn
                start_time = time.time()
                ai_move = get_ai_move(chess_board)
                ai_time_ms = int((time.time() - start_time) * 1000)

                pygame.display.set_caption("Chess AI")

                if ai_move is None:
                    result_str = chess_board.result()
                    print(f"Game over (no legal moves for AI): {result_str}")
                    end_current_game(result_str, "NO_LEGAL_MOVES")
                    running = False
                    continue

                # Record and execute AI move
                if game_id is not None:
                    recorder.record_move(
                        game_id=game_id,
                        board=chess_board,
                        move=ai_move,
                        depth=AI_DEPTH,
                        time_ms=ai_time_ms
                    )
                chess_board.push(ai_move)
                sync_pygame_from_chess(board, chess_board)

                # CRITICAL: Draw board immediately after AI move
                draw_board()
                print(f"AI played: {ai_move} ({ai_time_ms}ms)")

                # Check if game ended after AI move
                if chess_board.is_game_over():
                    outcome = chess_board.outcome()
                    result_str = chess_board.result()
                    termination = outcome.termination.name if outcome else None
                    print(f"Game over: {result_str} ({termination})")
                    end_current_game(result_str, termination)
                    wait_with_events(3)
                    running = False
                    continue

        # Always redraw every frame
        draw_board()
        clock.tick(60)

    pygame.quit()

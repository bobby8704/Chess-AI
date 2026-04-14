"""
Monte Carlo Tree Search (MCTS) for Chess

This module implements MCTS with neural network guidance, similar to AlphaZero.
The search uses the policy network to guide exploration and the value network
to evaluate leaf nodes.

Key features:
- PUCT exploration formula (Polynomial Upper Confidence Trees)
- Virtual loss for parallel search (future extension)
- Temperature-based move selection
- Dirichlet noise for exploration during training
"""

import math
import numpy as np
import chess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from features import board_to_tensor


# Material values for evaluation
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0  # King can't be captured
}


def calculate_material(board: chess.Board) -> float:
    """
    Calculate material balance from White's perspective.
    Positive = White advantage, Negative = Black advantage.
    """
    material = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                material += value
            else:
                material -= value
    return material


def get_attacked_pieces(board: chess.Board, color: chess.Color) -> Dict[chess.Square, float]:
    """
    Get all pieces of 'color' that are attacked by the opponent.
    Returns dict of {square: piece_value} for attacked pieces.
    """
    attacked = {}
    opponent = not color

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            # Check if this square is attacked by opponent
            if board.is_attacked_by(opponent, square):
                attacked[square] = PIECE_VALUES.get(piece.piece_type, 0)

    return attacked


def get_hanging_pieces(board: chess.Board, color: chess.Color) -> Dict[chess.Square, float]:
    """
    Get pieces that are attacked but not defended (hanging).
    Returns dict of {square: piece_value} for hanging pieces.
    """
    hanging = {}
    opponent = not color

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            # Check if attacked by opponent
            if board.is_attacked_by(opponent, square):
                # Check if defended by own pieces
                if not board.is_attacked_by(color, square):
                    hanging[square] = PIECE_VALUES.get(piece.piece_type, 0)
                else:
                    # Even if defended, check if attacker is less valuable
                    # This is a simplified check - full SEE would be better
                    pass

    return hanging


def evaluate_material_safety(board: chess.Board) -> float:
    """
    Evaluate position based on material and piece safety.
    Returns a value from -1 to 1 from the current player's perspective.
    """
    # Calculate current material balance (positive = White advantage)
    material = calculate_material(board)

    # Get hanging pieces for both sides
    # The side to move can capture opponent hanging pieces
    current_color = board.turn
    opponent_color = not board.turn

    current_hanging = get_hanging_pieces(board, current_color)
    opponent_hanging = get_hanging_pieces(board, opponent_color)

    # Current player's hanging pieces are at risk; opponent's can be captured
    current_at_risk = sum(current_hanging.values())
    opponent_at_risk = sum(opponent_hanging.values())

    # Adjust material from White's perspective
    if current_color == chess.WHITE:
        adjusted_material = material - current_at_risk + opponent_at_risk
    else:
        adjusted_material = material + current_at_risk - opponent_at_risk

    # Convert to current player's perspective, normalized to [-1, 1]
    if current_color == chess.BLACK:
        adjusted_material = -adjusted_material

    normalized = adjusted_material / 15.0
    return max(-1.0, min(1.0, normalized))



def is_blunder_move(board: chess.Board, move: chess.Move) -> Tuple[bool, float]:
    """
    Check if a move is a blunder (hangs a piece, walks into mate, allows
    a bad exchange, or makes a bad voluntary capture).

    Returns:
        (is_blunder, material_lost)
    """
    our_color = board.turn  # The color making the move
    opponent_color = not our_color

    # Check if this move is a capture - evaluate if it's a BAD capture
    if board.is_capture(move):
        # Get the piece we're moving (the attacker)
        attacker_piece = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

        # Get the piece we're capturing
        captured_square = move.to_square
        captured_piece = board.piece_at(captured_square)

        # Handle en passant
        if captured_piece is None and attacker_piece and attacker_piece.piece_type == chess.PAWN:
            # En passant capture
            captured_value = 1.0  # Pawn
        else:
            captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0

        # Make the move
        board_after = board.copy()
        board_after.push(move)

        # Check if opponent can recapture our piece
        can_be_recaptured = False
        recapturer_value = 0.0

        for opp_move in board_after.legal_moves:
            if opp_move.to_square == move.to_square and board_after.is_capture(opp_move):
                can_be_recaptured = True
                recapturer = board_after.piece_at(opp_move.from_square)
                recapturer_value = PIECE_VALUES.get(recapturer.piece_type, 0) if recapturer else 0
                break  # Found a recapture

        if can_be_recaptured:
            # We capture 'captured_value', they recapture 'attacker_value'
            # Net exchange: we gain captured_value, we lose attacker_value
            net_exchange = captured_value - attacker_value

            # If we're losing material in the exchange, it's a blunder
            # E.g., Rook(5) takes Bishop(3.25), they recapture = -1.75 for us
            if net_exchange < -1.5:  # Losing more than 1.5 pawns worth
                return True, -net_exchange

    # Make the move temporarily
    board_copy = board.copy()

    # Track material gained by this move (captures)
    material_gained = 0.0
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            material_gained = PIECE_VALUES.get(captured_piece.piece_type, 0)
        elif board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
            material_gained = 1.0  # en passant

    board_copy.push(move)

    # Check for immediate checkmate (we got mated)
    if board_copy.is_checkmate():
        return True, 100.0

    # Check if opponent can make a profitable capture after our move
    # Exclude the square we just moved to (that exchange is already evaluated above)
    best_opponent_gain = 0.0
    for opp_move in board_copy.legal_moves:
        if board_copy.is_capture(opp_move):
            # Skip recaptures on the square we just moved to — already handled
            if opp_move.to_square == move.to_square:
                continue

            captured_square = opp_move.to_square
            captured_piece = board_copy.piece_at(captured_square)
            if captured_piece is None:
                continue

            captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
            attacker_square = opp_move.from_square
            attacker_piece = board_copy.piece_at(attacker_square)
            attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

            # Calculate exchange value
            board_copy2 = board_copy.copy()
            board_copy2.push(opp_move)

            # Check if we can recapture
            can_recapture = False
            for our_recapture in board_copy2.legal_moves:
                if our_recapture.to_square == opp_move.to_square and board_copy2.is_capture(our_recapture):
                    can_recapture = True
                    break

            # Net material change: opponent gains captured_value, might lose attacker_value
            if can_recapture:
                net_gain = captured_value - attacker_value
            else:
                net_gain = captured_value

            # If opponent gains material, it's a blunder for us
            if net_gain > best_opponent_gain:
                best_opponent_gain = net_gain

    # Subtract material we gained — winning a queen but losing a knight is net positive
    effective_loss = best_opponent_gain - material_gained

    # Consider it a blunder only if we're net losing significant material
    if effective_loss >= 2.0:
        return True, effective_loss

    # Also check if we're leaving high-value pieces undefended
    hanging = get_hanging_pieces(board_copy, our_color)
    if hanging:
        max_hanging = max(hanging.values())
        # Don't flag if we gained more material than we're risking
        if max_hanging >= 3.0 and max_hanging > material_gained:
            return True, max_hanging - material_gained

    return False, 0.0


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""
    num_simulations: int = 800        # Number of MCTS simulations per move
    c_puct: float = 1.5               # PUCT exploration constant
    dirichlet_alpha: float = 0.3      # Dirichlet noise alpha (for chess)
    dirichlet_epsilon: float = 0.25   # Root noise mixing coefficient
    temperature: float = 1.0          # Move selection temperature
    temperature_threshold: int = 30   # Move number after which temp -> 0
    add_noise: bool = True            # Add exploration noise at root
    use_material_eval: bool = True    # Use hybrid material evaluation
    material_weight: float = 0.35     # Weight for material eval - NN value head is weak, material fills the gap
    blunder_penalty: float = 0.8      # Penalty for blunder moves (reduces prior)


class MCTSNode:
    """
    A node in the MCTS tree.

    Each node represents a game state and stores statistics for the UCB formula.
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional['MCTSNode'] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # P(s, a) from policy network

        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.is_expanded: bool = False
        self.is_terminal: bool = board.is_game_over()

        # Terminal value (if terminal)
        if self.is_terminal:
            result = board.result()
            if result == "1-0":
                self.terminal_value = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                self.terminal_value = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                self.terminal_value = 0.0
        else:
            self.terminal_value = None

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate UCB score for node selection.

        Uses PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Note: self.value is from THIS node's perspective (the opponent of the parent).
        We negate it because the parent wants to maximize ITS value, not the opponent's.
        """
        if self.visit_count == 0:
            # Unvisited nodes have high exploration bonus
            return float('inf')

        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        # Negate value because child stores value from opponent's perspective
        return -self.value + exploration

    def expand(self, move_probs: Dict[chess.Move, float]):
        """
        Expand this node by creating child nodes for all legal moves.

        Args:
            move_probs: Policy network probabilities for each legal move
        """
        if self.is_expanded or self.is_terminal:
            return

        for move, prob in move_probs.items():
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(
                board=child_board,
                parent=self,
                move=move,
                prior=prob
            )

        self.is_expanded = True

    def select_child(self, c_puct: float) -> Tuple[chess.Move, 'MCTSNode']:
        """Select the child with highest UCB score."""
        best_score = float('-inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def backpropagate(self, value: float):
        """
        Backpropagate a value up the tree.

        The value alternates sign as we go up (opponent's perspective).
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective
            node = node.parent

    def add_dirichlet_noise(self, alpha: float, epsilon: float):
        """
        Add Dirichlet noise to the prior probabilities at this node.

        This encourages exploration during self-play training.
        """
        if not self.children:
            return

        noise = np.random.dirichlet([alpha] * len(self.children))
        for i, child in enumerate(self.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Usage:
        mcts = MCTS(config, evaluate_fn)
        move = mcts.search(board)
    """

    def __init__(
        self,
        config: MCTSConfig = None,
        evaluate_fn=None,
        root_evaluate_fn=None
    ):
        """
        Initialize MCTS.

        Args:
            config: MCTS configuration
            evaluate_fn: Function (board) -> (move_probs, value)
                        Used for non-root nodes (fast path)
            root_evaluate_fn: Function (board) -> (move_probs, value)
                        Used for root node only (includes heuristics/blunder checks)
                        If None, uses evaluate_fn for root too
        """
        self.config = config or MCTSConfig()
        self.evaluate_fn = evaluate_fn or self._default_evaluate
        self.root_evaluate_fn = root_evaluate_fn

    def _default_evaluate(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """Default evaluation: uniform policy, zero value."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0
        uniform_prob = 1.0 / len(legal_moves)
        return {move: uniform_prob for move in legal_moves}, 0.0

    def search(
        self,
        board: chess.Board,
        num_simulations: int = None
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """
        Run MCTS from the given position.

        Args:
            board: Current board position
            num_simulations: Number of simulations (overrides config)

        Returns:
            (best_move, move_probabilities)
            where move_probabilities is the visit count distribution
        """
        if board.is_game_over():
            return None, {}

        num_sims = num_simulations or self.config.num_simulations

        # Create root node
        root = MCTSNode(board)

        # Expand root (use full evaluate with heuristics)
        root_eval_fn = self.root_evaluate_fn or self.evaluate_fn
        move_probs, _ = root_eval_fn(board)
        root.expand(move_probs)

        # Add exploration noise at root during training
        if self.config.add_noise:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Run simulations (use fast evaluate for non-root nodes)
        for _ in range(num_sims):
            node = root
            path = [node]

            # Selection: traverse tree to leaf
            while node.is_expanded and not node.is_terminal:
                _, node = node.select_child(self.config.c_puct)
                path.append(node)

            # Evaluation
            if node.is_terminal:
                value = node.terminal_value
            else:
                # Expand and evaluate (fast path — no heuristics/blunder checks)
                move_probs, value = self.evaluate_fn(node.board)
                node.expand(move_probs)

            # Backpropagation
            node.backpropagate(value)

        # Calculate move probabilities from visit counts
        move_visits = {
            move: child.visit_count
            for move, child in root.children.items()
        }

        total_visits = sum(move_visits.values())
        move_probs = {
            move: visits / total_visits
            for move, visits in move_visits.items()
        }

        # Select move based on temperature
        best_move = self._select_move(root, move_probs)

        return best_move, move_probs

    def _select_move(
        self,
        root: MCTSNode,
        move_probs: Dict[chess.Move, float]
    ) -> chess.Move:
        """
        Select a move based on visit counts and temperature.

        At low temperature, select the most visited move.
        At higher temperature, sample proportionally to visits.
        """
        if not move_probs:
            return None

        if self.config.temperature == 0:
            # Greedy selection
            return max(move_probs.items(), key=lambda x: x[1])[0]

        # Apply temperature
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])

        # Apply temperature scaling
        probs = np.power(probs, 1.0 / self.config.temperature)
        probs = probs / probs.sum()

        # Sample
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    def get_search_statistics(
        self,
        board: chess.Board,
        num_simulations: int = None
    ) -> Dict:
        """
        Run search and return detailed statistics.

        Useful for training and analysis.
        """
        move, move_probs = self.search(board, num_simulations)

        return {
            "selected_move": move,
            "move_probs": move_probs,
            "num_simulations": num_simulations or self.config.num_simulations,
        }


def _apply_heuristic_boosts(
    board: chess.Board,
    move_probs: Dict[chess.Move, float]
) -> Dict[chess.Move, float]:
    """
    Apply heuristic policy adjustments to compensate for NN weaknesses.
    Boosts/penalizes specific move types based on chess principles.
    """
    move_number = board.fullmove_number
    our_color = board.turn

    for move in list(move_probs.keys()):
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue

        # --- 1. Boost captures of undefended / winning exchanges ---
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                captured_val = PIECE_VALUES.get(captured.piece_type, 0)
                attacker_val = PIECE_VALUES.get(piece.piece_type, 0)
                opponent = not our_color
                is_defended = board.is_attacked_by(opponent, move.to_square)
                if not is_defended and captured_val >= 3.0:
                    move_probs[move] = max(move_probs[move], 0.3)
                elif is_defended and captured_val > attacker_val + 1.0:
                    move_probs[move] = max(move_probs[move], 0.2)

        # --- 2. Boost castling in the opening ---
        if board.is_castling(move) and move_number <= 15:
            # Strong boost — castling is almost always good in the opening
            move_probs[move] = max(move_probs[move], 0.25)

        # --- 3. Penalize early queen trades at equal material ---
        if board.is_capture(move) and piece.piece_type == chess.QUEEN:
            captured = board.piece_at(move.to_square)
            if captured and captured.piece_type == chess.QUEEN:
                # Check if opponent can recapture (making it a trade)
                board_after = board.copy()
                board_after.push(move)
                can_recapture = any(
                    m.to_square == move.to_square and board_after.is_capture(m)
                    for m in board_after.legal_moves
                )
                if can_recapture and move_number <= 20:
                    # Check material balance — trading is OK when ahead
                    material = calculate_material(board)
                    our_advantage = material if our_color == chess.WHITE else -material
                    if our_advantage < 3.0:  # Not significantly ahead
                        move_probs[move] *= 0.3  # Penalize queen trade

        # --- 4. Boost passed pawn pushes in endgames ---
        if piece.piece_type == chess.PAWN and not board.is_capture(move):
            total_pieces = len(board.piece_map())
            if total_pieces <= 16:  # Endgame-ish (half or fewer pieces)
                if _is_passed_pawn(board, move.from_square, our_color):
                    to_rank = chess.square_rank(move.to_square)
                    if our_color == chess.WHITE:
                        closeness = to_rank / 7.0
                    else:
                        closeness = (7 - to_rank) / 7.0
                    boost = 0.15 + 0.35 * closeness
                    move_probs[move] = max(move_probs[move], boost)

        # --- 5. Boost checks, penalize stalemate ---
        board_after = board.copy()
        board_after.push(move)
        if board_after.is_checkmate():
            move_probs[move] = max(move_probs[move], 0.95)
        elif board_after.is_stalemate():
            # Stalemate is almost always terrible when we're ahead
            move_probs[move] *= 0.001
        elif board_after.is_check():
            move_probs[move] = max(move_probs[move], 0.12)

        # --- 6. Prefer queen promotion over underpromotion ---
        if move.promotion:
            if move.promotion == chess.QUEEN:
                move_probs[move] = max(move_probs[move], 0.5)
            else:
                # Underpromotion is almost never better than queen
                # Only keep it if it's checkmate (already boosted above)
                if not board_after.is_checkmate():
                    move_probs[move] *= 0.05

        # --- 7. Penalize passive king moves (Kg8/Kh8 shuffling) ---
        if piece.piece_type == chess.KING and not board.is_check():
            # If king moves but doesn't castle, and there are non-king moves available
            if not board.is_castling(move):
                non_king_moves = [m for m in board.legal_moves
                                  if board.piece_at(m.from_square) and
                                  board.piece_at(m.from_square).piece_type != chess.KING]
                if len(non_king_moves) > 3:
                    # Penalize king shuffling when there are plenty of other moves
                    move_probs[move] *= 0.4

    # --- 8. Boost piece development in the opening ---
    if move_number <= 12:
        move_probs = _boost_development(board, move_probs, our_color)

    # --- 9. King safety: penalize pawn pushes near castled king ---
    move_probs = _penalize_king_shelter_weakening(board, move_probs, our_color)

    # --- 10. Checkmate forcing: when opponent has lone king, boost restricting moves ---
    move_probs = _boost_mate_forcing(board, move_probs, our_color)

    # --- 11. Simplification: when ahead in material, boost equal trades ---
    move_probs = _boost_simplification(board, move_probs, our_color)

    return move_probs


def _boost_development(
    board: chess.Board,
    move_probs: Dict[chess.Move, float],
    color: chess.Color
) -> Dict[chess.Move, float]:
    """Boost moves that develop unplayed minor pieces in the opening."""
    back_rank = 0 if color == chess.WHITE else 7

    # Find minor pieces (knights/bishops) still on their starting squares
    starting_minors = []
    for file in range(8):
        sq = chess.square(file, back_rank)
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type in (chess.KNIGHT, chess.BISHOP):
            starting_minors.append(sq)

    if not starting_minors:
        return move_probs

    for move, prob in list(move_probs.items()):
        # Boost moves that move an undeveloped minor piece
        if move.from_square in starting_minors:
            move_probs[move] = max(prob, 0.12)

        # Mildly penalize moving the queen early (before minor pieces are out)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.QUEEN and len(starting_minors) >= 2:
            if not board.is_capture(move):
                move_probs[move] *= 0.6

    return move_probs


def _penalize_king_shelter_weakening(
    board: chess.Board,
    move_probs: Dict[chess.Move, float],
    color: chess.Color
) -> Dict[chess.Move, float]:
    """Penalize pawn pushes that weaken the castled king's shelter."""
    # Only apply if we've castled
    king_sq = board.king(color)
    if king_sq is None:
        return move_probs

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    # Check if king is on a castled position (g1/g8 for kingside, c1/c8 for queenside)
    is_castled_kingside = king_file >= 6 and king_rank in (0, 7)
    is_castled_queenside = king_file <= 2 and king_rank in (0, 7)

    if not is_castled_kingside and not is_castled_queenside:
        return move_probs

    # Determine which files shelter the king
    if is_castled_kingside:
        shelter_files = [5, 6, 7]  # f, g, h
    else:
        shelter_files = [0, 1, 2]  # a, b, c

    shelter_rank = 1 if color == chess.WHITE else 6  # rank of shelter pawns

    for move, prob in list(move_probs.items()):
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            continue
        # If this pawn is on a shelter file and shelter rank, penalize pushing it
        from_file = chess.square_file(move.from_square)
        from_rank = chess.square_rank(move.from_square)
        if from_file in shelter_files and from_rank == shelter_rank:
            # Pushing shelter pawn — mild penalty (sometimes it's needed)
            move_probs[move] *= 0.5

    return move_probs


def _is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """Check if a pawn on the given square is a passed pawn (no opposing pawns blocking or adjacent)."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    opponent = not color

    # Check files: same file and adjacent files
    for f in range(max(0, file - 1), min(7, file + 1) + 1):
        # Check all ranks ahead of this pawn
        if color == chess.WHITE:
            check_ranks = range(rank + 1, 8)
        else:
            check_ranks = range(0, rank)
        for r in check_ranks:
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == opponent:
                return False
    return True


def _boost_simplification(
    board: chess.Board,
    move_probs: Dict[chess.Move, float],
    color: chess.Color
) -> Dict[chess.Move, float]:
    """
    When we're ahead in material, boost trades (captures where opponent recaptures).
    Simplifying when ahead makes the advantage easier to convert.
    When behind, penalize trades (keep pieces on the board for counterplay).
    """
    # Calculate our material advantage
    material = calculate_material(board)
    our_advantage = material if color == chess.WHITE else -material

    if abs(our_advantage) < 2.0:
        return move_probs  # Roughly equal — no simplification pressure

    for move in list(move_probs.keys()):
        if not board.is_capture(move):
            continue

        piece = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square)
        if piece is None or captured is None:
            continue

        piece_val = PIECE_VALUES.get(piece.piece_type, 0)
        captured_val = PIECE_VALUES.get(captured.piece_type, 0)

        # Is this a roughly equal trade? (within 1 pawn of value)
        is_equal_trade = abs(piece_val - captured_val) <= 1.0

        if our_advantage >= 2.0:
            # We're AHEAD — boost equal trades to simplify
            if is_equal_trade:
                # Don't trade queens unless way ahead (queen trades reduce mating chances)
                if piece.piece_type == chess.QUEEN and captured.piece_type == chess.QUEEN:
                    if our_advantage >= 8.0:
                        move_probs[move] = max(move_probs[move], 0.2)
                    else:
                        move_probs[move] *= 0.5  # Avoid queen trades unless dominant
                else:
                    # Trade knights, bishops, rooks — simplify!
                    move_probs[move] = max(move_probs[move], 0.18)
        elif our_advantage <= -2.0:
            # We're BEHIND — avoid equal trades, keep pieces for complications
            if is_equal_trade:
                move_probs[move] *= 0.5

    return move_probs


def _boost_mate_forcing(
    board: chess.Board,
    move_probs: Dict[chess.Move, float],
    color: chess.Color
) -> Dict[chess.Move, float]:
    """
    When the opponent has only a king, boost moves that restrict its mobility
    and drive it toward the edge. This guides MCTS toward the mating pattern.
    """
    opponent = not color

    # Check if opponent has only a king
    opp_pieces = board.pieces(chess.PAWN, opponent) | \
                 board.pieces(chess.KNIGHT, opponent) | \
                 board.pieces(chess.BISHOP, opponent) | \
                 board.pieces(chess.ROOK, opponent) | \
                 board.pieces(chess.QUEEN, opponent)
    if len(opp_pieces) > 0:
        return move_probs

    opp_king = board.king(opponent)
    our_king = board.king(color)
    if opp_king is None or our_king is None:
        return move_probs

    # Count opponent king's current escape squares
    def count_escapes(b, king_sq, attacker_color):
        escapes = 0
        for delta_f in [-1, 0, 1]:
            for delta_r in [-1, 0, 1]:
                if delta_f == 0 and delta_r == 0:
                    continue
                f = chess.square_file(king_sq) + delta_f
                r = chess.square_rank(king_sq) + delta_r
                if 0 <= f <= 7 and 0 <= r <= 7:
                    sq = chess.square(f, r)
                    if not b.is_attacked_by(attacker_color, sq):
                        p = b.piece_at(sq)
                        if p is None or p.color == attacker_color:
                            escapes += 1
        return escapes

    current_escapes = count_escapes(board, opp_king, color)

    for move in list(move_probs.keys()):
        board_after = board.copy()
        board_after.push(move)

        if board_after.is_checkmate():
            move_probs[move] = max(move_probs[move], 0.99)
            continue

        if board_after.is_stalemate():
            move_probs[move] *= 0.001
            continue

        # How many escapes does the opponent king have after this move?
        new_opp_king = board_after.king(opponent)
        if new_opp_king is None:
            continue
        new_escapes = count_escapes(board_after, new_opp_king, color)

        # Reward moves that reduce escape squares
        if new_escapes < current_escapes:
            reduction = current_escapes - new_escapes
            move_probs[move] = max(move_probs[move], 0.15 + reduction * 0.1)

        # Reward king approach (our king moving closer)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING:
            old_dist = max(abs(chess.square_file(our_king) - chess.square_file(opp_king)),
                          abs(chess.square_rank(our_king) - chess.square_rank(opp_king)))
            new_dist = max(abs(chess.square_file(move.to_square) - chess.square_file(opp_king)),
                          abs(chess.square_rank(move.to_square) - chess.square_rank(opp_king)))
            if new_dist < old_dist:
                move_probs[move] = max(move_probs[move], 0.2)

    return move_probs


def _select_lone_king_mate_move(board: chess.Board) -> Optional[chess.Move]:
    """
    When opponent has only a king, use eval-based 1-ply search to find the
    best forcing move. Bypasses MCTS since search depth is the bottleneck.
    Picks the move that maximizes the hand-coded eval (which rewards
    driving the king to the edge and restricting its squares).
    """
    our_color = board.turn
    opponent = not our_color

    # Check if opponent has only a king
    opp_pieces = board.pieces(chess.PAWN, opponent) | \
                 board.pieces(chess.KNIGHT, opponent) | \
                 board.pieces(chess.BISHOP, opponent) | \
                 board.pieces(chess.ROOK, opponent) | \
                 board.pieces(chess.QUEEN, opponent)
    if len(opp_pieces) > 0:
        return None

    # Need mating material
    our_queens = len(board.pieces(chess.QUEEN, our_color))
    our_rooks = len(board.pieces(chess.ROOK, our_color))
    if our_queens == 0 and our_rooks == 0:
        return None

    from evaluation import evaluate as hc_evaluate

    best_move = None
    best_score = float('-inf')

    for move in board.legal_moves:
        board.push(move)

        if board.is_checkmate():
            board.pop()
            return move  # Immediate mate — take it

        if board.is_stalemate():
            board.pop()
            continue  # Skip stalemate moves

        # Evaluate from opponent's perspective (after our move, it's their turn)
        # Negate to get our perspective
        score = -hc_evaluate(board)

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


class MCTSPlayer:
    """
    Chess player using MCTS with neural network.

    This class provides a simple interface for playing chess using MCTS.
    """

    def __init__(
        self,
        model=None,
        config: MCTSConfig = None,
        device=None
    ):
        """
        Initialize MCTS player.

        Args:
            model: DualNet or similar model with get_policy_value method
            config: MCTS configuration
            device: PyTorch device
        """
        import torch
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model is not None:
            model.to(self.device)
            model.eval()

        # Fast evaluate for non-root nodes, full evaluate for root only
        self.mcts = MCTS(
            config=self.config,
            evaluate_fn=self._evaluate_fast,
            root_evaluate_fn=self._evaluate_full,
        )

    def _nn_forward(self, board: chess.Board):
        """Run NN forward pass and return (move_probs_dict, nn_value)."""
        import torch
        from neural_network import get_legal_move_mask, encode_move, POLICY_OUTPUT_SIZE, is_cnn_model

        if self.model is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            return {m: 1.0 / len(legal_moves) for m in legal_moves}, 0.0

        if is_cnn_model(self.model):
            from features import board_to_tensor_2d
            x = torch.from_numpy(board_to_tensor_2d(board)).float().to(self.device).unsqueeze(0)
        else:
            x = torch.from_numpy(board_to_tensor(board)).float().to(self.device).unsqueeze(0)
        mask = get_legal_move_mask(board).to(self.device).unsqueeze(0)

        with torch.no_grad():
            policy_probs, nn_value = self.model.get_policy_value(x, mask)

        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        move_probs = {}
        for move in board.legal_moves:
            idx = encode_move(move)
            if idx < POLICY_OUTPUT_SIZE:
                move_probs[move] = float(policy_probs[idx])

        return move_probs, nn_value.item()

    def _evaluate_fast(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """Fast evaluation for non-root nodes: NN policy + quiescence search value.
        Quiescence extends captures to avoid evaluating unstable positions."""
        move_probs, _ = self._nn_forward(board)

        from evaluation import evaluate_quiescence
        value = evaluate_quiescence(board)

        # Normalize
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p / total for m, p in move_probs.items()}

        return move_probs, value

    def _evaluate_full(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """Full evaluation for root node: NN + heuristics + blunder detection."""
        move_probs, _ = self._nn_forward(board)

        # Heuristic policy adjustments (only at root — too expensive per-sim)
        move_probs = _apply_heuristic_boosts(board, move_probs)

        # Position value with quiescence search (plays out captures)
        from evaluation import evaluate_quiescence
        value = evaluate_quiescence(board)

        # Blunder detection (only at root)
        blunder_moves = {}
        safe_moves = {}
        for move, prob in move_probs.items():
            is_blunder, material_lost = is_blunder_move(board, move)
            if is_blunder:
                blunder_moves[move] = (prob, material_lost)
            else:
                safe_moves[move] = prob

        if safe_moves:
            adjusted_probs = {}
            total_safe_prob = sum(safe_moves.values())
            if total_safe_prob > 0:
                for move, prob in safe_moves.items():
                    adjusted_probs[move] = (prob / total_safe_prob) * 0.99
                for move, (prob, material_lost) in blunder_moves.items():
                    inverse_severity = max(0.1, 1.0 - material_lost / 10.0)
                    adjusted_probs[move] = 0.01 * inverse_severity / max(len(blunder_moves), 1)
            else:
                for move, prob in safe_moves.items():
                    adjusted_probs[move] = prob
                for move, (prob, _) in blunder_moves.items():
                    adjusted_probs[move] = prob * 0.01
            move_probs = adjusted_probs
        elif blunder_moves:
            adjusted_probs = {}
            for move, (prob, material_lost) in blunder_moves.items():
                inverse_loss = 10.0 - material_lost
                adjusted_probs[move] = prob * inverse_loss
            move_probs = adjusted_probs

        # Normalize
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p / total for m, p in move_probs.items()}

        return move_probs, value

    def select_move(
        self,
        board: chess.Board,
        temperature: float = None,
        num_simulations: int = None,
        return_policy: bool = False
    ):
        """
        Select a move using MCTS.

        Args:
            board: Current position
            temperature: Override temperature (None uses config)
            num_simulations: Override simulation count
            return_policy: Whether to return the policy distribution

        Returns:
            Selected move, or (move, policy_dict) if return_policy=True
        """
        # Endgame tablebase: perfect play for ≤7 pieces
        import tablebase
        if tablebase.should_probe(board):
            tb_move = tablebase.probe(board)
            if tb_move is not None:
                if return_policy:
                    return tb_move, {tb_move: 1.0}
                return tb_move

        # Fallback: lone king eval-based search (when tablebase unavailable)
        lone_king_move = _select_lone_king_mate_move(board)
        if lone_king_move is not None:
            if return_policy:
                return lone_king_move, {lone_king_move: 1.0}
            return lone_king_move

        if temperature is not None:
            old_temp = self.config.temperature
            self.config.temperature = temperature

        move, move_probs = self.mcts.search(board, num_simulations)

        if temperature is not None:
            self.config.temperature = old_temp

        # Anti-repetition: if best move would cause a draw by repetition,
        # pick the next best move instead
        if move and move_probs and not board.is_game_over():
            test_board = board.copy()
            test_board.push(move)
            if test_board.can_claim_draw() or test_board.is_repetition(2):
                # Try alternatives sorted by visit probability
                sorted_moves = sorted(move_probs.items(), key=lambda x: -x[1])
                for alt_move, prob in sorted_moves:
                    if alt_move == move:
                        continue
                    alt_board = board.copy()
                    alt_board.push(alt_move)
                    if not alt_board.can_claim_draw() and not alt_board.is_repetition(2):
                        move = alt_move
                        break

        if return_policy:
            return move, move_probs
        return move

    def get_move_probabilities(
        self,
        board: chess.Board,
        num_simulations: int = None
    ) -> Dict[chess.Move, float]:
        """Get the MCTS visit count distribution."""
        _, move_probs = self.mcts.search(board, num_simulations)
        return move_probs


def best_move_mcts(
    board: chess.Board,
    num_simulations: int = 100,
    model=None
) -> Optional[chess.Move]:
    """
    Simple function to get the best move using MCTS.

    This is a convenience function for integration with existing code.
    """
    if board.is_game_over():
        return None

    config = MCTSConfig(
        num_simulations=num_simulations,
        temperature=0,  # Greedy selection
        add_noise=False  # No noise for play
    )

    player = MCTSPlayer(model=model, config=config)
    return player.select_move(board)


if __name__ == "__main__":
    # Test MCTS
    print("Testing MCTS...")
    print("=" * 50)

    # Test with default evaluation
    config = MCTSConfig(num_simulations=100, temperature=1.0)
    mcts = MCTS(config)

    board = chess.Board()
    print(f"\nPosition:\n{board}")

    move, probs = mcts.search(board)
    print(f"\nSelected move: {move}")
    print(f"Top 5 moves by visit probability:")
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:5]
    for m, p in sorted_probs:
        print(f"  {m}: {p:.3f}")

    # Test MCTSPlayer
    print("\n" + "=" * 50)
    print("Testing MCTSPlayer (no model)...")

    player = MCTSPlayer(model=None, config=config)
    move = player.select_move(board)
    print(f"MCTSPlayer selected: {move}")

    # Test best_move_mcts
    print("\n" + "=" * 50)
    print("Testing best_move_mcts...")

    move = best_move_mcts(board, num_simulations=50)
    print(f"best_move_mcts selected: {move}")

    print("\nAll MCTS tests passed!")

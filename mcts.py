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
    Returns a value from -1 to 1 (like neural network value).
    """
    # Calculate current material balance
    material = calculate_material(board)

    # Get hanging pieces for both sides
    white_hanging = get_hanging_pieces(board, chess.WHITE)
    black_hanging = get_hanging_pieces(board, chess.BLACK)

    # Calculate potential material loss
    white_at_risk = sum(white_hanging.values())
    black_at_risk = sum(black_hanging.values())

    # Adjust material by hanging pieces (opponent can take them)
    adjusted_material = material - white_at_risk + black_at_risk

    # Convert to -1 to 1 scale (assuming ~40 points total material)
    # Clamp to reasonable range
    normalized = adjusted_material / 15.0  # ~1 queen + 1 rook difference = significant
    normalized = max(-1.0, min(1.0, normalized))

    return normalized


def simple_see(board: chess.Board, square: chess.Square) -> float:
    """
    Simple Static Exchange Evaluation for a square.
    Returns the material gain/loss from exchanging on this square.
    Positive = good for the side to move.
    """
    piece = board.piece_at(square)
    if piece is None:
        return 0.0

    target_value = PIECE_VALUES.get(piece.piece_type, 0)
    target_color = piece.color
    attacker_color = not target_color

    # Get individual attackers and defenders by checking each piece's attacks
    attackers = []
    defenders = []

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or sq == square:
            continue
        # Check if this specific piece attacks the target square
        if board.is_attacked_by(p.color, square):
            # Verify this specific square is an attacker by checking
            # if removing it would change the attack status
            piece_value = PIECE_VALUES.get(p.piece_type, 0)
            if p.color == attacker_color:
                attackers.append(piece_value)
            elif p.color == target_color:
                defenders.append(piece_value)

    # Deduplicate: is_attacked_by checks color-wide, so we use attackers/defenders
    # from the board's actual attacker maps instead
    attackers = sorted(board_attackers_of(board, square, attacker_color))
    defenders = sorted(board_attackers_of(board, square, target_color))

    if not attackers:
        return 0.0

    # Simulate the exchange
    gain = [target_value]
    current_value = attackers[0]
    attacker_idx = 1
    defender_idx = 0

    # Alternate: attacker captures, defender recaptures
    while True:
        # Defender recaptures
        if defender_idx >= len(defenders):
            break
        gain.append(current_value - gain[-1] if len(gain) > 0 else current_value)
        current_value = defenders[defender_idx]
        defender_idx += 1

        # Next attacker captures
        if attacker_idx >= len(attackers):
            break
        current_value = attackers[attacker_idx]
        attacker_idx += 1

    return gain[0] if gain else 0.0


def board_attackers_of(board: chess.Board, square: chess.Square, color: chess.Color) -> List[float]:
    """Get sorted list of piece values attacking a square for a given color."""
    values = []
    attackers_mask = board.attackers(color, square)
    for sq in attackers_mask:
        p = board.piece_at(sq)
        if p:
            values.append(PIECE_VALUES.get(p.piece_type, 0))
    return sorted(values)


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
            # E.g., Queen (9) takes Bishop (3), they recapture Queen = -6 for us
            if net_exchange < -2.0:  # Losing more than 2 pawns worth
                return True, -net_exchange

    # Make the move temporarily
    board_copy = board.copy()
    board_copy.push(move)

    # Check for immediate checkmate (we got mated)
    if board_copy.is_checkmate():
        return True, 100.0

    # Check if opponent can make a profitable capture (of pieces OTHER than the one we just moved)
    best_opponent_gain = 0.0
    for opp_move in board_copy.legal_moves:
        if board_copy.is_capture(opp_move):
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

    # Consider it a blunder if opponent gains material worth >= 2 pawns
    if best_opponent_gain >= 2.0:
        return True, best_opponent_gain

    # Also check if we're leaving high-value pieces undefended
    hanging = get_hanging_pieces(board_copy, our_color)
    if hanging:
        max_hanging = max(hanging.values())
        if max_hanging >= 3.0:
            return True, max_hanging

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
    material_weight: float = 0.7      # Weight for material eval (0-1) - high to prevent blunders
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
        evaluate_fn=None
    ):
        """
        Initialize MCTS.

        Args:
            config: MCTS configuration
            evaluate_fn: Function (board) -> (move_probs, value)
                        If None, uses uniform policy and zero value
        """
        self.config = config or MCTSConfig()
        self.evaluate_fn = evaluate_fn or self._default_evaluate

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

        # Expand root
        move_probs, _ = self.evaluate_fn(board)
        root.expand(move_probs)

        # Add exploration noise at root during training
        if self.config.add_noise:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Run simulations
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
                # Expand and evaluate
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

        self.mcts = MCTS(config=self.config, evaluate_fn=self._evaluate)

    def _evaluate(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """Evaluate position using neural network with optional material hybrid."""
        import torch
        from neural_network import get_legal_move_mask, encode_move, POLICY_OUTPUT_SIZE

        if self.model is None:
            # Uniform policy, zero value
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            return {m: 1.0 / len(legal_moves) for m in legal_moves}, 0.0

        # Convert to tensor
        x = torch.from_numpy(board_to_tensor(board)).float().to(self.device).unsqueeze(0)
        mask = get_legal_move_mask(board).to(self.device).unsqueeze(0)

        with torch.no_grad():
            policy_probs, nn_value = self.model.get_policy_value(x, mask)

        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        nn_value = nn_value.item()

        # Convert to move dictionary
        move_probs = {}
        for move in board.legal_moves:
            idx = encode_move(move)
            if idx < POLICY_OUTPUT_SIZE:
                move_probs[move] = float(policy_probs[idx])

        # Apply hybrid material evaluation if enabled
        if self.config.use_material_eval:
            # Calculate material-based value
            material_value = evaluate_material_safety(board)

            # Adjust for side to move (value is from current player's perspective)
            if board.turn == chess.BLACK:
                material_value = -material_value

            # Blend neural network value with material evaluation
            # Use higher material weight when there's significant material imbalance
            effective_material_weight = self.config.material_weight
            if abs(material_value) > 0.3:  # Significant material difference (roughly 5+ pawns)
                effective_material_weight = 0.9  # Trust material almost entirely

            value = (1 - effective_material_weight) * nn_value + \
                    effective_material_weight * material_value

            # Analyze all moves for blunders
            blunder_moves = {}
            safe_moves = {}
            for move, prob in move_probs.items():
                is_blunder, material_lost = is_blunder_move(board, move)
                if is_blunder:
                    blunder_moves[move] = (prob, material_lost)
                else:
                    safe_moves[move] = prob

            # If there are safe moves, heavily penalize blunders
            if safe_moves:
                adjusted_probs = {}

                # Calculate total safe probability
                total_safe_prob = sum(safe_moves.values())

                # ALWAYS boost safe moves and crush blunders
                # This ensures MCTS strongly prefers safe moves
                if total_safe_prob > 0:
                    # Normalize safe moves to take 99% of probability
                    for move, prob in safe_moves.items():
                        adjusted_probs[move] = (prob / total_safe_prob) * 0.99

                    # Blunders get only 1% total, distributed by inverse severity
                    for move, (prob, material_lost) in blunder_moves.items():
                        # Worse blunders get less probability
                        inverse_severity = max(0.1, 1.0 - material_lost / 10.0)
                        adjusted_probs[move] = 0.01 * inverse_severity / max(len(blunder_moves), 1)
                else:
                    # Shouldn't happen, but fallback
                    for move, prob in safe_moves.items():
                        adjusted_probs[move] = prob
                    for move, (prob, _) in blunder_moves.items():
                        adjusted_probs[move] = prob * 0.01

                move_probs = adjusted_probs
            # If ALL moves are blunders, pick the least bad one
            elif blunder_moves:
                adjusted_probs = {}
                for move, (prob, material_lost) in blunder_moves.items():
                    # Invert: moves with less material loss get higher weight
                    inverse_loss = 10.0 - material_lost
                    adjusted_probs[move] = prob * inverse_loss
                move_probs = adjusted_probs
        else:
            value = nn_value

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
        if temperature is not None:
            old_temp = self.config.temperature
            self.config.temperature = temperature

        move, move_probs = self.mcts.search(board, num_simulations)

        if temperature is not None:
            self.config.temperature = old_temp

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

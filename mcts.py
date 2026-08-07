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
import time
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



# Severity assigned to a move that lets the opponent mate immediately. Deliberately
# above any material loss (a queen is 9.0) so it outranks every other blunder.
MATE_BLUNDER_LOSS = 20.0


def _has_mate_in_1(board: chess.Board) -> bool:
    """
    True if the side to move has a mate in one.

    Cheaper than it looks: is_checkmate() tests is_check() first — a bitmask — and only
    generates legal moves when the king is actually in check, so nearly every reply
    exits immediately. This is used at the root only, over ~40 candidate moves.
    """
    for reply in board.legal_moves:
        board.push(reply)
        mated = board.is_checkmate()
        board.pop()
        if mated:
            return True
    return False


def _blunder_weight(material_lost: float) -> float:
    """
    Relative weight for a blunder, by severity: monotone decreasing, bounded in (0, 1].

    Must never reach zero or go negative. These values become root priors, and a
    negative prior makes PUCT's exploration term negative, so the search would steer
    away from precisely the moves it scored best. The previous formula (10 - loss) had
    no floor and did exactly that for any severity above 10.
    """
    return 1.0 / (1.0 + max(0.0, material_lost))


def is_blunder_move(board: chess.Board, move: chess.Move) -> Tuple[bool, float]:
    """
    Check if a move is a blunder (hangs a piece, allows a bad exchange, or makes a
    bad voluntary capture).

    NOTE: this does NOT detect walking into mate. The only opponent-reply scan below
    is capture-only, so a non-capturing mating reply is invisible to it. Measured on
    positions one ply from mate, the engine plays a mate-allowing move roughly a
    quarter of the time when a safe move existed.

    Returns:
        (is_blunder, material_lost)
    """
    our_color = board.turn  # The color making the move
    opponent_color = not our_color

    # One post-move board, shared by both mate checks below.
    board_after_move = board.copy()
    board_after_move.push(move)

    # Delivering mate is never a blunder, and this is checked FIRST so correctness does
    # not rest on an accident further down: the bad-capture branch below can only fire
    # when board_after.legal_moves is non-empty, which is false after checkmate. That is
    # a coincidence of using legal_moves, and rewriting that scan with attackers() or
    # pseudo-legal generation — an obvious future optimisation on this hot path — would
    # silently resurrect the bug for mating captures.
    if board_after_move.is_checkmate():
        return False, 0.0

    # Allowing an immediate mate is the worst blunder available, and nothing else in
    # this function can see it: the opponent scan further down is capture-only
    # (`if board_copy.is_capture(opp_move)`), so a QUIET mating reply — the common kind
    # — is invisible. Measured on positions one ply from mate where a safe alternative
    # existed, the engine walked into mate in roughly a quarter to a third of them.
    #
    # Checked before the capture branch on purpose: a move that both loses material and
    # allows mate must be scored at mate severity, because severity is what picks the
    # least-bad move in the branch where every option is a blunder.
    if _has_mate_in_1(board_after_move):
        return True, MATE_BLUNDER_LOSS

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

    # (Checkmate is handled at the top of this function. It used to be tested HERE and
    # return (True, 100.0) — reading "opponent is mated" as "we got mated" — which sent
    # every mating move through the blunder branch and crushed its root prior. The
    # engine found mate in 1 in only 8.8% of positions where one existed.)

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


# chess.Board.copy() defaults to stack=True, which rebuilds the whole game history
# ([copy.copy(m) for m in self.move_stack]) on EVERY copy. Measured cost per copy:
# 2.9us at ply 0, 41.7us at ply 16, 140.3us at ply 60 — so search cost grew with game
# length, and by move 30 a "hard" move cost twice what the same search cost at move 1.
#
# Tree nodes never need the full game history: the search tree is only a few ply deep,
# so a short window is enough for in-tree repetition detection. The halfmove clock is a
# plain board field and survives truncation, so fifty/seventy-five-move rules are
# unaffected — only repetition detection sees a shorter horizon.
# Set to False to drop history entirely: measured a further 1.13x, but it disables
# in-tree repetition detection outright. Left at 8 until a strength harness exists to
# confirm that costs nothing. (The top-level anti-repetition check in
# MCTSPlayer.select_move runs on the real board and is unaffected either way.)
TREE_STACK_DEPTH = 8


def leaf_value(board: chess.Board, nn_value: float, weight: float) -> float:
    """
    Value of a non-terminal leaf: a blend of the network's value head and quiescence.

        weight = 0.0  quiescence only    (what ships, and what should keep shipping)
        weight = 1.0  value head only    (skips quiescence; measured 1.23x, and -307 Elo)
        0 < w < 1     w*value_head + (1-w)*quiescence

    THE DEFAULT IS 0.0 AND SHOULD STAY THERE. Using the value head at a leaf was measured
    head-to-head with bench/elo.py, on dual_model_v2 for BOTH arms so that the leaf value
    was the only difference, at 100 simulations:

        weight   arm         Elo vs quiescence      95% CI            games
        0.5      vh-blend         -137.0        [-186.5, -92.3]        240
        1.0      vh-only          -307.1        [-410.3, -234.9]       120

    Monotone in the weight, both far outside noise (each run resolves ~+/-60 Elo). The
    value head is not a viable leaf evaluator here at ANY weight, and this is not a
    training problem — v2's head is well trained (82% teacher agreement, r 0.570 against
    depth-16 Stockfish). A leaf is usually mid-capture-sequence, and a static evaluator
    cannot see the recapture. That is the whole job evaluate_quiescence exists to do.

    THE OFFLINE SCREEN SAID THE OPPOSITE, CONFIDENTLY. On the 1474-position committed
    suite, scored against the depth-16 Stockfish evals cached there as base_cp:

        evaluator                    MSE     pearson r   spearman
        quiescence                 0.1862     0.638       0.576
        value head (v2)            0.1736     0.570       0.514
        0.5*head + 0.5*quiescence  0.1305     0.702       0.630

    The blend wins on every column, and the reasoning behind it was not obviously wrong:
    the two evaluators' residuals correlate only +0.45, quiescence is largely a material
    counter (r 0.769 with plain material) and the head is not (r 0.543), so the blend
    genuinely does carry more information ABOUT THOSE POSITIONS. The flaw is the position
    set. Suite positions are quiet — sampled at ply boundaries of Stockfish games — while
    the positions this function is actually called on are the tactically unstable ones the
    search just descended into. The screen measured the right quantity on the wrong
    distribution, which is the same mistake export_onnx.py made when it verified on
    gaussian noise instead of real piece planes.

    So: r against a quiet-position reference does NOT predict leaf-evaluator quality. Do
    not resurrect this on the strength of an offline metric; it costs about four hours of
    games to find out, and the offline number was 0.702 vs 0.638 in the wrong direction.

    ONLY ABOUT HALF THE LOSS IS THE VALUE HEAD. A control arm (bench/elo.py
    `quiesc-scaled`) played plain quiescence scaled by 0.70 — the exact factor by which
    blending shrinks the leaf value's spread, sd 0.354 against quiescence's 0.504 — which
    is rank-preserving and therefore carries IDENTICAL information. It lost 73.5 Elo,
    95% CI [-140.3, -11.7]. So merely compressing the leaf value costs most of a hundred
    Elo on its own, and the blend was paying that penalty on top of its worse evaluation.

    That matters well beyond the value head, because scaling every leaf value by k is
    equivalent to scaling c_puct by 1/k: PUCT selects on -Q + c_puct*P*sqrt(N)/(1+N), and
    dividing by k leaves c_puct/k (exactly, apart from terminal values, which do not
    scale). The control is therefore approximately c_puct 1.5 -> 2.14, and it cost 73.5
    Elo — so c_puct is a high-leverage parameter here and MORE exploration is worse. Note
    that points the opposite way to the earlier ACPL sweep, which drifted toward more
    exploration and called 8.0 best at p=0.23; that sweep was an underpowered proxy. Any
    future evaluator swapped in here must match quiescence's spread, or it will be
    penalised for its scale on top of its merits.

    Both terms are on the SAME scale and perspective, which is the only reason they can be
    added at all: evaluate_quiescence returns tanh(cp/400) from the side to move, and
    training/train_stockfish.py trains the value head on tanh(eval/4.0) in pawns from the
    side to move — the same transform. Any future attempt must re-check that first.

    Also worth knowing before trying again: the shipped v1 head is far worse than v2's
    (r 0.244, sd 0.140 — it barely leaves zero, having been trained against the old
    stochastic Skill-Level-12 teacher), so any value-head experiment run against the
    SERVED model measures a false negative for the wrong reason.
    """
    if weight <= 0.0:
        from evaluation import evaluate_quiescence
        return evaluate_quiescence(board)

    if weight >= 1.0:
        # These two checks are NOT redundant with MCTSNode.is_terminal, which uses
        # board.is_game_over() — that does not claim draws. evaluate_quiescence tests
        # both and returns 0, so dropping it here would quietly hand the value head
        # positions it has no way to judge: nothing in a 13-plane board tensor encodes
        # repetition or the halfmove clock, so a drawn shuffle would keep scoring as
        # whatever the material happens to be.
        if board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return nn_value

    from evaluation import evaluate_quiescence
    return weight * nn_value + (1.0 - weight) * evaluate_quiescence(board)


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
    # Share of the leaf value taken from the network's value head; see leaf_value().
    # 0.0 = quiescence only, which is what shipped before this knob existed.
    value_head_weight: float = 0.0
    # Carry the search tree between consecutive moves of one game: after our move and
    # the opponent's reply, the matching grandchild subtree becomes the next root, so
    # its visits and values are inherited for free before the new simulations start.
    # Requires the SAME player instance across moves and a board that keeps its move
    # stack; anything else fails closed to a fresh tree. Off by default: flag-off is
    # bit-identical to the pre-flag engine.
    #
    # MEASURED 2026-08-07 (elo_reuse100_vs_current100_models.json): +6.5 Elo, 95% CI
    # [-22.3, +35.4], 240 pairs at 100 sims, int8 on both arms — a null from a run
    # powered to +/-41, so the effect is below ~+35 and probably single digits. The
    # free-sims arithmetic over-promised because the inherited subtree is the line the
    # fresh search re-finds fastest anyway: ~10-20% extra visits concentrated exactly
    # where the priors already point. NOT adopted in serving; the flag stays for any
    # future search where reuse fractions are larger (e.g. after a native kernel).
    tree_reuse: bool = False
    # Wall-clock move budget in seconds; 0 = off (fixed num_simulations, the shipped
    # behaviour). When set, simulations run until the deadline, still capped by
    # num_simulations, and never fewer than one. Latency becomes bounded, and compute
    # shifts toward positions whose simulations are cheap; whether that reallocation
    # helps, hurts, or does nothing is a bench/elo.py question, not an assumption.
    # Deliberately nondeterministic — sims per move vary with position and load.
    #
    # MEASURED 2026-08-07/08, two independent 240-pair runs vs current:100, int8 both
    # arms (elo_timed1300_vs_current100_models.json and -2.json). Run 1 carried a +12%
    # time surplus and scored +45.1 [+18.6,+72.2]; run 2 a -3% deficit and +16.7
    # [-7.3,+40.8]. Time-correcting both at ~127 Elo/doubling leaves the SAME residual,
    # +23.3 and +22.9 — the reallocation itself is worth roughly +10..+30 Elo at this
    # scale and is at worst free. Verdict: prefer a time budget over fixed sims when
    # serving; it also caps tail latency, which fixed sims never can.
    time_budget_s: float = 0.0


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
        prior: float = 0.0,
        own_board: bool = False
    ):
        # own_board=True means the caller built this board solely for us and will not
        # touch it again, so we can keep it instead of copying. Nothing in the search
        # ever leaves a node's board mutated (quiescence pushes and pops in balanced
        # pairs), so the defensive copy was pure overhead: it made every node cost two
        # Board.copy() calls instead of one — over 100k copies per 1300-sim move.
        self.board = board if own_board else board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # P(s, a) from policy network

        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        # Policy priors for every legal move, in policy order. Children are created
        # lazily from this on first selection, so children is a subset of child_priors.
        self.child_priors: Dict[chess.Move, float] = {}
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
        PUCT score, from the PARENT's point of view:

            Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        self.value is stored from THIS node's perspective (the opponent of the parent),
        so it is negated — the parent maximises its own value, not the opponent's.

        Note this method is never actually reached with visit_count == 0: select_child
        creates a child and the simulation immediately backpropagates through it, so
        every existing node has at least one visit. First-play-urgency for a move with
        no node yet is applied by the inline term in select_child, which uses the same
        formula (Q = 0, N = 0). The zero-visit case is kept correct here anyway so the
        two paths cannot drift apart.

        This used to return float('inf') for any unvisited child, which forced the
        search to visit EVERY sibling once before a prior could matter. Measured: in a
        47-legal-move position at 50 simulations, all 47 moves received exactly one
        visit and nothing else happened — the entire budget went on a breadth-first
        sweep in python-chess move-generation order, and the policy network had no
        effect on the result whatsoever. At 200 sims, 32-46% of all selection decisions
        were still being made this way.
        """
        # max(1, ...) matters on the very first simulation, when the root has no visits
        # yet: sqrt(0) would zero every exploration term and hand the choice back to
        # insertion order, which is the failure being fixed.
        exploration = (c_puct * self.prior * math.sqrt(max(1, parent_visits))
                       / (1 + self.visit_count))
        return -self.value + exploration

    def expand(self, move_probs: Dict[chess.Move, float]):
        """
        Mark this node expanded and record the policy prior for each legal move.

        Child nodes are NOT built here — they are created on first selection, in
        select_child. The search only ever reaches a small fraction of them: at 1300
        simulations the eager version created ~50,600 nodes and evaluated 1,300, so
        ~97% of its board copies and is_game_over() calls were pure waste.

        Args:
            move_probs: Policy network probabilities for each legal move
        """
        if self.is_expanded or self.is_terminal:
            return

        # Copied so later prior edits (e.g. Dirichlet noise) cannot reach the caller's
        # dict. Insertion order is preserved and is load-bearing — see select_child.
        self.child_priors = dict(move_probs)
        self.is_expanded = True

    def select_child(self, c_puct: float) -> Tuple[chess.Move, 'MCTSNode']:
        """
        Select the child with highest UCB score, creating it if it does not exist yet.

        Iterates child_priors rather than children so that not-yet-created moves are
        still considered. A missing child has never been visited, so it is scored with
        exactly what ucb_score would return for it: Q = 0 and N = 0, leaving
        c_puct * P * sqrt(N_parent). Computing it inline avoids building a node just to
        ask its score — which is the whole point of lazy expansion.
        """
        best_score = float('-inf')
        best_move = None
        best_child = None
        sqrt_parent = math.sqrt(max(1, self.visit_count))

        for move in self.child_priors:
            child = self.children.get(move)
            score = (c_puct * self.child_priors[move] * sqrt_parent if child is None
                     else child.ucb_score(c_puct, self.visit_count))
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_move is None:
            return None, None

        if best_child is None:
            child_board = self.board.copy(stack=TREE_STACK_DEPTH)
            child_board.push(best_move)
            best_child = MCTSNode(
                board=child_board,
                parent=self,
                move=best_move,
                prior=self.child_priors[best_move],
                own_board=True
            )
            self.children[best_move] = best_child

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
        if not self.child_priors:
            return

        noise = np.random.dirichlet([alpha] * len(self.child_priors))
        for i, move in enumerate(self.child_priors):
            noised = (1 - epsilon) * self.child_priors[move] + epsilon * noise[i]
            self.child_priors[move] = noised
            # Normally no children exist yet (noise is applied right after expand),
            # but keep any that do in sync with the priors they were built from.
            child = self.children.get(move)
            if child is not None:
                child.prior = noised


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

        # Tree-reuse state (config.tree_reuse). _last_root/_last_fen are written by
        # search(); _last_played is written by note_played() AFTER the post-search
        # vetoes, because the move the search chose is not always the move played.
        self._last_root = None
        self._last_fen = None
        self._last_played = None
        self.reuse_hits = 0

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

        # Reuse the subtree from the previous search when this position is exactly
        # (previous position + our played move + the opponent's reply); otherwise a
        # fresh root, which is also the only path when the flag is off.
        root = self._promote_reused_root(board) if self.config.tree_reuse else None
        if root is None:
            root = MCTSNode(board)

        root_eval_fn = self.root_evaluate_fn or self.evaluate_fn
        move_probs, _ = root_eval_fn(board)
        if root.is_expanded:
            # Promoted subtree root. It was expanded with the FAST priors (no
            # heuristic boosts, no blunder scan, no mate dominance — those are
            # root-only by design), so replace its priors with the root pipeline's
            # while keeping the inherited visit statistics. Existing children are
            # kept in sync exactly as add_dirichlet_noise does.
            root.child_priors = dict(move_probs)
            for mv, child in root.children.items():
                child.prior = root.child_priors.get(mv, 0.0)
        else:
            root.expand(move_probs)

        # Add exploration noise at root during training
        if self.config.add_noise:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Run simulations (use fast evaluate for non-root nodes)
        if self.config.time_budget_s > 0:
            # Wall-clock budget: simulate until the deadline, num_sims as a hard
            # cap, never fewer than one. The perf_counter call is ~70ns against a
            # ~2.4ms simulation, so the check itself costs nothing measurable.
            deadline = time.perf_counter() + self.config.time_budget_s
            sims_run = 0
            while sims_run < num_sims and (sims_run == 0
                                           or time.perf_counter() < deadline):
                self._run_simulation(root)
                sims_run += 1
        else:
            for _ in range(num_sims):
                self._run_simulation(root)

        # Calculate move probabilities from visit counts
        # Iterate the priors, not the children: with lazy expansion a move that was
        # never selected has no node, and it must still appear here with zero visits so
        # the returned distribution keeps exactly the keys (and order) it always had.
        move_visits = {
            move: (root.children[move].visit_count if move in root.children else 0)
            for move in root.child_priors
        }

        total_visits = sum(move_visits.values())
        move_probs = {
            move: visits / total_visits
            for move, visits in move_visits.items()
        }

        # Select move based on temperature
        best_move = self._select_move(root, move_probs)

        if self.config.tree_reuse:
            # Keep the tree for the next call. _last_played stays None until
            # note_played(): the vetoes in MCTSPlayer.select_move can override
            # best_move, and reusing the subtree of a move that was NOT played
            # would search the wrong position's tree.
            self._last_root = root
            self._last_fen = board.fen()
            self._last_played = None

        return best_move, move_probs

    def _run_simulation(self, root: MCTSNode):
        """One simulation: select down to a leaf, evaluate/expand it, backpropagate.

        (The old inline loop also built a `path` list on every simulation that
        nothing ever read — dropped in the extraction; verify.py gates the change.)
        """
        node = root
        while node.is_expanded and not node.is_terminal:
            _, node = node.select_child(self.config.c_puct)
        if node.is_terminal:
            value = node.terminal_value
        else:
            # Fast path — no heuristics/blunder checks below the root.
            move_probs, value = self.evaluate_fn(node.board)
            node.expand(move_probs)
        node.backpropagate(value)

    def _promote_reused_root(self, board: chess.Board) -> Optional[MCTSNode]:
        """
        Return the previous search's grandchild subtree for `board`, or None.

        Matches only when board is exactly (last searched position + the move
        note_played() recorded + one opponent reply), verified by move stack AND by
        FEN — a same-looking stack after a new game or an undo fails the FEN check
        and falls back to a fresh tree. Every failure mode here is fail-closed.
        """
        last_root, last_fen = self._last_root, self._last_fen
        played = self._last_played
        if last_root is None or played is None or len(board.move_stack) < 2:
            return None
        if board.move_stack[-2] != played:
            return None
        prev = board.copy()
        reply = prev.pop()
        prev.pop()
        if prev.fen() != last_fen:
            return None
        child = last_root.children.get(played)
        node = child.children.get(reply) if child is not None else None
        if node is None or node.is_terminal or not node.is_expanded:
            return None
        # Detach: backpropagation walks node.parent and must stop at the new root —
        # attached, every new simulation would also pollute the dead old tree's
        # statistics from the wrong perspective. Dropping the reference also lets
        # everything except this subtree be garbage-collected.
        node.parent = None
        node.move = None
        node.prior = 0.0
        # Tree boards carry a TREE_STACK_DEPTH-truncated move stack. As root this
        # node must see the REAL board — full history, exactly like a fresh root —
        # or root-level repetition behaviour would differ from the fresh-tree path.
        node.board = board.copy()
        self.reuse_hits += 1
        return node

    def note_played(self, move: Optional[chess.Move]):
        """Record the move actually PLAYED (post-veto) for the next search's reuse."""
        if self._last_root is not None:
            self._last_played = move

    def invalidate_reuse(self):
        """Drop any carried tree — the next search starts fresh."""
        self._last_root = self._last_fen = self._last_played = None

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
    # Collected in rule 5 (where the post-move board already exists) and re-applied in
    # rule 12 after every other pass has run. Collecting here keeps rule 12 free.
    mating_moves = set()

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
            mating_moves.add(move)
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

    # --- 12. Mate dominance (must be LAST) ---
    # Rule 5 raises a mating move to 0.95, but that is a floor applied mid-pipeline and
    # four later passes multiply priors down without checking for mate: rule 7's king
    # penalty (x0.4), _boost_development's early-queen penalty (x0.6),
    # _penalize_king_shelter_weakening (x0.5) and _boost_simplification (x0.5).
    # Measured, they demoted the mate in 4 of 5 known misses — 0.95 -> 0.38 for a mating
    # king move, -> 0.57 for a mating queen move, -> 0.475 for a mating pawn push.
    #
    # A floor is not enough even when nothing demotes it: in the fifth case the network
    # gave a non-mating move a raw prior of 0.9736, above 0.95. So mating moves are
    # placed strictly ABOVE the current maximum rather than clamped to a constant.
    #
    # This matters because the search cannot recover from a bad prior here: quiescence
    # returns tanh(cp/400), which saturates, so a forced mate and an ordinary winning
    # move both back up as Q = -1.0 and visits track priors alone. Raising simulations
    # does not help — 4000 sims still missed 3 of the 5.
    #
    # Provably inert when no mate exists: the dict is untouched, and with temperature=0
    # and no Dirichlet noise the search is deterministic.
    if mating_moves:
        ceiling = max(move_probs.values())
        for move in mating_moves:
            move_probs[move] = ceiling * 2.0

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
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device

        # torch is imported only when there is actually a torch model to place. With
        # ONNX inference and no checkpoint, the serving path never needs it.
        if model is not None:
            import torch
            if self.device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
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
        # inference is the torch-free module; importing it must not pull torch in.
        from inference import legal_move_indices, get_onnx_session

        # Resolve ONNX BEFORE looking at self.model. A deployment can ship only the
        # exported .onnx and no torch checkpoint, in which case self.model is None but
        # we still have real weights — checking self.model first would silently answer
        # every position with a uniform random policy while looking perfectly healthy.
        session = get_onnx_session()

        if session is None and self.model is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return {}, 0.0
            return {m: 1.0 / len(legal_moves) for m in legal_moves}, 0.0

        # One pass over the legal moves gives both the mask indices and the gather
        # indices. This used to be two passes plus a 4288-element Python write loop.
        moves, idxs = legal_move_indices(board)
        if not moves:
            return {}, 0.0

        # The export is of the CNN, so use ONNX when there is no torch model at all, or
        # when the torch model is the CNN architecture it was exported from. is_cnn_model
        # compares against a torch class, so only consult it once a torch model exists —
        # otherwise a torch-free deployment would import torch just to answer this.
        use_onnx = session is not None
        if use_onnx and self.model is not None:
            from neural_network import is_cnn_model
            use_onnx = is_cnn_model(self.model)

        # Preferred path: onnxruntime, which is faster than torch at batch size 1.
        # Softmax is taken over only the legal logits, which is identical to masking
        # the full 4288-wide vector with -inf and softmaxing that (the illegal entries
        # contribute exactly zero to the sum) while doing a fraction of the work.
        if use_onnx:
            from features import board_to_tensor_2d
            x = board_to_tensor_2d(board).astype(np.float32, copy=False)[None]
            logits, nn_value = session.run(x)
            sel = logits[0][idxs]
            sel = sel - sel.max()
            np.exp(sel, out=sel)
            sel /= sel.sum()
            return (dict(zip(moves, (float(p) for p in sel))),
                    float(np.reshape(nn_value, -1)[0]))

        import torch
        from neural_network import _mask_from_indices, is_cnn_model
        if is_cnn_model(self.model):
            from features import board_to_tensor_2d
            x = torch.from_numpy(board_to_tensor_2d(board)).float().to(self.device).unsqueeze(0)
        else:
            x = torch.from_numpy(board_to_tensor(board)).float().to(self.device).unsqueeze(0)
        mask = _mask_from_indices(idxs).to(self.device).unsqueeze(0)

        with torch.no_grad():
            policy_probs, nn_value = self.model.get_policy_value(x, mask)

        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        move_probs = dict(zip(moves, (float(p) for p in policy_probs[idxs])))

        return move_probs, nn_value.item()

    def _evaluate_fast(
        self,
        board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """Fast evaluation for non-root nodes: NN policy + leaf value.

        The value used to be evaluate_quiescence unconditionally, with the value head
        computed by the forward pass above and thrown away. config.value_head_weight now
        decides the mix — see leaf_value(). It defaults to 0.0, i.e. quiescence only, so
        this is inert until something sets it."""
        move_probs, nn_value = self._nn_forward(board)

        value = leaf_value(board, nn_value, self.config.value_head_weight)

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
        move_probs, nn_value = self._nn_forward(board)

        # Heuristic policy adjustments (only at root — too expensive per-sim)
        move_probs = _apply_heuristic_boosts(board, move_probs)

        # Kept in step with _evaluate_fast so the two cannot drift, though MCTS.search
        # discards the root value (`move_probs, _ = root_eval_fn(board)`) — the root is
        # never backpropagated through, only expanded. Costs one call per move.
        value = leaf_value(board, nn_value, self.config.value_head_weight)

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
                    adjusted_probs[move] = (0.01 * _blunder_weight(material_lost)
                                            / max(len(blunder_moves), 1))
            else:
                for move, prob in safe_moves.items():
                    adjusted_probs[move] = prob
                for move, (prob, _) in blunder_moves.items():
                    adjusted_probs[move] = prob * 0.01
            move_probs = adjusted_probs
        elif blunder_moves:
            # EVERY move loses something, so this branch chooses the least-bad one —
            # its ordering matters most exactly when the position is worst. It used to
            # compute (10.0 - material_lost) with no floor, so any severity above 10
            # produced NEGATIVE priors: measured priors of -85.5 and distribution sums
            # of -167 under the old mate semantics. _blunder_weight is bounded in (0,1]
            # and cannot degenerate, whatever severity is handed to it.
            adjusted_probs = {}
            for move, (prob, material_lost) in blunder_moves.items():
                adjusted_probs[move] = prob * _blunder_weight(material_lost)
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
                # No search ran, so any carried tree is now one move stale.
                self.mcts.invalidate_reuse()
                if return_policy:
                    return tb_move, {tb_move: 1.0}
                return tb_move

        # Fallback: lone king eval-based search (when tablebase unavailable)
        lone_king_move = _select_lone_king_mate_move(board)
        if lone_king_move is not None:
            self.mcts.invalidate_reuse()
            if return_policy:
                return lone_king_move, {lone_king_move: 1.0}
            return lone_king_move

        if temperature is not None:
            old_temp = self.config.temperature
            self.config.temperature = temperature

        move, move_probs = self.mcts.search(board, num_simulations)

        if temperature is not None:
            self.config.temperature = old_temp

        # Mate veto: never hand the opponent an immediate mate when any alternative
        # exists. This MUST live here, after the search, because demoting the prior
        # cannot prevent it — measured, 7.9% of sharp positions still walked into mate
        # with the prior demotion alone. Two mechanisms defeat the prior:
        #   1. ucb_score returns float('inf') for an unvisited child, so every move
        #      gets a mandatory first visit no matter how small its prior; and
        #   2. the child's value comes from evaluate_quiescence, which searches only
        #      CAPTURES, so a quiet mating reply is invisible and the move evaluates as
        #      perfectly healthy.
        # Once visited with a healthy Q, exploitation carries it regardless of prior.
        # Cheap: only runs when the chosen move actually allows mate, which is rare.
        if move and move_probs and not board.is_game_over():
            test_board = board.copy()
            test_board.push(move)
            if _has_mate_in_1(test_board):
                for alt_move, _ in sorted(move_probs.items(), key=lambda x: -x[1]):
                    if alt_move == move:
                        continue
                    alt_board = board.copy()
                    alt_board.push(alt_move)
                    if not _has_mate_in_1(alt_board):
                        move = alt_move
                        break

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
                    # Also require the alternative not to allow mate, or this loop
                    # could undo the veto above by swapping back to a mate-allowing
                    # move purely to dodge a repetition. A draw beats a loss.
                    if (not alt_board.can_claim_draw()
                            and not alt_board.is_repetition(2)
                            and not _has_mate_in_1(alt_board)):
                        move = alt_move
                        break

        # Post-veto, so the recorded move is the one that will actually appear on
        # the board — tree reuse keys its continuation check on it.
        self.mcts.note_played(move)

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

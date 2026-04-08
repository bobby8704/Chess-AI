#!/usr/bin/env python3
"""
Supervised Training for Chess AI - EXPANDED VERSION

This script trains the model on EXPERT chess games first (imitation learning).
The model needs to learn basic chess patterns before self-play can be effective.

DATA SOURCES:
1. 30+ grandmaster games from expert_data.py
2. 50+ tactical puzzles (forks, pins, skewers, mates)
3. 50+ opening positions with best moves
4. 100+ endgame positions with correct technique

The problem with pure self-play from scratch:
- MCTS with few simulations plays nearly randomly
- Training on random moves teaches random play
- Vicious cycle - never learns good chess

Solution:
1. First train supervised on expert games (this script)
2. Then fine-tune with Stockfish teacher (train_stockfish.py)
3. Finally self-play refinement (train_clean.py)
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import io
from datetime import datetime
from pathlib import Path

print("Loading modules...", flush=True)

from neural_network import DualNet, POLICY_OUTPUT_SIZE, encode_move, get_legal_move_mask
from features import board_to_tensor

# Import expanded expert data
try:
    from expert_data import (
        EXPERT_GAMES_PGN as EXPANDED_GAMES_PGN,
        TACTICAL_POSITIONS as EXPANDED_TACTICS,
        OPENING_POSITIONS as EXPANDED_OPENINGS,
        ENDGAME_POSITIONS as EXPANDED_ENDGAMES,
    )
    USE_EXPANDED_DATA = True
    print("Loaded EXPANDED expert data!", flush=True)
except ImportError:
    USE_EXPANDED_DATA = False
    print("Using built-in expert data (expanded data not found)", flush=True)

# ==================== Expert Chess Games ====================

# Extensive collection of expert games - more games = better learning
EXPERT_GAMES_PGN = """
[Event "Italian Game Classic"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 O-O 8. O-O d6 9. Bg5 h6 10. Bh4 Be6 11. Bb3 Bxc3 12. bxc3 Bxb3 13. axb3 1-0

[Event "Queens Gambit Declined"]
[Result "1-0"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Bd3 c6 8. O-O dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 11. Nxd5 cxd5 12. Bd3 1-0

[Event "Sicilian Najdorf"]
[Result "1-0"]
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5 7. Nb3 Be7 8. O-O O-O 9. Be3 Be6 10. f3 Nbd7 11. Qd2 Rc8 12. Rfd1 1-0

[Event "French Defense"]
[Result "1-0"]
1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. e5 Nfd7 5. f4 c5 6. Nf3 Nc6 7. Be3 cxd4 8. Nxd4 Bc5 9. Qd2 O-O 10. O-O-O a6 11. h4 Nxd4 12. Bxd4 1-0

[Event "Reti Opening"]
[Result "1-0"]
1. Nf3 d5 2. g3 Nf6 3. Bg2 c6 4. O-O Bg4 5. d3 Nbd7 6. Nbd2 e6 7. e4 dxe4 8. dxe4 Bc5 9. Qe2 O-O 10. h3 Bh5 11. e5 Nd5 12. Nc4 1-0

[Event "Ruy Lopez"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 1-0

[Event "Scotch Game"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 Bc5 5. Be3 Qf6 6. c3 Nge7 7. Bc4 O-O 8. O-O d6 9. Nd2 Nxd4 10. Bxd4 Bxd4 11. cxd4 Be6 12. Bxe6 fxe6 1-0

[Event "London System"]
[Result "1-0"]
1. d4 d5 2. Bf4 Nf6 3. e3 e6 4. Nf3 c5 5. c3 Nc6 6. Nbd2 Bd6 7. Bg3 O-O 8. Bd3 Re8 9. O-O Bxg3 10. hxg3 e5 11. dxe5 Nxe5 12. Nxe5 Rxe5 1-0

[Event "Caro Kann Defense"]
[Result "1-0"]
1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7 8. Bd3 Bxd3 9. Qxd3 e6 10. Bf4 Qa5+ 11. Bd2 Qc7 12. O-O-O Ngf6 1-0

[Event "Kings Indian Defense"]
[Result "1-0"]
1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. f3 f5 11. Be3 f4 12. Bf2 g5 1-0

[Event "English Opening"]
[Result "1-0"]
1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Nb6 6. Nf3 Nc6 7. O-O Be7 8. d3 O-O 9. Be3 Be6 10. Rc1 f6 11. Na4 Nxa4 12. Qxa4 Bd5 1-0

[Event "Slav Defense"]
[Result "1-0"]
1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. e3 Nbd7 6. Bd3 dxc4 7. Bxc4 b5 8. Bd3 Bb7 9. O-O Be7 10. e4 b4 11. Na4 c5 12. e5 Nd5 1-0

[Event "Nimzo Indian"]
[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 9. bxc3 dxc4 10. Bxc4 Qc7 11. Bd3 e5 12. Qc2 Re8 1-0

[Event "Pirc Defense"]
[Result "1-0"]
1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. f4 Bg7 5. Nf3 O-O 6. Bd3 Nc6 7. O-O e5 8. fxe5 dxe5 9. d5 Nd4 10. Nxd4 exd4 11. Nb5 Nh5 12. Qf3 f5 1-0

[Event "Scandinavian Defense"]
[Result "1-0"]
1. e4 d5 2. exd5 Qxd5 3. Nc3 Qa5 4. d4 Nf6 5. Nf3 Bf5 6. Bc4 e6 7. Bd2 c6 8. Nd5 Qd8 9. Nxf6+ gxf6 10. O-O Nd7 11. Re1 Nb6 12. Bb3 Be7 1-0

[Event "Catalan Opening"]
[Result "1-0"]
1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2 Be7 5. Nf3 O-O 6. O-O dxc4 7. Qc2 a6 8. a4 Bd7 9. Qxc4 Bc6 10. Bg5 Nbd7 11. Nc3 h6 12. Bxf6 Nxf6 1-0

[Event "Two Knights Defense"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Na5 6. Bb5+ c6 7. dxc6 bxc6 8. Be2 h6 9. Nf3 e4 10. Ne5 Bd6 11. d4 exd3 12. Nxd3 O-O 1-0

[Event "Four Knights Game"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. Bb5 Bb4 5. O-O O-O 6. d3 d6 7. Bg5 Bxc3 8. bxc3 Qe7 9. Re1 Nd8 10. d4 Ne6 11. Bc1 Rd8 12. Bf1 c6 1-0

[Event "Petroff Defense"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nf6 3. Nxe5 d6 4. Nf3 Nxe4 5. d4 d5 6. Bd3 Nc6 7. O-O Be7 8. c4 Nb4 9. Be2 O-O 10. Nc3 Bf5 11. a3 Nxc3 12. bxc3 Nc6 1-0

[Event "Vienna Game"]
[Result "1-0"]
1. e4 e5 2. Nc3 Nf6 3. f4 d5 4. fxe5 Nxe4 5. Nf3 Nc6 6. d3 Nxc3 7. bxc3 Be7 8. d4 O-O 9. Bd3 f6 10. exf6 Bxf6 11. O-O Bg4 12. Qe1 Qd7 1-0

[Event "Bird Opening"]
[Result "1-0"]
1. f4 d5 2. Nf3 Nf6 3. e3 g6 4. Be2 Bg7 5. O-O O-O 6. d3 c5 7. Qe1 Nc6 8. c3 Qc7 9. Nbd2 b6 10. e4 dxe4 11. dxe4 e5 12. f5 gxf5 1-0

[Event "Giuoco Piano"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 Nf6 5. O-O d6 6. c3 O-O 7. Re1 a6 8. h3 Ba7 9. Nbd2 Be6 10. Bb3 Bxb3 11. Qxb3 Qd7 12. Nf1 Nh5 1-0

[Event "Evans Gambit"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d6 8. cxd4 Bb6 9. Nc3 Nf6 10. Bg5 h6 11. Bh4 O-O 12. e5 dxe5 1-0

[Event "Dutch Defense"]
[Result "1-0"]
1. d4 f5 2. g3 Nf6 3. Bg2 e6 4. Nf3 Be7 5. O-O O-O 6. c4 d6 7. Nc3 Qe8 8. Re1 Qh5 9. e4 fxe4 10. Nxe4 Nxe4 11. Rxe4 Nc6 12. d5 exd5 1-0

[Event "Grunfeld Defense"]
[Result "1-0"]
1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 Bg7 7. Nf3 c5 8. Be3 Qa5 9. Qd2 O-O 10. Rc1 cxd4 11. cxd4 Qxd2+ 12. Kxd2 Nc6 1-0

[Event "Benoni Defense"]
[Result "1-0"]
1. d4 Nf6 2. c4 c5 3. d5 e6 4. Nc3 exd5 5. cxd5 d6 6. e4 g6 7. Nf3 Bg7 8. Be2 O-O 9. O-O Re8 10. Nd2 Na6 11. f3 Nc7 12. a4 b6 1-0

[Event "Alekhine Defense"]
[Result "1-0"]
1. e4 Nf6 2. e5 Nd5 3. d4 d6 4. Nf3 g6 5. Bc4 Nb6 6. Bb3 Bg7 7. exd6 cxd6 8. O-O O-O 9. h3 Nc6 10. Re1 Na5 11. c3 Nxb3 12. axb3 Bf5 1-0

[Event "Kings Gambit"]
[Result "1-0"]
1. e4 e5 2. f4 exf4 3. Nf3 g5 4. h4 g4 5. Ne5 Nf6 6. Bc4 d5 7. exd5 Bd6 8. d4 Nh5 9. O-O Qxh4 10. Qe1 Qxe1 11. Rxe1 f6 12. Nd3 f3 1-0

[Event "Philidor Defense"]
[Result "1-0"]
1. e4 e5 2. Nf3 d6 3. d4 Nf6 4. Nc3 Nbd7 5. Bc4 Be7 6. O-O O-O 7. a4 c6 8. Re1 Qc7 9. h3 exd4 10. Nxd4 Ne5 11. Ba2 Ng6 12. Qf3 Bd7 1-0

[Event "Center Game"]
[Result "1-0"]
1. e4 e5 2. d4 exd4 3. Qxd4 Nc6 4. Qe3 Nf6 5. Nc3 Bb4 6. Bd2 O-O 7. O-O-O Re8 8. Qg3 d6 9. f3 Be6 10. Nh3 d5 11. e5 Ne4 12. fxe4 dxe4 1-0
"""

# Common opening positions with best moves (for more varied training)
# These teach basic principles: develop pieces, control center, castle early
OPENING_POSITIONS = [
    # Starting position - develop center pawns and knights
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", ["e2e4", "d2d4", "c2c4", "g1f3"]),

    # After 1.e4 - play in center
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", ["e7e5", "c7c5", "e7e6", "c7c6"]),

    # After 1.d4
    ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", ["d7d5", "g8f6", "e7e6"]),

    # After 1.e4 e5 - develop knight
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["g1f3", "b1c3", "f1c4"]),

    # After 1.e4 e5 2.Nf3 - develop knight
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", ["b8c6", "g8f6"]),

    # Italian setup - bishop to c4
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ["f1c4", "f1b5"]),

    # After Bc4 - develop pieces
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", ["g8f6", "f8c5"]),

    # Sicilian - White plays Nf3
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["g1f3", "b1c3"]),

    # Queens Gambit position
    ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2", ["e7e6", "c7c6"]),

    # Time to castle! (White ready)
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e1g1"]),

    # Time to castle! (Black ready)
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5", ["e8g8"]),

    # Both castled - develop remaining pieces
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 2 6", ["b1c3", "c1e3", "b1d2"]),

    # Control center with pawns
    ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", ["c2c4", "g1f3"]),
]

# More extensive middlegame positions to teach strategy
MIDDLEGAME_POSITIONS = [
    # Develop knights before bishops
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", ["g1f3", "b1c3"]),

    # Don't move queen too early
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["g1f3", "b1c3", "f1c4"]),  # NOT Qh5

    # Pin the knight - good tactic
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["c1g5", "e1g1", "d2d3"]),

    # Attack the pinned piece
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 5 5", ["c1g5", "b1c3"]),

    # Develop rooks to open files
    ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 7", ["b1c3", "c1e3", "f1e1"]),

    # Control the center
    ("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3", ["e5d4", "d7d6"]),

    # Trade when ahead in material
    ("r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bP5/2N1PN2/PP2BPPP/R1BQK2R w KQ - 0 7", ["c3d5", "e1g1"]),

    # Protect the king
    ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 w - - 4 7", ["h2h3", "b1c3"]),

    # Fianchetto bishop
    ("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 0 2", ["d7d5", "g8f6"]),

    # Push pawns in the center
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 0 4", ["e5d4", "d7d6"]),

    # Knight outpost
    ("r1bq1rk1/ppp1bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", ["c3d5", "c1e3"]),

    # Rook on the 7th rank
    ("r4rk1/pppq1ppp/2n2n2/3pp3/2PP4/2N2N2/PP2QPPP/R1B1R1K1 w - - 0 10", ["e1e7", "c4d5"]),

    # Queen and rook battery
    ("r1bq1rk1/ppp2ppp/2n2n2/3pp3/2PP4/2N2N2/PP2QPPP/R1B1R1K1 w - - 0 8", ["e1e2", "c4d5"]),

    # Don't leave pieces hanging
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", ["g8f6", "f8c5"]),  # NOT Nxe4 which loses

    # Create passed pawns in endgame
    ("8/pp3ppp/8/3k4/3P4/8/PP3PPP/4K3 w - - 0 1", ["d4d5", "f2f4"]),

    # King activity in endgame
    ("8/8/4k3/8/3PK3/8/8/8 w - - 0 1", ["e4e5", "d4d5"]),

    # Protect backward pawns
    ("r1bq1rk1/ppp2ppp/2np1n2/4p3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 7", ["f8e8", "c8e6"]),

    # Attack weak pawns
    ("r1bq1rk1/p1p2ppp/2np1n2/1p2p3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8", ["c4b5", "c3d5"]),

    # Connected rooks
    ("r4rk1/ppp2ppp/2n2n2/3qp3/3P4/2N2N2/PP2QPPP/R1B2RK1 w - - 0 10", ["a1d1", "f1d1"]),

    # Bishop pair advantage
    ("r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bP5/2N1PN2/PP1BBPPP/R2QK2R b KQ - 0 7", ["d5c4", "f8e8"]),
]

# Tactical positions - must find the right move
TACTICAL_POSITIONS = [
    # Fork with knight
    ("r1bqkb1r/pppp1ppp/2n2n2/4N3/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["e5f7"]),  # Fork king and rook

    # Back rank mate threat
    ("6k1/ppp2ppp/8/8/8/8/PPP2PPP/R5K1 w - - 0 1", ["a1a8"]),

    # Pin and win
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 5", ["c1g5"]),

    # Discovered attack
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["f3g5"]),

    # Remove the defender
    ("r1b1kb1r/ppppqppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5", ["c4f7"]),

    # Skewer
    ("r3k2r/ppp2ppp/2n2n2/3qp3/1b1P4/2N2N2/PP2BPPP/R1BQ1RK1 w kq - 0 8", ["e2b5"]),

    # Double attack
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 0 4", ["e5d4"]),

    # Capture the undefended piece
    ("r1bqkb1r/pppp1ppp/2n5/4p2n/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["f3e5"]),
]

# Endgame positions - teach endgame technique (EXPANDED - critical for winning!)
ENDGAME_POSITIONS = [
    # ===== KING AND PAWN ENDGAMES =====
    # King in front of pawn = win
    ("8/8/4k3/8/4K3/4P3/8/8 w - - 0 1", ["e4d5", "e4f5"]),  # Gain opposition
    ("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", ["e3d4", "e3f4"]),  # Advance with king
    ("8/8/8/8/4k3/8/4PK2/8 w - - 0 1", ["f2e3", "f2f3"]),  # King supports pawn
    ("8/4k3/8/4K3/4P3/8/8/8 w - - 0 1", ["e5d5", "e5f5"]),  # Key square control
    ("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1", ["e3d4", "e3f4"]),  # Push forward

    # Opposition - critical endgame concept
    ("8/8/4k3/8/4K3/8/4P3/8 w - - 0 1", ["e4d4", "e4f4"]),  # Take opposition
    ("8/8/8/4k3/8/4K3/8/4P3 w - - 0 1", ["e3d4", "e3f4"]),  # Advance
    ("8/8/3k4/8/3K4/8/3P4/8 w - - 0 1", ["d4c5", "d4e5"]),  # Outflank

    # Passed pawn - push it!
    ("8/8/8/3Pk3/8/3K4/8/8 w - - 0 1", ["d5d6"]),  # Push passed pawn
    ("8/8/3P4/4k3/8/3K4/8/8 w - - 0 1", ["d6d7"]),  # Keep pushing
    ("8/3P4/4k3/8/8/3K4/8/8 w - - 0 1", ["d7d8q"]),  # Queen it!

    # ===== BASIC CHECKMATES =====
    # King + Queen vs King - box in the king
    ("8/8/8/4k3/8/8/8/4K2Q w - - 0 1", ["h1h5", "h1e1"]),  # Cut off king
    ("8/8/8/3Qk3/8/8/4K3/8 w - - 0 1", ["d5d4", "d5e5"]),  # Box in
    ("3k4/3Q4/8/8/8/8/8/4K3 w - - 0 1", ["d7d6", "e1e2"]),  # Force to edge
    ("k7/1Q6/8/8/8/8/8/4K3 w - - 0 1", ["b7a7", "b7b8"]),  # Deliver mate

    # King + Rook vs King - cut off and push
    ("8/8/8/4k3/8/8/8/4K2R w - - 0 1", ["h1h5", "h1a1"]),  # Cut off ranks
    ("4k3/4R3/8/8/8/8/8/4K3 w - - 0 1", ["e7e6", "e1f2"]),  # Push to edge
    ("k7/R7/8/8/8/8/8/1K6 w - - 0 1", ["a7a8"]),  # Checkmate

    # King + 2 Rooks vs King - ladder mate
    ("4k3/8/8/8/8/8/8/R3K2R w - - 0 1", ["a1a8", "h1h8"]),  # Start ladder
    ("1k6/R7/1R6/8/8/8/8/4K3 w - - 0 1", ["a7a8"]),  # Finish ladder

    # ===== ROOK ENDGAMES =====
    # Active rook - rook belongs behind passed pawns
    ("8/8/4k3/8/4P3/8/8/R3K3 w - - 0 1", ["a1a8", "e4e5"]),  # Rook supports pawn
    ("8/8/4k3/4P3/8/8/8/R3K3 w - - 0 1", ["a1e1", "e5e6"]),  # Push pawn
    ("8/4P3/4k3/8/8/8/8/R3K3 w - - 0 1", ["a1e1", "e7e8q"]),  # Promote

    # Cut off the king
    ("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1", ["e2e6", "e2a2"]),  # Cut off king
    ("8/4k3/4R3/8/4K3/8/8/8 w - - 0 1", ["e6a6", "e4e5"]),  # Restrict king

    # Lucena position - building the bridge
    ("1K6/1P6/8/8/8/8/1k4r1/4R3 w - - 0 1", ["e1e4"]),  # Bridge step 1
    ("1K6/1P6/8/8/4R3/8/1k4r1/8 w - - 0 1", ["e4a4"]),  # Continue bridge

    # Philidor position - draw technique (as defender)
    ("8/3k4/8/3KP3/8/8/8/r7 b - - 0 1", ["a1a6"]),  # Third rank defense

    # ===== QUEEN ENDGAMES =====
    # Queen vs advanced pawn
    ("8/P7/8/8/8/8/8/k1K5 w - - 0 1", ["a7a8q"]),  # Just queen
    ("8/5Q2/8/8/8/p7/8/k1K5 w - - 0 1", ["f7a2"]),  # Stop pawn, then capture

    # ===== BISHOP ENDGAMES =====
    # Bishop and pawn - same color complex wins
    ("8/8/8/5k2/8/4KB2/4P3/8 w - - 0 1", ["e3d4", "e2e4"]),  # Right color bishop
    ("8/8/4k3/8/4P3/4KB2/8/8 w - - 0 1", ["e4e5", "e3d4"]),  # Advance pawn

    # Two bishops checkmate - coordinate bishops
    ("8/8/8/4k3/8/2B1K3/8/2B5 w - - 0 1", ["c3d4", "e3e4"]),  # Box in king
    ("3k4/8/3K4/2B5/8/8/B7/8 w - - 0 1", ["c5b6", "a2c4"]),  # Force to corner

    # ===== KNIGHT ENDGAMES =====
    # Knight and pawn
    ("8/8/8/4k3/8/4K3/4P3/4N3 w - - 0 1", ["e3d4", "e2e4"]),  # Support pawn
    ("8/8/4k3/8/4P3/4K3/8/4N3 w - - 0 1", ["e1c2", "e4e5"]),  # Knight + pawn

    # Two knights cannot checkmate (but can with pawn help)
    ("8/8/8/4k3/8/4K3/8/2N1N3 w - - 0 1", ["c1d3", "e1c2"]),  # Coordinate

    # ===== AVOID STALEMATE =====
    # Don't stalemate when winning!
    ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", ["h1a1", "h1h8"]),  # Mate, not stalemate
    ("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", ["a1h8", "a1g7"]),  # Check to win, not stale
    ("k7/2Q5/1K6/8/8/8/8/8 w - - 0 1", ["c7c8", "c7a7"]),  # Mate patterns

    # ===== CONVERTING MATERIAL ADVANTAGE =====
    # Trade pieces when up material
    ("r3k3/8/8/8/8/8/4R3/4K3 w - - 0 1", ["e2e8"]),  # Trade rooks when up
    ("4k3/8/8/8/2b5/8/4B3/4K3 w - - 0 1", ["e2c4"]),  # Trade bishops

    # Simplify to winning endgame
    ("r3k3/ppp2ppp/8/8/8/8/PPP2PPP/R3K3 w - - 0 1", ["a1a8"]),  # Trade rooks

    # ===== ACTIVATING THE KING =====
    # King is a fighting piece in endgame!
    ("8/8/8/4k3/4p3/8/4K3/8 w - - 0 1", ["e2e3", "e2d3"]),  # Activate king
    ("8/8/8/8/4pk2/8/4K3/8 w - - 0 1", ["e2f3", "e2e3"]),  # Block pawn
    ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", ["e3e4", "e3d4"]),  # Centralize king
]


def parse_pgn_games(pgn_text):
    """Parse PGN text and return list of games."""
    games = []
    pgn_io = io.StringIO(pgn_text)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)

    return games


def extract_training_data_from_game(game):
    """Extract (board_state, best_move) pairs from a game."""
    training_data = []
    board = game.board()

    # Determine game result for value targets
    result = game.headers.get("Result", "*")
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0

    for move in game.mainline_moves():
        # Create training sample: current position -> move made
        tensor = board_to_tensor(board)

        # Create policy target (one-hot for the actual move)
        policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
        move_idx = encode_move(move)
        if move_idx < POLICY_OUTPUT_SIZE:
            policy_target[move_idx] = 1.0

        # Value from current player's perspective
        value = white_value if board.turn == chess.WHITE else -white_value

        training_data.append((tensor, policy_target, value))
        board.push(move)

    return training_data


def extract_training_data_from_position(fen, good_moves):
    """Create training data from a position with known good moves."""
    training_data = []
    board = chess.Board(fen)
    tensor = board_to_tensor(board)

    # Create soft policy target (distribute probability among good moves)
    policy_target = np.zeros(POLICY_OUTPUT_SIZE, dtype=np.float32)
    prob = 1.0 / len(good_moves)

    for move_uci in good_moves:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            move_idx = encode_move(move)
            if move_idx < POLICY_OUTPUT_SIZE:
                policy_target[move_idx] = prob

    # Normalize
    total = policy_target.sum()
    if total > 0:
        policy_target /= total

    # Neutral value for opening positions
    value = 0.0

    training_data.append((tensor, policy_target, value))
    return training_data


def load_all_expert_data():
    """Load all expert training data - uses EXPANDED data if available."""
    all_data = []

    # Choose data source
    if USE_EXPANDED_DATA:
        games_pgn = EXPANDED_GAMES_PGN
        openings = EXPANDED_OPENINGS
        tactics = EXPANDED_TACTICS
        endgames = EXPANDED_ENDGAMES
        middlegames = MIDDLEGAME_POSITIONS  # Use built-in middlegame
        print("Using EXPANDED expert data!", flush=True)
    else:
        games_pgn = EXPERT_GAMES_PGN
        openings = OPENING_POSITIONS
        tactics = TACTICAL_POSITIONS
        endgames = ENDGAME_POSITIONS
        middlegames = MIDDLEGAME_POSITIONS
        print("Using built-in expert data", flush=True)

    # Parse PGN games
    print("Parsing expert games...", flush=True)
    games = parse_pgn_games(games_pgn)
    print(f"Found {len(games)} expert games", flush=True)

    for game in games:
        data = extract_training_data_from_game(game)
        all_data.extend(data)
    print(f"  Positions from games: {len(all_data)}", flush=True)

    # Add opening positions
    print("Adding opening positions...", flush=True)
    opening_count = 0
    for fen, moves in openings:
        data = extract_training_data_from_position(fen, moves)
        all_data.extend(data)
        opening_count += len(data)
    print(f"  Opening positions: {opening_count}", flush=True)

    # Add middlegame positions
    print("Adding middlegame positions...", flush=True)
    middle_count = 0
    for fen, moves in middlegames:
        data = extract_training_data_from_position(fen, moves)
        all_data.extend(data)
        middle_count += len(data)
    print(f"  Middlegame positions: {middle_count}", flush=True)

    # Add tactical positions (5x weighted - CRITICAL!)
    print("Adding tactical positions...", flush=True)
    tactic_count = 0
    for fen, moves in tactics:
        data = extract_training_data_from_position(fen, moves)
        for _ in range(5):  # 5x weight for tactics - must learn these!
            all_data.extend(data)
        tactic_count += len(data) * 5
    print(f"  Tactical positions (5x weighted): {tactic_count}", flush=True)

    # Add endgame positions (5x weighted - CRITICAL for winning games!)
    print("Adding endgame positions...", flush=True)
    endgame_count = 0
    for fen, moves in endgames:
        data = extract_training_data_from_position(fen, moves)
        for _ in range(5):  # 5x weight for endgames - must learn these!
            all_data.extend(data)
        endgame_count += len(data) * 5
    print(f"  Endgame positions (5x weighted): {endgame_count}", flush=True)

    print(f"Total expert positions: {len(all_data)}", flush=True)
    return all_data


def augment_data(data, augment_factor=5):
    """Augment data by replaying positions multiple times."""
    # For chess, we can't really flip/rotate like images
    # But we can repeat important positions
    augmented = list(data)

    # Repeat data
    for _ in range(augment_factor - 1):
        augmented.extend(data)

    random.shuffle(augmented)
    return augmented


# ==================== Training ====================

def train_supervised(model, optimizer, X, policy_targets, value_targets, device, epochs=10):
    """Train model on expert data."""
    model.train()
    n_samples = len(X)
    batch_size = 64  # Smaller batch for small dataset

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        total_policy_loss = 0
        total_value_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]

            x_batch = torch.tensor(X[batch_idx], dtype=torch.float32).to(device)
            policy_batch = torch.tensor(policy_targets[batch_idx], dtype=torch.float32).to(device)
            value_batch = torch.tensor(value_targets[batch_idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            policy_logits, value_pred = model(x_batch)

            # Policy loss (cross-entropy with soft targets)
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(policy_batch * log_probs) / len(batch_idx)

            # Value loss (MSE)
            value_loss = torch.mean((value_pred.squeeze() - value_batch) ** 2)

            # Combined loss
            loss = policy_loss + value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_policy = total_policy_loss / max(n_batches, 1)
        avg_value = total_value_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}: policy_loss={avg_policy:.4f}, value_loss={avg_value:.4f}", flush=True)

    return avg_policy, avg_value


def test_model_moves(model, device):
    """Test if model suggests reasonable moves."""
    model.eval()

    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "After 1.e4"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "After 1.e4 e5 2.Nf3 Nc6"),
    ]

    print("\nTesting model move suggestions:", flush=True)

    with torch.no_grad():
        for fen, desc in test_positions:
            board = chess.Board(fen)
            tensor = board_to_tensor(board)
            x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)
            mask = get_legal_move_mask(board).to(device).unsqueeze(0)

            policy_logits, value = model(x)
            policy_logits = policy_logits.masked_fill(mask == 0, -1e9)
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0)

            # Get top 3 moves
            legal_moves = list(board.legal_moves)
            move_probs = []
            for move in legal_moves:
                idx = encode_move(move)
                if idx < POLICY_OUTPUT_SIZE:
                    move_probs.append((move, policy_probs[idx].item()))

            move_probs.sort(key=lambda x: x[1], reverse=True)
            top_moves = move_probs[:3]

            print(f"  {desc}:", flush=True)
            print(f"    Value: {value.item():.3f}", flush=True)
            print(f"    Top moves: {[(str(m), f'{p:.2%}') for m, p in top_moves]}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("SUPERVISED TRAINING ON EXPERT GAMES", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load or create model
    model_path = "dual_model_mcts.pt"
    model = DualNet(input_dim=781).to(device)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded existing model: {model_path}", flush=True)
        except Exception as e:
            print(f"Could not load model, starting fresh: {e}", flush=True)

    # Test model BEFORE training
    print("\n--- BEFORE SUPERVISED TRAINING ---", flush=True)
    test_model_moves(model, device)

    # Load expert data
    print("\n" + "=" * 60, flush=True)
    expert_data = load_all_expert_data()

    # Augment data (repeat to have more training samples)
    # Higher augmentation = more training = better learning
    augmented_data = augment_data(expert_data, augment_factor=50)
    print(f"Augmented to {len(augmented_data)} samples", flush=True)

    # Prepare tensors
    X = np.array([d[0] for d in augmented_data], dtype=np.float32)
    policy = np.array([d[1] for d in augmented_data], dtype=np.float32)
    values = np.array([d[2] for d in augmented_data], dtype=np.float32)

    print(f"\nTraining data shape: X={X.shape}, policy={policy.shape}, values={values.shape}", flush=True)

    # Train
    print("\n" + "=" * 60, flush=True)
    print("TRAINING ON EXPERT GAMES", flush=True)
    print("=" * 60, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3 rounds is sufficient - loss plateaus after round 1
    NUM_ROUNDS = 3
    for round_num in range(NUM_ROUNDS):
        print(f"\nRound {round_num + 1}/{NUM_ROUNDS}:", flush=True)
        train_supervised(model, optimizer, X, policy, values, device, epochs=15)

        # Save checkpoint after each round (so you can safely stop anytime)
        print(f"  Saving checkpoint after round {round_num + 1}...", flush=True)
        torch.save({
            "input_dim": 781,
            "state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "training": f"supervised_round_{round_num + 1}"
        }, model_path)

        # Shuffle data between rounds
        indices = np.random.permutation(len(X))
        X = X[indices]
        policy = policy[indices]
        values = values[indices]

    # Test model AFTER training
    print("\n--- AFTER SUPERVISED TRAINING ---", flush=True)
    test_model_moves(model, device)

    # Save model
    print("\n" + "=" * 60, flush=True)
    torch.save({
        "input_dim": 781,
        "state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
        "training": "supervised_expert"
    }, model_path)
    print(f"Model saved to {model_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUPERVISED TRAINING COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print("\nNext steps:", flush=True)
    print("1. Test the model: .venv\\Scripts\\python.exe main.py --engine mcts --load_model dual_model_mcts.pt --sims 100", flush=True)
    print("2. Continue with self-play: .venv\\Scripts\\python.exe train_clean.py --hours 4", flush=True)


if __name__ == "__main__":
    main()

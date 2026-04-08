# Chess AI Project

An AI chess engine with neural network evaluation and interactive GUI gameplay.

## Features

- Interactive chess GUI using Pygame
- Multiple AI engines:
  - Basic Minimax with alpha-beta pruning
  - Neural network-based position evaluation
- Self-play training pipeline for model improvement
- Arena mode for comparing different AI engines

## Project Structure

```
├── main.py              # Interactive GUI - Human vs AI gameplay
├── ai.py                # Basic Minimax engine with material evaluation
├── ai_nn.py             # Neural network-enhanced Minimax engine
├── features.py          # Chess board to tensor feature extraction
├── selfplay.py          # Self-play game generator for training data
├── train_value.py       # Neural network training script
├── arena.py             # AI vs AI tournament evaluation
├── data/
│   ├── classes/         # Custom chess implementation (Board, Pieces, etc.)
│   ├── imgs/            # Chess piece images for GUI
│   ├── *.npz            # Training datasets
│   ├── *.pgn            # Game records
│   └── value_model.pt   # Trained neural network model
└── requirements.txt     # Python dependencies
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Play Chess with GUI
```bash
python main.py
```
Click on a piece to select it, then click on a valid square to move.

### Generate Self-Play Games
```bash
python selfplay.py --engine minimax --games 100 --output selfplay.pgn
```

### Train the Neural Network
```bash
python train_value.py --dataset dataset.npz --epochs 50
```

### Run AI vs AI Tournaments
```bash
python arena.py --engine1 minimax --engine2 nn --games 20
```

## AI Engines

- **minimax**: Basic negamax with alpha-beta pruning and material evaluation
- **nn**: Neural network position evaluation (requires trained model)

## Neural Network Architecture

The value network uses a simple feedforward architecture:
- Input: 781 features (12 piece bitboards + auxiliary features)
- Hidden: 512 → 256 neurons with ReLU activation
- Output: 1 value (position evaluation, tanh-normalized)

## Training Pipeline

1. Generate games via self-play: `selfplay.py`
2. Train value network: `train_value.py`
3. Evaluate performance: `arena.py`
4. Iterate with improved model

## Requirements

- Python 3.10+
- PyTorch
- python-chess
- pygame
- numpy

## Notes

- The project uses both a custom chess implementation (data/classes/) and python-chess library
- Trained models are saved as `value_model.pt`
- Game records are stored in PGN format
- Training datasets use compressed NumPy format (.npz)

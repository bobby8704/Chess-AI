#!/usr/bin/env python3
"""
Improved Training Script for Chess AI

Features:
- Position augmentation via horizontal mirroring (2x data)
- Train/validation split for monitoring overfitting
- Learning rate scheduling
- Early stopping
- Model checkpointing (save best model)
- Training history logging
- Integration with game database for training tracking

Usage:
    python train_improved.py --dataset data/datasets/dataset.npz --epochs 50

    # With augmentation (recommended)
    python train_improved.py --dataset data/datasets/dataset.npz --augment --epochs 50

    # With early stopping
    python train_improved.py --dataset data/datasets/dataset.npz --early_stopping --patience 5
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ai_nn import ValueNet
from training_utils import (
    augment_dataset,
    create_validation_split,
    compute_dataset_statistics
)
from game_recorder import GameRecorder, ModelRecord


class TrainingHistory:
    """Track and save training history."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def update(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True  # New best
        return False

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def train(
    dataset_path: str,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    out_path: str = "value_model.pt",
    augment: bool = True,
    val_fraction: float = 0.1,
    early_stopping: bool = False,
    patience: int = 10,
    lr_scheduler: bool = True,
    save_history: bool = True,
    db_path: str = "data/games.db",
    parent_model_id: int = None,
    verbose: bool = True
) -> dict:
    """
    Train the value network with improved features.

    Args:
        dataset_path: Path to .npz dataset
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Initial learning rate
        out_path: Path to save trained model
        augment: Whether to augment data with mirrored positions
        val_fraction: Fraction of data for validation
        early_stopping: Whether to stop early if validation loss plateaus
        patience: Epochs to wait before early stopping
        lr_scheduler: Whether to reduce LR on plateau
        save_history: Whether to save training history
        db_path: Path to game database for logging
        parent_model_id: ID of parent model (for lineage tracking)
        verbose: Whether to print progress

    Returns:
        Dictionary with training results
    """
    # Load data
    if verbose:
        print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    # Compute statistics before augmentation
    stats_before = compute_dataset_statistics(X, y)
    if verbose:
        print(f"  Original: {stats_before['num_positions']} positions")
        print(f"  Target distribution: {stats_before['white_wins_pct']:.1f}% W, "
              f"{stats_before['black_wins_pct']:.1f}% B, {stats_before['draws_pct']:.1f}% D")

    # Augment dataset
    if augment:
        if verbose:
            print("Augmenting dataset with mirrored positions...")
        X, y = augment_dataset(X, y)
        if verbose:
            print(f"  Augmented: {len(X)} positions (2x)")

    # Train/validation split
    if verbose:
        print(f"Splitting data ({int((1-val_fraction)*100)}% train, {int(val_fraction*100)}% val)...")
    X_train, y_train, X_val, y_val = create_validation_split(X, y, val_fraction)
    if verbose:
        print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on: {device}")

    # Create model
    input_dim = X_train.shape[1]
    model = ValueNet(input_dim).to(device)
    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")

    # Create data loaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.reshape(-1, 1))
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val.reshape(-1, 1))
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = None
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = TrainingHistory()
    best_model_state = None
    epochs_without_improvement = 0

    if verbose:
        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 60)

    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update history and check for improvement
        is_best = history.update(epoch, train_loss, val_loss, current_lr)

        if is_best:
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)

        # Print progress
        if verbose:
            best_marker = " *" if is_best else ""
            print(f"Epoch {epoch:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"LR: {current_lr:.2e}{best_marker}")

        # Early stopping
        if early_stopping and epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model from epoch {history.best_epoch}")

    # Save model
    torch.save({
        "input_dim": input_dim,
        "state_dict": model.state_dict(),
        "training_info": {
            "epochs_trained": epoch,
            "final_train_loss": history.train_losses[-1],
            "final_val_loss": history.val_losses[-1],
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch,
            "augmented": augment,
            "dataset": dataset_path,
            "timestamp": datetime.now().isoformat()
        }
    }, out_path)

    if verbose:
        print(f"\nSaved model to {out_path}")

    # Save training history
    if save_history:
        history_path = out_path.replace('.pt', '_history.json')
        history.save(history_path)
        if verbose:
            print(f"Saved training history to {history_path}")

    # Register model in database
    try:
        recorder = GameRecorder(db_path)
        model_id = recorder.register_model(ModelRecord(
            name="ValueNet",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            file_path=out_path,
            parent_model_id=parent_model_id,
            architecture=f"ValueNet_{input_dim}_512_256_1",
            training_games=stats_before['num_positions'] // 50,  # Estimate
            training_positions=stats_before['num_positions'],
            training_epochs=epoch,
            training_loss=history.train_losses[-1],
            validation_loss=history.best_val_loss,
            notes=f"Trained on {dataset_path}, augment={augment}"
        ))
        if verbose:
            print(f"Registered model in database with ID: {model_id}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not register model in database: {e}")
        model_id = None

    # Return results
    return {
        "model_path": out_path,
        "model_id": model_id,
        "epochs_trained": epoch,
        "best_epoch": history.best_epoch,
        "final_train_loss": history.train_losses[-1],
        "final_val_loss": history.val_losses[-1],
        "best_val_loss": history.best_val_loss,
        "train_positions": len(X_train),
        "val_positions": len(X_val),
        "augmented": augment
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train chess value network with improved features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_improved.py --dataset data/datasets/dataset.npz

    # Full training with all features
    python train_improved.py --dataset data/datasets/dataset.npz \\
        --epochs 100 --augment --early_stopping --patience 10

    # Quick training (no augmentation)
    python train_improved.py --dataset data/datasets/dataset.npz \\
        --epochs 20 --no-augment
        """
    )

    parser.add_argument("--dataset", default="data/datasets/dataset.npz",
                        help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--out", default="value_model.pt",
                        help="Output model path")

    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Augment with mirrored positions (default: True)")
    parser.add_argument("--no-augment", action="store_false", dest="augment",
                        help="Disable augmentation")

    # Validation and early stopping
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction for validation (default: 0.1)")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")

    # LR scheduling
    parser.add_argument("--lr_scheduler", action="store_true", default=True,
                        help="Use learning rate scheduler (default: True)")
    parser.add_argument("--no-lr_scheduler", action="store_false", dest="lr_scheduler",
                        help="Disable LR scheduler")

    # Database
    parser.add_argument("--db_path", default="data/games.db",
                        help="Path to game database")
    parser.add_argument("--parent_model_id", type=int, default=None,
                        help="Parent model ID for lineage tracking")

    args = parser.parse_args()

    # Run training
    results = train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        augment=args.augment,
        val_fraction=args.val_fraction,
        early_stopping=args.early_stopping,
        patience=args.patience,
        lr_scheduler=args.lr_scheduler,
        db_path=args.db_path,
        parent_model_id=args.parent_model_id,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {results['model_path']}")
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation loss: {results['best_val_loss']:.5f}")
    print(f"Training positions: {results['train_positions']:,}")
    if results['model_id']:
        print(f"Model ID in database: {results['model_id']}")


if __name__ == "__main__":
    main()

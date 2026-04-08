#!/usr/bin/env python3
"""
Training Script for Dual Network (Policy + Value)

This script trains the DualNet which outputs both:
- Policy: Probability distribution over moves
- Value: Position evaluation

The training uses:
- Policy loss: Cross-entropy with MCTS visit counts as targets
- Value loss: MSE with game outcomes as targets
- Combined loss: policy_loss + value_weight * value_loss

Usage:
    # Train from self-play data
    python train_dual.py --dataset data/datasets/dataset.npz --epochs 50

    # Train with policy targets from MCTS (when available)
    python train_dual.py --dataset data/datasets/mcts_dataset.npz --epochs 100
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neural_network import DualNet, DualNetResidual, POLICY_OUTPUT_SIZE
from training_utils import augment_dataset, create_validation_split
from game_recorder import GameRecorder, ModelRecord


class PolicyValueDataset(torch.utils.data.Dataset):
    """Dataset for training DualNet with policy and value targets."""

    def __init__(self, X, policy_targets, value_targets):
        """
        Args:
            X: Board features (N, 781)
            policy_targets: Move probability targets (N, POLICY_OUTPUT_SIZE)
            value_targets: Game outcome targets (N,)
        """
        self.X = torch.from_numpy(X).float()
        self.policy = torch.from_numpy(policy_targets).float()
        self.value = torch.from_numpy(value_targets).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.policy[idx], self.value[idx]


def create_uniform_policy_targets(X, y, legal_masks=None):
    """
    Create uniform policy targets for positions without MCTS data.

    When we don't have MCTS visit counts, we use uniform distribution
    over legal moves as a weak supervision signal.

    For now, this creates a placeholder that will be masked during training.
    """
    n = len(X)
    policy_targets = np.zeros((n, POLICY_OUTPUT_SIZE), dtype=np.float32)

    # Without legal move information, just use uniform small values
    # The actual policy loss will be masked or use a different approach
    policy_targets[:] = 1.0 / POLICY_OUTPUT_SIZE

    return policy_targets


def train_dual(
    dataset_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    out_path: str = "dual_model.pt",
    use_residual: bool = False,
    num_residual_blocks: int = 4,
    hidden_dim: int = 512,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    val_fraction: float = 0.1,
    augment: bool = True,
    early_stopping: bool = False,
    patience: int = 10,
    db_path: str = "data/games.db",
    parent_model_id: int = None,
    verbose: bool = True
):
    """
    Train the DualNet (policy + value network).

    Args:
        dataset_path: Path to .npz dataset with X (features) and y (outcomes)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        out_path: Output model path
        use_residual: Whether to use DualNetResidual
        num_residual_blocks: Number of residual blocks (if use_residual)
        hidden_dim: Hidden layer dimension
        value_weight: Weight for value loss
        policy_weight: Weight for policy loss
        val_fraction: Validation fraction
        augment: Whether to augment data
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        db_path: Game database path
        parent_model_id: Parent model ID for lineage
        verbose: Print progress

    Returns:
        Training results dictionary
    """
    # Load data
    if verbose:
        print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    # Check for policy targets
    has_policy = "policy" in data
    if has_policy:
        policy_targets = data["policy"].astype(np.float32)
        if verbose:
            print("  Found policy targets in dataset")
    else:
        # For large datasets, don't create policy targets (saves memory)
        # Policy loss will be skipped during training
        if len(X) > 500000:
            policy_targets = None
            if verbose:
                print("  Large dataset - skipping policy targets to save memory")
                print("  (Training will focus on value learning only)")
        else:
            # Create placeholder policy targets for smaller datasets
            policy_targets = create_uniform_policy_targets(X, y)
            if verbose:
                print("  No policy targets found, using uniform distribution")
                print("  (Policy loss will be minimal - focus on value learning)")

    if verbose:
        print(f"  Loaded {len(X)} positions")

    # Augment dataset
    if augment:
        if verbose:
            print("Augmenting dataset...")
        X, y = augment_dataset(X, y)
        # Also augment policy targets if they exist
        if policy_targets is not None:
            policy_targets = np.concatenate([policy_targets, policy_targets], axis=0)
        if verbose:
            print(f"  Augmented to {len(X)} positions")

    # Train/val split
    n = len(X)
    indices = np.random.permutation(n)
    val_size = int(n * val_fraction)

    X_val = X[indices[:val_size]]
    y_val = y[indices[:val_size]]

    X_train = X[indices[val_size:]]
    y_train = y[indices[val_size:]]

    if policy_targets is not None:
        policy_val = policy_targets[indices[:val_size]]
        policy_train = policy_targets[indices[val_size:]]
    else:
        policy_val = None
        policy_train = None

    if verbose:
        print(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on: {device}")

    # Create model
    input_dim = X_train.shape[1]
    if use_residual:
        model = DualNetResidual(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks
        ).to(device)
        model_type = "DualNetResidual"
    else:
        model = DualNet(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        model_type = "DualNet"

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_type} with {param_count:,} parameters")

    # Create data loaders
    if policy_train is not None:
        train_ds = PolicyValueDataset(X_train, policy_train, y_train)
        val_ds = PolicyValueDataset(X_val, policy_val, y_val)
        use_policy_loss = has_policy  # Only use policy loss if we had real targets
    else:
        # Value-only training for large datasets
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float().unsqueeze(1)
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float().unsqueeze(1)
        )
        use_policy_loss = False

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    value_loss_fn = nn.MSELoss()

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_value_loss": [], "val_value_loss": [],
        "train_policy_loss": [], "val_policy_loss": [],
        "lr": []
    }
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0

    if verbose:
        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 70)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_value_loss = 0.0
        train_policy_loss = 0.0
        train_total_loss = 0.0

        for batch in train_dl:
            if use_policy_loss:
                xb, pb, vb = batch
                pb = pb.to(device)
            else:
                xb, vb = batch
                pb = None

            xb = xb.to(device)
            vb = vb.to(device)

            optimizer.zero_grad()

            policy_logits, value = model(xb)

            # Value loss
            v_loss = value_loss_fn(value, vb)

            # Policy loss (cross-entropy with soft targets)
            # Only compute if we have meaningful policy targets
            if use_policy_loss and pb is not None:
                p_loss = -(pb * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
            else:
                # No policy loss when no targets
                p_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = value_weight * v_loss + policy_weight * p_loss

            loss.backward()
            optimizer.step()

            train_value_loss += v_loss.item() * xb.size(0)
            train_policy_loss += p_loss.item() * xb.size(0)
            train_total_loss += loss.item() * xb.size(0)

        train_value_loss /= len(train_ds)
        train_policy_loss /= len(train_ds)
        train_total_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_value_loss = 0.0
        val_policy_loss = 0.0
        val_total_loss = 0.0

        with torch.no_grad():
            for batch in val_dl:
                if use_policy_loss:
                    xb, pb, vb = batch
                    pb = pb.to(device)
                else:
                    xb, vb = batch
                    pb = None

                xb = xb.to(device)
                vb = vb.to(device)

                policy_logits, value = model(xb)

                v_loss = value_loss_fn(value, vb)
                if use_policy_loss and pb is not None:
                    p_loss = -(pb * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
                else:
                    p_loss = torch.tensor(0.0, device=device)

                loss = value_weight * v_loss + policy_weight * p_loss

                val_value_loss += v_loss.item() * xb.size(0)
                val_policy_loss += p_loss.item() * xb.size(0)
                val_total_loss += loss.item() * xb.size(0)

        val_value_loss /= len(val_ds)
        val_policy_loss /= len(val_ds)
        val_total_loss /= len(val_ds)

        # Update scheduler
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history["train_loss"].append(train_total_loss)
        history["val_loss"].append(val_total_loss)
        history["train_value_loss"].append(train_value_loss)
        history["val_value_loss"].append(val_value_loss)
        history["train_policy_loss"].append(train_policy_loss)
        history["val_policy_loss"].append(val_policy_loss)
        history["lr"].append(current_lr)

        # Check for improvement
        is_best = val_total_loss < best_val_loss
        if is_best:
            best_val_loss = val_total_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        if verbose:
            marker = " *" if is_best else ""
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_total_loss:.5f} (V:{train_value_loss:.5f} P:{train_policy_loss:.5f}) | "
                  f"Val: {val_total_loss:.5f}{marker}")

        # Early stopping
        if early_stopping and epochs_no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model from epoch {best_epoch}")

    # Save model
    save_dict = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "training_info": {
            "epochs_trained": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "dataset": dataset_path,
            "timestamp": datetime.now().isoformat()
        }
    }

    if use_residual:
        save_dict["num_residual_blocks"] = num_residual_blocks

    torch.save(save_dict, out_path)

    if verbose:
        print(f"\nSaved model to {out_path}")

    # Save history
    history_path = out_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    if verbose:
        print(f"Saved history to {history_path}")

    # Register in database
    try:
        recorder = GameRecorder(db_path)
        model_id = recorder.register_model(ModelRecord(
            name=model_type,
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            file_path=out_path,
            parent_model_id=parent_model_id,
            architecture=f"{model_type}_{input_dim}_{hidden_dim}",
            training_epochs=epoch,
            training_loss=history["train_loss"][-1],
            validation_loss=best_val_loss,
            notes=f"Trained on {dataset_path}"
        ))
        if verbose:
            print(f"Registered model with ID: {model_id}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not register model: {e}")
        model_id = None

    return {
        "model_path": out_path,
        "model_id": model_id,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train DualNet (policy + value network)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dataset", default="data/datasets/dataset.npz",
                        help="Path to training dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="dual_model.pt")

    # Architecture
    parser.add_argument("--residual", action="store_true",
                        help="Use residual network architecture")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="Number of residual blocks")
    parser.add_argument("--hidden_dim", type=int, default=512)

    # Loss weights
    parser.add_argument("--value_weight", type=float, default=1.0)
    parser.add_argument("--policy_weight", type=float, default=1.0)

    # Training options
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    # Database
    parser.add_argument("--db_path", default="data/games.db")
    parser.add_argument("--parent_model_id", type=int, default=None)

    args = parser.parse_args()

    results = train_dual(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        use_residual=args.residual,
        num_residual_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        value_weight=args.value_weight,
        policy_weight=args.policy_weight,
        val_fraction=args.val_fraction,
        augment=args.augment,
        early_stopping=args.early_stopping,
        patience=args.patience,
        db_path=args.db_path,
        parent_model_id=args.parent_model_id,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {results['model_path']}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation loss: {results['best_val_loss']:.5f}")


if __name__ == "__main__":
    main()

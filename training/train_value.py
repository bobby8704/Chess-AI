
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ai_nn import ValueNet

def train(dataset_path: str = "data/datasets/dataset.npz", epochs: int = 5, batch_size: int = 512,
          lr: float = 1e-3, out_path: str = "value_model.pt"):
    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32).reshape(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(X.shape[1]).to(device)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        print(f"Epoch {epoch:02d} | loss = {running / len(ds):.5f}")

    torch.save({"input_dim": X.shape[1], "state_dict": model.state_dict()}, out_path)
    print(f"Saved value model to {out_path}")
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/datasets/dataset.npz")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="value_model.pt")
    args = ap.parse_args()
    train(args.dataset, args.epochs, args.batch_size, args.lr, args.out)

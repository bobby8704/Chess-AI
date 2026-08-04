"""
Export a trained dual-network checkpoint to ONNX for CPU inference.

Serving runs inference through onnxruntime rather than torch: it is measurably faster
at batch size 1 (which is what MCTS does), and it lets a deployment ship without the
~430 MB torch dependency at all.

Only forward() is exported — policy logits and value. Legal-move masking and softmax
are applied by the caller over just the legal indices, which is cheaper than, and
mathematically identical to, masking the full 4288-wide logit vector before softmax.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint models/dual_model_mcts.pt --out models/dual_model_mcts_fp32.onnx
"""

import argparse
import os

import numpy as np
import torch

from neural_network import load_dual_model, is_cnn_model

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(ROOT, "models", "dual_model_mcts.pt")
DEFAULT_OUT = os.path.join(ROOT, "models", "dual_model_mcts_fp32.onnx")


def export(checkpoint: str = DEFAULT_CKPT, out: str = DEFAULT_OUT, opset: int = 17):
    model = load_dual_model(checkpoint)
    model.eval()

    if is_cnn_model(model):
        dummy = torch.zeros(1, model.in_channels if hasattr(model, "in_channels") else 13, 8, 8)
    else:
        dummy = torch.zeros(1, 781)

    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # The dynamo exporter writes weights to a sidecar "<name>.onnx.data" file. That
    # works only as long as both files travel together, which is a deployment trap.
    # Re-save everything into a single self-contained .onnx.
    import onnx
    sidecar = out + ".data"
    if os.path.exists(sidecar):
        model_proto = onnx.load(out)  # resolves the sidecar
        onnx.save_model(model_proto, out, save_as_external_data=False)
        os.remove(sidecar)

    print(f"exported {checkpoint} -> {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")

    # Verify against torch on random boards before anyone trusts this file.
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst_logit = worst_value = 0.0
    for _ in range(32):
        x = rng.standard_normal(tuple(dummy.shape)).astype(np.float32)
        with torch.no_grad():
            t_logits, t_value = model(torch.from_numpy(x))
        o_logits, o_value = sess.run(None, {"board": x})
        worst_logit = max(worst_logit, float(np.abs(t_logits.numpy() - o_logits).max()))
        worst_value = max(worst_value, float(np.abs(t_value.numpy() - o_value).max()))
    print(f"torch vs onnx over 32 random inputs: "
          f"max |d logit| = {worst_logit:.3e}, max |d value| = {worst_value:.3e}")

    # Confirm the batch axis is genuinely dynamic, so batched search stays possible later.
    batch_shape = (4,) + tuple(dummy.shape[1:])
    b_logits, _ = sess.run(None, {"board": np.zeros(batch_shape, dtype=np.float32)})
    print(f"dynamic batch check: input {batch_shape} -> policy_logits {b_logits.shape}")
    if worst_logit > 1e-4 or worst_value > 1e-4:
        raise SystemExit("ONNX export diverges from torch by more than 1e-4 — not writing this off as noise")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    export(a.checkpoint, a.out, a.opset)

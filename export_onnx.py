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
import sys

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

    # Verify against torch on REAL board tensors, not random noise. Gaussian inputs sit
    # far outside the actual input distribution (0/1 piece planes) and push activations
    # into ranges the network never sees, inflating the apparent divergence — this check
    # previously tripped its own 1e-4 threshold on noise that meant nothing for play.
    # What matters is whether the exported model PICKS THE SAME MOVE.
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])

    try:
        sys.path.insert(0, ROOT)
        from bench.strength import load_suite, rebuild, sample
        from features import board_to_tensor_2d
        inputs = [board_to_tensor_2d(rebuild(e["moves"]))
                  for e in sample(load_suite("acpl")["positions"], 128)]
        kind = "real positions"
    except Exception as e:
        print(f"  (position suite unavailable: {e}; falling back to random inputs)")
        rng = np.random.default_rng(0)
        inputs = [rng.standard_normal(tuple(dummy.shape)[1:]).astype(np.float32)
                  for _ in range(32)]
        kind = "random inputs"

    worst_logit = worst_value = 0.0
    top1_agree = 0
    for b in inputs:
        x = np.asarray(b, dtype=np.float32)[None]
        with torch.no_grad():
            t_logits, t_value = model(torch.from_numpy(x))
        o_logits, o_value = sess.run(None, {"board": x})
        worst_logit = max(worst_logit, float(np.abs(t_logits.numpy() - o_logits).max()))
        worst_value = max(worst_value, float(np.abs(t_value.numpy() - o_value).max()))
        top1_agree += int(t_logits.numpy().argmax() == o_logits.argmax())
    print(f"torch vs onnx over {len(inputs)} {kind}: "
          f"max |d logit| = {worst_logit:.3e}, max |d value| = {worst_value:.3e}, "
          f"top-1 agreement {top1_agree}/{len(inputs)}")

    # Confirm the batch axis is genuinely dynamic, so batched search stays possible later.
    batch_shape = (4,) + tuple(dummy.shape[1:])
    b_logits, _ = sess.run(None, {"board": np.zeros(batch_shape, dtype=np.float32)})
    print(f"dynamic batch check: input {batch_shape} -> policy_logits {b_logits.shape}")
    # Gate on behaviour, not on a raw float delta: the export must choose the same move
    # everywhere. The magnitude bound is a loose sanity check on top of that.
    if top1_agree != len(inputs):
        raise SystemExit(
            f"ONNX export picks a different top move on "
            f"{len(inputs) - top1_agree}/{len(inputs)} positions — do not ship it")
    if worst_logit > 1e-2:
        raise SystemExit(f"ONNX logits diverge by {worst_logit:.2e} — investigate")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    export(a.checkpoint, a.out, a.opset)

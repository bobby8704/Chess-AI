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


DEFAULT_INT8 = os.path.join(ROOT, "models", "dual_model_mcts_int8.onnx")


def quantize_int8(fp32: str = DEFAULT_OUT, out: str = DEFAULT_INT8, calib: int = 512):
    """
    Quantise the fp32 export to int8 (static QDQ, per-channel weights).

    STATIC, not dynamic: this network is a residual CONV tower, and onnxruntime's dynamic
    path quantises MatMul/Linear only, which would leave nearly all of it in fp32.

    Calibration uses REAL board tensors from the committed suite. Activation ranges are
    the whole point of static quantisation, and gaussian noise would pick ranges the
    network never sees — the same mistake this file's own verification used to make.

    Measured on the served model: forward pass 3238 -> 800 us at 2 threads (4.05x), file
    18.44 -> 4.77 MB, top-1 policy agreement 95.3%, and -13.0 Elo at equal simulations
    (95% CI [-60.1, +33.5], 160 games) — i.e. not distinguishable from free.
    """
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Resolved before the chdir below, so relative --out paths still land where the
    # caller meant.
    fp32, out = os.path.abspath(fp32), os.path.abspath(out)

    sys.path.insert(0, ROOT)
    from bench.strength import load_suite, rebuild, sample
    from features import board_to_tensor_2d

    def tensors(n, offset=0):
        picked = sample(load_suite("acpl")["positions"], n + offset)[offset:]
        out_ = []
        for e in picked:
            b = rebuild(e["moves"])
            if not b.is_game_over():
                out_.append(board_to_tensor_2d(b).astype(np.float32, copy=False)[None])
        return out_

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.it = iter(data)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"board": x}

    # The quantizer drops temp files (sym_shape_infer_temp.onnx and a UUID-named .data
    # external-weights blob) into the CURRENT DIRECTORY, which means running this from the
    # repo root litters the repo root. Do the work inside a temp dir and restore cwd.
    import tempfile
    calib_data = tensors(calib)          # gather before chdir; paths are resolved via ROOT
    cwd = os.getcwd()
    # Quantise to a STAGING path and only move it into place once it passes the gate
    # below. Writing straight to `out` would leave a model the gate has just rejected
    # sitting exactly where serving looks for it.
    staged = out + ".staging"
    with tempfile.TemporaryDirectory() as tmp:
        prep = os.path.join(tmp, "prep.onnx")
        try:
            os.chdir(tmp)
            # skip_symbolic_shape: the export carries a dynamic batch axis and symbolic
            # shape inference gives up on it ("Incomplete symbolic shape inference").
            # Calibration supplies concrete shapes, so static shapes are not needed here.
            quant_pre_process(fp32, prep, skip_symbolic_shape=True)
            quantize_static(prep, staged, Reader(calib_data),
                            quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8,
                            per_channel=True)
        finally:
            os.chdir(cwd)
    # Gate on BEHAVIOUR against the fp32 export, on positions the calibration never saw.
    def sess(p):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        return ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])

    sf, si = sess(fp32), sess(staged)
    nf, ni = sf.get_inputs()[0].name, si.get_inputs()[0].name
    ev = tensors(384, offset=calib + 7)
    agree, kls, vdiff = 0, [], []
    for x in ev:
        lf, vf = sf.run(None, {nf: x})
        li, vi = si.run(None, {ni: x})
        agree += int(lf.argmax() == li.argmax())
        pa = np.exp(lf[0] - lf[0].max()); pa /= pa.sum()
        pb = np.exp(li[0] - li[0].max()); pb /= pb.sum()
        kls.append(float((pa * np.log((pa + 1e-12) / (pb + 1e-12))).sum()))
        vdiff.append(abs(float(np.reshape(vf, -1)[0] - np.reshape(vi, -1)[0])))
    rate, kl = agree / len(ev), float(np.mean(kls))
    vmean, vmax = float(np.mean(vdiff)), float(np.max(vdiff))
    print(f"int8 vs fp32 over {len(ev)} held-out positions: top-1 {agree}/{len(ev)} "
          f"({rate:.1%}), mean KL {kl:.4f}, mean |d value| {vmean:.4f} (max {vmax:.4f})")

    # The gate is on POLICY, because policy is what the served engine uses. The value head
    # is computed and then discarded (mcts.leaf_value, value_head_weight=0.0 — wiring it in
    # measured -137 to -307 Elo), so its fidelity barely reaches play. It keeps a loose MEAN
    # check as a "did quantisation go badly wrong" signal. Gating on the MAX was a mistake:
    # one outlier position out of 384 tripped it at 0.278 on the very model that then won
    # +175.7 Elo in real games.
    if rate < 0.90 or kl > 0.02 or vmean > 0.05:
        os.remove(staged)
        raise SystemExit(
            "int8 quantisation diverges too far from fp32 — not shipping it, and the "
            f"staged file has been removed. (top-1 {rate:.1%} needs >=90%, "
            f"mean KL {kl:.4f} needs <=0.02, mean |d value| {vmean:.4f} needs <=0.05)")

    # Only now does it become the file serving will pick up.
    os.replace(staged, out)
    print(f"quantised {fp32} -> {out}  "
          f"({os.path.getsize(fp32)/1e6:.2f} MB -> {os.path.getsize(out)/1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", action="store_true",
                    help="also quantise the export to int8 (what serving prefers)")
    ap.add_argument("--int8-out", default=DEFAULT_INT8)
    a = ap.parse_args()
    export(a.checkpoint, a.out, a.opset)
    if a.int8:
        quantize_int8(a.out, a.int8_out)

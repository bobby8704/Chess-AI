# Native kernel port

Goal: make simulations cheaper and spend the speed on more of them — the only
strategy this project has ever measured to work (sims curve: 127/154/172 Elo per
doubling through 2600 sims, not flattening; `bench/results/elo_current*_models.json`).

## The measured map (bench/profile_move.py, 1300 sims, 2026-08-08)

    evaluate_fast (leaves)            93.1%
      onnx_run          36.3%     1145 us/call
      quiescence        48.7%     1533 us/call
        _evaluate_raw     28.4%    116 us x ~80k calls/move
        can_claim_draw     8.5%    268 us x 1/leaf (root probe)
        movegen/order/push ~12%
      tensorize+legal_idx  4.0%
      glue                 4.1%
    evaluate_full (root)   0.5%
    tree walk + vetoes     6.4%

Two derived estimates preceded this profile and both were wrong in the details
(forward "~30%" and "40-50%"); the tree walk everyone assumed mattered is 6%.
Nothing below is sized from arithmetic — only from this table.

## Stages

**Stage 1 — native quiescence (this stage). Ceiling: ~1.9x ≈ +160 within-family Elo.**
C++17 + pybind11 (MSVC 14.39 on the dev box; any C++17 compiler on CI/Linux).
Surface to port, exactly and only what `evaluate_quiescence` touches:
  - bitboard position (12 piece bitboards + turn/castling/ep/halfmove),
    unpacked per call from python-chess internals (ints, no FEN strings);
  - captures+promotions movegen, MVV-LVA order, 200cp delta pruning, depth 2;
  - mate/stalemate detection (any-legal-move existence);
  - `_evaluate_raw`: material + PST + pawn structure + king safety
    (phase-weighted) + back rank + lone-king + bishop pair;
  - root-only draw claim: halfmove>=100 or threefold within the (<=8-ply
    truncated) stack — history passed in as packed positions.

**Gates, in order:**
  1. Differential harness: native vs `evaluation.evaluate_quiescence` on every
     position of the committed 1474 suite plus randomised capture-rich
     positions — target EXACT agreement (compute cp in the same order/types).
  2. `bench/verify.py` 0/26 + 0/26 (move-preserving if exact agreement holds).
  3. If ulp drift proves unavoidable: fall back to mate/walkinto suites +
     `elo.py` h2h null vs current at equal sims, then equal time.
  4. Adoption: `elo.py` h2h at equal WALL CLOCK (the speed spent on sims),
     which is where the Elo should appear.

**Stage 2 — forward-pass share (36.3%).** Not a C++ problem. Candidates, to be
measured not assumed: batched leaf evaluation with virtual loss (move-changing,
needs games), a smaller distilled net, or onnxruntime API-level wins. Decide
after stage 1 lands — quiescence going native raises the forward's share to
~70% and changes the arithmetic.

**Explicitly NOT being ported:** the tree walk (6.4% — weeks of risk for a
rounding error), the root pipeline (0.5%, and it is the measured safety layer),
the vetoes. Python stays the orchestrator; C++ is a leaf-evaluation service.

## Layout

    native/
      PORT.md            this file
      chesskernel.cpp    the extension (single translation unit)
      setup_native.py    build: .venv/Scripts/python.exe native/setup_native.py build_ext --inplace
      test_native.py     differential harness vs python-chess / evaluation.py

The extension is OPTIONAL at runtime: `evaluation.py` will prefer it when
importable and fall back to pure Python otherwise, so serving deploys (Render)
keep working with zero native toolchain until a wheel is built for them.

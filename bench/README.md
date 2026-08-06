# bench/

Measurement tooling for the chess engine. Three questions, three tools.

| Question | Tool |
|---|---|
| Did this change the moves the engine plays? | `verify.py` |
| Is this move-quality change worth pursuing? | `strength.py` (ACPL — a cheap screen) |
| **Is the engine actually stronger?** | **`elo.py` (real games — the deciding answer)** |
| Can the app still be deployed? | `test_deploy.py` |

## Read this before trusting a centipawn number

**ACPL massively understates game strength, and the relationship is not linear.** The
same change measured **−10.33 cp** of ACPL and **+209 Elo** head-to-head — it wins three
games in four. In a game, small per-move edges compound: a deeper search wins material
repeatedly and each edge feeds the next.

Two consequences, both learned the hard way:

- **Never justify removing something with an ACPL null.** A 95% CI of ±8 cp sounds tight
  and is worth roughly ±150 Elo. Acting on exactly that interval cut the `hard` preset
  from 1300 to 600 sims and cost a measured **220 Elo**.
- **Treat every "N cp" figure as a lower bound on game impact**, never as a measure of it.

ACPL is still worth having: it is ~100× cheaper than playing games, deterministic, and
fine for screening whether a change is worth a real match. It just cannot settle one.

## verify.py — move equivalence and latency

Runs the engine over a small fixed position set and records the chosen move, the
**full root visit distribution**, and wall-clock time. Comparing two runs proves a
change is move-preserving — not "looks the same", but identical visit counts to 12
decimal places.

```bash
python bench/verify.py run before --sims 200 600
# ...make a change...
python bench/verify.py run after --sims 200 600
python bench/verify.py compare before after
```

To compare across commits, check the old revision out into a git worktree and run it
there — never by stashing, which leaves the tree in the wrong state if a run is
interrupted.

Use this for any change that is supposed to be a pure speedup. If it reports a
difference, the change is not a pure speedup, whatever the intent was.

## strength.py — objective playing strength

Average centipawn loss against Stockfish at fixed depth, measured **paired** on a
frozen suite so that between-position variance cancels.

```bash
python bench/strength.py build --games 750 --per-game 2   # once; suite is committed
python bench/strength.py run baseline-600 --sims 600
python bench/strength.py run candidate-600 --sims 600
python bench/strength.py compare baseline-600 candidate-600
python bench/strength.py mate baseline --sims 200
```

Three things make the numbers trustworthy, and all three are easy to lose:

- **Stockfish gets `ucinewgame` before every analysis.** Without it the transposition
  table persists and one position scores 48/47/51 cp on repeated calls.
- **Positions come from shallow-Stockfish games with randomised top-k**, not random
  legal moves, so they are positions a real game could reach.
- **`bench/suites/` is committed and must not be rebuilt casually.** Rebuilding
  changes the positions and silently invalidates every previously recorded number.

### Read null results correctly

Per-position loss has a standard deviation near 90–140 cp. A 50-position run resolves
nothing. `compare` always prints the N needed for the variance it actually observed,
and says outright when a result is underpowered. **"Not significant" means "this
experiment could not tell", never "there is no difference."**

Run `run --uniform` for a positive control: search with no network at all. It must
come out clearly worse. If a comparison against *that* is not significant, the harness
has no power on that sample and no null result from it should be believed.

## elo.py — head-to-head games

The only instrument here that measures strength as a player experiences it.

```bash
python bench/elo.py h2h --a current:600 --b inf-rule:600 --pairs 80
python bench/elo.py gauntlet --a current:600 --elos 1320 --pairs 25
python bench/elo.py variants
```

Design choices that make a few hundred games informative: **paired openings with colour
reversal** (statistics over pairs, so opening difficulty and colour advantage cancel),
**near-balanced starting positions** reusing the Stockfish evaluations already cached in
the suite, **independent Stockfish adjudication** rather than the engine's own eval,
which would be circular, and the **Lichess tablebase probe disabled** so no measurement
depends on a third-party service.

Every run prints two self-checks. If the colour-reversed games of a pair are
move-for-move identical, the arms are the same engine and the run measures only colour —
that failure once produced a perfectly plausible "Elo 0.0". A large White/Black
imbalance is reported separately, because that is an engine property, not an arm
difference.

**Gauntlet caveat:** `UCI_Elo` assumes standard time controls. Stockfish gets 100 ms per
move here, so it plays below its nominal rating and the absolute figure *flatters* the
engine. Read it as a soft upper bound. The `h2h` differences do not have this problem.

**Comparing two weak engines through a much stronger common opponent discriminates
poorly** — both lose, scores compress against the floor. The gauntlet showed 100 and 600
sims scoring identically against Stockfish 1320; direct `h2h` put them 328 Elo apart.
Use `h2h` for A/B, and the gauntlet only for an absolute anchor.

## test_deploy.py — deployability

Launches the app in a subprocess four ways and asserts it can never come up healthy
while playing randomly:

| case | expectation |
|---|---|
| checkpoint + ONNX | serves |
| ONNX only, no `.pt` | serves, **same move** as above |
| neither | refuses to start, non-zero exit |
| `import torch` blocked | serves, torch never enters `sys.modules` |

Case B simulates a clone-based deploy, and case D proves the serving path genuinely
does not need torch rather than merely not using it today.

## test_edge_cases.py

Covers what `verify.py` structurally cannot, because it runs the engine exactly as the
web app does (`temperature=0`, `add_noise=False`, model loaded): the Dirichlet noise
path, lazily-created terminal nodes, the `model=None` fallback, ONNX-vs-torch
agreement, and the visit-distribution key set.

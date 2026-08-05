"""
Deployment regression test: the server must never come up healthy and play randomly.

This guards a failure mode that is invisible in normal testing. The web app loads a
torch checkpoint that is NOT in git; on a clone-based deploy that load fails, and the
engine used to fall back to a uniform random policy behind a printed warning. Every
route returned 200, the board rendered, moves were legal — and the AI was a
random-move generator. Nothing about the running site looked wrong.

Three cases are checked by launching the app in a subprocess:

  A  checkpoint + ONNX both reachable          -> serves, uses ONNX
  B  ONNX only, no torch checkpoint            -> serves, SAME move as A
  C  neither reachable                         -> refuses to start, non-zero exit

Case B is simulated by running from a foreign working directory: app.py resolves
MODEL_PATH relative to the CWD, while the ONNX path is absolute (derived from
neural_network.py's own location). That reproduces a deploy with only the export,
and doubles as a check that a launcher started from the wrong directory (systemd,
supervisor, a Docker WORKDIR mismatch) cannot silently degrade the AI.

Run:  python bench/test_deploy.py
"""

import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FOREIGN_CWD = os.path.expanduser("~")

# Makes `import torch` fail, so we can prove the serving path never needs it.
BLOCK_TORCH = r'''
import sys
from importlib.abc import MetaPathFinder
class _BlockTorch(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is blocked by test_deploy.py")
        return None
sys.meta_path.insert(0, _BlockTorch())
'''

PROBE = r'''
import sys, chess
sys.path.insert(0, r"{proj}")
import app as A
c = A.app.test_client()
gid = c.post("/api/new-game", json={{"difficulty": "easy"}}).get_json()["game_id"]
b = A.games[gid]
for uci in ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","d2d3","g8f6"]:
    b.push(chess.Move.from_uci(uci))
j = c.post("/api/move", json={{"game_id": gid, "move": "b1c3"}}).get_json()
ai = [h for h in j["history"] if h["color"] == "black"]
print("RESULT",
      "torch" if A.dual_model is not None else "no-torch",
      "onnx" if A._inference_session is not None else "no-onnx",
      ai[-1]["san"],
      "torch-imported" if "torch" in sys.modules else "torch-absent")
'''.format(proj=PROJ)


def run(cwd, env_extra=None, prelude=""):
    env = dict(os.environ)
    env.pop("CHESS_USE_ONNX", None)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([PY, "-c", prelude + PROBE], cwd=cwd, env=env,
                       capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    return r.returncode, (line.split()[1:] if line else None), r.stdout + r.stderr


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  - ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("\nA. checkpoint + ONNX both reachable")
    code_a, res_a, _ = run(PROJ)
    check("serves a move", code_a == 0 and res_a is not None, str(res_a))

    print("\nB. ONNX only, torch checkpoint unreachable")
    code_b, res_b, log_b = run(FOREIGN_CWD)
    check("still starts", code_b == 0, f"exit {code_b}")
    check("has no torch model", bool(res_b) and res_b[0] == "no-torch", str(res_b))
    check("uses ONNX", bool(res_b) and res_b[1] == "onnx", str(res_b))
    check("plays the SAME move as case A (real weights, not random)",
          bool(res_a) and bool(res_b) and res_a[2] == res_b[2],
          f"A={res_a[2] if res_a else '?'} B={res_b[2] if res_b else '?'}")

    print("\nC. neither reachable")
    code_c, res_c, log_c = run(FOREIGN_CWD, {"CHESS_USE_ONNX": "0"})
    check("refuses to start", code_c != 0, f"exit {code_c}")
    check("does not serve a move", res_c is None)
    check("explains what is missing", "FATAL: no usable model" in log_c)

    print("\nD. torch not installed at all (the real deployment shape)")
    code_d, res_d, log_d = run(PROJ, prelude=BLOCK_TORCH)
    check("serves without torch", code_d == 0 and res_d is not None,
          f"exit {code_d}" + ("" if res_d else " / " + log_d.strip().splitlines()[-1:][0]
                              if log_d.strip() else ""))
    check("torch never entered sys.modules",
          bool(res_d) and res_d[3] == "torch-absent", str(res_d))
    check("plays the SAME move as case A", bool(res_a) and bool(res_d)
          and res_a[2] == res_d[2],
          f"A={res_a[2] if res_a else '?'} D={res_d[2] if res_d else '?'}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("deployment checks passed - the server cannot silently serve random moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())

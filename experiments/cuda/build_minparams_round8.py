"""Propose coarse residual-scale profiles using only completed search evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from nas.cuda_minparams_search import architecture_hash, parameter_count, validate

EVIDENCE = "d2d0d08d70fad3f06257fddc651763c8c040f9989a09c054d67b1945a22c4825"
OUT = PROJECT / "experiments/cuda/minparams-20261111-agent-proposals-round8"
BASES = [
    ("n24", "minparams-20261111-agent-proposals-round5/01-r5-global-attention125-ffn075-fixed-d192n24.json"),
    ("n37", "minparams-20261111-agent-proposals-round7/04-r7-b888-d192n37-middle960-ends1408.json"),
]


def profile(kind, i, n):
    if kind == "ramp":
        value = round(0.5 + 0.75 * i / (n - 1), 6)
        return value, value
    if i == n - 1:
        return 0.75, 1.0
    if i >= n - 3:
        return 1.5, 1.25
    value = (0.5, 0.75, 1.0)[min(2, 3 * i // n)]
    return value, value


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for shape, relative in BASES:
        base = json.loads((PROJECT / "experiments/cuda" / relative).read_text())
        base_id = architecture_hash(base["frame"], base["spec"])
        for kind in ("stages", "ramp"):
            fixed_id = None
            for learned in (False, True):
                x = deepcopy(base)
                name = f"r8-{shape}-early-damped-{kind}-{'learned' if learned else 'fixed'}"
                x["name"] = x["spec"]["name"] = name
                x["parents"] = [fixed_id] if learned else [base_id, EVIDENCE]
                x["operator"] = "matched-learned-depth-profile" if learned else "fixed-depth-profile-residual-scales"
                x["spec"]["description"] = (
                    "Coarse early-damped residual scales suggested by completed search run "
                    "r00342-d2d0d08d70-s0. No checkpoint weights are reused. "
                    f"Profile={kind}; learned={learned}. The matched fixed and learned pair "
                    "has identical scalar initialization and unchanged common parameter initialization. "
                    "Only authorized branch scalars change relative to the structural parent; "
                    "all data, optimizer, dropout, precision and 5000-step training settings stay fixed."
                )
                n = len(x["spec"]["layers"])
                for i, layer in enumerate(x["spec"]["layers"]):
                    for branch, value in zip(("attn", "ffn"), profile(kind, i, n)):
                        layer[branch + "_scale"] = {"init": value, "learned": True} if learned else value
                validate(x["frame"], x["spec"])
                count = parameter_count(x["frame"], x["spec"])
                aid = architecture_hash(x["frame"], x["spec"])
                assert count < 11_312_256
                if not learned:
                    fixed_id = aid
                filename = f"{len(rows)+1:02d}-{name}.json"
                (OUT / filename).write_text(json.dumps(x, indent=2) + "\n")
                rows.append({"file": filename, "parameters": count, "id": aid})
    assert len({r["id"] for r in rows}) == 8
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

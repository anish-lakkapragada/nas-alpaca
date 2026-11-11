# Architecture search log

Every attempt, in order, including failures. Dev = 1,000 conversations held out from
train.json; validation.json is untouched. Baseline = N=8 GPT, 27,144,960 parameters.
Δ = best dev loss minus the baseline's at the same K (negative is better).

| # | Attempt | K | Layout | Params | Best dev | Δ vs base | Best step | s/step | Plot |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | s00-baseline-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7` | 27,144,960 | 3.8110 | +0.0000 | 2000 | 0.727 | [loss.png](s00-baseline-k2000/loss.png) |

## Notes per attempt

- **s00-baseline-k2000** — Reference: unmodified N=8 baseline at this K.

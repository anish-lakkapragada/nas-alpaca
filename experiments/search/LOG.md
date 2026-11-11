# Architecture search log

Every attempt, in order, including failures. Dev = 1,000 conversations held out from
train.json; validation.json is untouched. Baseline = N=8 GPT, 27,144,960 parameters.
Δ = best dev loss minus the baseline's at the same K (negative is better).

| # | Attempt | K | Layout | Params | Best dev | Δ vs base | Best step | s/step | Plot |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | s00-baseline-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7` | 27,144,960 | 3.8110 | +0.0000 | 2000 | 0.727 | [loss.png](s00-baseline-k2000/loss.png) |
| 2 | s01-relu2-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7  [all FFN: relu2,1536]` | 27,144,960 | 3.6684 | -0.1425 | 2000 | 0.714 | [loss.png](s01-relu2-k2000/loss.png) |
| 3 | s02-silu-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7  [all FFN: silu,1536]` | 27,144,960 | 3.9754 | +0.1644 | 2000 | 0.726 | [loss.png](s02-silu-k2000/loss.png) |

## Notes per attempt

- **s00-baseline-k2000** — Reference: unmodified N=8 baseline at this K.
- **s01-relu2-k2000** — Activation: GELU -> ReLU^2 in every FFN (0 params). Prior: Primer found ReLU^2 helps small LMs.
- **s02-silu-k2000** — Activation: GELU -> SiLU in every FFN (0 params).

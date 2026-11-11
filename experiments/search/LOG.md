# Architecture search log

Every attempt, in order, including failures. Dev = 1,000 conversations held out from
train.json; validation.json is untouched. Baseline = N=8 GPT, 27,144,960 parameters.
Δ = best dev loss minus the baseline's at the same K (negative is better).

| # | Attempt | K | Layout | Params | Best dev | Δ vs base | Best step | s/step | Plot |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | s00-baseline-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7` | 27,144,960 | 3.8110 | +0.0000 | 2000 | 0.727 | [loss.png](s00-baseline-k2000/loss.png) |
| 2 | s01-relu2-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7  [all FFN: relu2,1536]` | 27,144,960 | 3.6684 | -0.1425 | 2000 | 0.714 | [loss.png](s01-relu2-k2000/loss.png) |
| 3 | s02-silu-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7  [all FFN: silu,1536]` | 27,144,960 | 3.9754 | +0.1644 | 2000 | 0.726 | [loss.png](s02-silu-k2000/loss.png) |
| 4 | s03-cycle4-k2000 | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A0/F0 A1/F1 A2/F2 A3/F3` | 20,060,928 | 3.9463 | +0.1354 | 2000 | 0.695 | [loss.png](s03-cycle4-k2000/loss.png) |
| 5 | s00-baseline-k2000-s1-aborted | 2000 | `A0/F0 A1/F1 A2/F2 A3/F3 A4/F4 A5/F5 A6/F6 A7/F7` | — | FAILED (rc -15) | | | | [log](s00-baseline-k2000-s1-aborted/train.log) |
| 6 | s04-loop4x2-inj-k2000 | 2000 | `A0/F0x2+inj A1/F1x2+inj A2/F2x2+inj A3/F3x2+inj` | 20,060,928 | 3.8967 | +0.0857 | 2000 | 1.021 | [loss.png](s04-loop4x2-inj-k2000/loss.png) |

## Notes per attempt

- **s00-baseline-k2000** — Reference: unmodified N=8 baseline at this K.
- **s01-relu2-k2000** — Activation: GELU -> ReLU^2 in every FFN (0 params). Prior: Primer found ReLU^2 helps small LMs.
- **s02-silu-k2000** — Activation: GELU -> SiLU in every FFN (0 params).
- **s03-cycle4-k2000** — Tie attention+FFN of layer i and i+4 (ABCDABCD): 4 unique layers, 8 applications. Data-limited regime: sharing may regularize.
- **s00-baseline-k2000-s1-aborted** — ABORTED at step ~500: run in parallel with s04 to test 2-job concurrency. The GPU time-sliced the jobs (this run 3.7 s/step at first, then s04 stalled 7+ min) and swap rose 8.7 -> 13.5 GB, so no throughput gain. Concurrency abandoned; seed check rerun sequentially later.
- **s04-loop4x2-inj-k2000** — Loop each of 4 layers twice with input injection (Yang et al.): 8 applications, 4 unique layers.

# Round 8: coarse residual-scale profiles

These eight proposals use the approved fixed/learned residual-scalar primitive.
All data, optimizer settings, dropout, precision, checkpoint rules, and 5,000
training steps remain fixed. No checkpoint weights are reused.

The hypothesis comes from the completed search run `r00342-d2d0d08d70-s0`:
its learned attention scalars were mostly 0.45–0.62 early, 0.7–1.1 in the
middle, and 0.94–1.6 late (final attention 0.856). FFN scalars also tended
to be smaller early and larger late. The checkpoint hash was verified before
CPU inspection; the measurement is recorded under the server run root at
`search-comparisons/20261112T0230-learned-scalars.json`. This does not establish
that transferring a scale profile will help another initialization or model.

Two deliberately coarse profiles avoid copying the exact learned values:

- `stages`: both branches use 0.5 / 0.75 / 1.0 by depth thirds; the two
  penultimate entries instead use attention 1.5 and FFN 1.25; the final
  entry uses attention 0.75 and FFN 1.0.
- `ramp`: both branches increase linearly from 0.5 to 1.25 across the full
  execution depth, rounded to six decimals.

Each profile is tested on two structural parents, with matched fixed and
learned variants. Common model weights and all initial scalar values must be
identical within each pair. Learned variants add one scalar for each branch
application. The n37 transfer is a new hypothesis; its structural parent was
submitted in round 7 and has not yet established an improvement.

| Entries | Structural parent | Fixed parameters | Learned parameters |
|---|---|---:|---:|
| 24 | Round 5, seven shared middle blocks, FFN832/ends1280 | 11,041,920 | 11,041,968 |
| 37 | Round 7 proposal04, improved attention schedule, FFN960/ends1408 | 11,188,608 | 11,188,682 |

Reproduce the JSONs with `python3 experiments/cuda/build_minparams_round8.py`.
Scheduler schema/count validation and a separate structural-parent review
passed. Frozen `SpecGPT` meta construction independently matched all eight
parameter counts. For each of the four fixed/learned pairs, common initial
weights and BF16 CPU outputs were exactly equal on a synthetic eight-token
input. CUDA remained uninitialized. All eight identities were absent from the
live search ledger at validation. The server record is
`operations/round8-validation.json` (2026-11-12 02:37:40 UTC).

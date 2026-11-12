# Active NVIDIA minimum-parameter search — November 11, 2026

This section supersedes the historical MLX protocol below for this branch.

- Branch: `research/nvidia-minparams`, based on `01e20c2`.
- Goal: minimize total parameters below the original 27,144,960 while matching
  or beating its mean dev CE. Local parameter increases are allowed when the
  total decreases. All authorized primitives are recorded in
  `experiments/cuda/minparams-20261111-authorization.json`.
- Every training run uses the same data, tokenizer, packing, optimizer,
  schedule, dropout, precision, checkpoint rule, and exactly 5,000 steps.
  Data order is paired by seed and independent of architecture initialization.
- Screen seed 0; confirm candidates against baseline on seeds 0/1/2.
  Freeze one smallest qualifying architecture. Train it and the baseline on
  fresh seeds 100–104, then evaluate the fixed held-out test split. No reselection
  after confirmation. The test split was previously scored in the earlier
  CUDA study; this limitation must remain explicit in the final report.
- Server: `ubuntu@192.222.54.228`, four H100 80GB GPUs. Project:
  `/home/ubuntu/nas-minparams`; Python:
  `/home/ubuntu/nas-minparams-venv/bin/python`.
- Run root: `/home/ubuntu/nas-minparams/runs/cuda/minparams-20261111`.
  Append-only `architecture_registry.jsonl` and `run_registry.jsonl` retain
  every proposal/attempt, including failures, with source/data identities,
  host/GPU, seed, parameters, recipe, timestamps and checkpoint metadata.
  Each attempt retains train/dev curves and metrics; lower-value checkpoint
  binaries may be pruned with a retention record.
- Independent server service: `nas-minparams-20261111.service`, started
  **2026-11-11 23:29:51 UTC / 3:29:51 PM PT**. It automatically runs search,
  freezes selection, performs fresh-seed confirmation, and writes the report.
  Laptop connectivity is not required. No API or SSH secrets were copied to
  the server.
- Search cutoff: **9:30 PM PT**. Final confirmation/report target: **10:25 PM
  PT**. Server timer `nas-minparams-20261111-training-cutoff.timer` stops this
  training service at **10:30 PM PT / 2026-11-12 06:30 UTC**. This does not
  terminate the rental; the user has not authorized termination of this instance.
- Verification: original CUDA tests and expanded-architecture GPU tests pass,
  including exact baseline weights/outputs/gradients against the frozen old
  implementation. 19 controller tests pass. Synthetic checkpoint/evaluation
  integration passes across differing widths/depths. Fixed non-unit residual
  scales now use nonpersistent tensors so matched fixed/learned initial values
  have identical BF16 arithmetic; the baseline unit-scale path is unchanged.
- Frozen execution sources/data are hashed in `workflow_source_manifest.json`.
  Do not edit deployed source during this experiment. Additional architecture
  proposals may be supplied atomically through `inbox/*.json` as
  `{name, frame, spec, parents?, operator?}`; these are fully logged.

### Live handoff snapshot — 2026-11-12 00:59 UTC / 4:59 PM PT

The goal remains active; selection and final confirmation are **not complete**.
At this snapshot, 194 runs had completed across 144 measured architectures,
without failed/interrupted/timed-out runs. The controller is PID 5643, supervised
by workflow PID 5640. Always recheck live service/process state and current
ledgers rather than assuming this snapshot remains current.

Search baseline seeds 0/1/2: 3.4088462352 / 3.4191115265 / 3.4342312650,
mean **3.4207296756**. Each finished 5,000 steps (~87 training seconds, ~101
seconds including process/setup overhead). Completed runs were audited for
frozen source/data identity and paired training token counts: seed 0 has
27,974,610 supervised tokens, seed 1 has 27,976,370, seed 2 has 27,989,048.

Smallest qualifying three-seed mean so far: **11,312,256 parameters**, architecture
`d0a6bb3cf3dc0c558a36ff57fefdc7cf07b8268578f78d5522388b7b460fc939`,
mean dev **3.4173459045**. Its individual values rounded to seven decimals are
3.4236843, 3.4182718, and 3.4100817. It loses to baseline
on seed 0 and has only a thin mean advantage; do not imply all-seed dominance.
The previously discussed 14,464,896-parameter model
`3ae00b99f608e4ba9754404a16d1fcf88dfd1ce342c2a58975ace88aaf19adf9`
has mean dev 3.4093774553 and does beat baseline on all three search seeds.
These are search results, not fresh-seed or held-out test confirmation.

Additional proposals were committed and submitted through the inbox:
`experiments/cuda/minparams-proposals/` (16), `minparams-proposals-round2/`
(16), and `minparams-proposals-round3/` (14). Round 3 targets 9.84–12.30M,
including complete repeated passes, separate attention/FFN sharing groups,
matched-initialization learned scalars, and a documented width160/head5×32
model. All other proposed widths use head dimension64. The expanded user
authorization allows embedding-width changes with compatible attention heads.
The autonomous adaptive search also continues after breadth phases.

`nas-minparams-report-20261111.timer` runs the independent, read-only
`cuda_minparams_report.py` every five minutes, producing `audit-report/`
with every-run/architecture Markdown logs, summary JSON, and PNG/SVG plots.
It stops at 06:31 UTC through `nas-minparams-report-cutoff-20261111.timer`.
Local artifact backups (excluding `.pt` checkpoints) are under
`/Users/anish/Documents/git-repos/nas/.lambda/artifacts/287cdb48d8e64da3b849dceaaf9b3bc2/cuda/minparams-20261111/`.

After selection freezes, run the separate `cuda_minparams_architecture_report.py
--root RUN_ROOT` to export exact ASCII execution and independently reconciled
parameter accounting under `architecture-report/`. This reads no training or
test examples. The frozen workflow already handles ten fresh training runs,
held-out scoring, plots and `FINAL_REPORT.json`/`.md` automatically. Verify
those actual artifacts, successful paired data/step checks, no reselection,
and complete logs before declaring the goal complete. Back up final reports,
all logs/curves/specs, and the selected fresh checkpoints locally. Report any
failure to match test mean honestly, including the existing test-reuse caveat.

The standalone `cuda_minparams_audit.py --root RUN_ROOT` must run **after**
`FINAL_REPORT.json` is complete. It verifies all architecture/run inventories,
fixed recipes and paired token progressions, provenance and retained checkpoint
hashes, exact smallest-qualifier selection, five fresh pairs, evaluation timing,
and test means recomputed from the saved conversation arrays. It reads no model
data and performs no model scoring. Its outputs go only under `final-audit/`;
exit 0 means integrity pass, 1 means a failed check, and 2 means incomplete or
invalid invocation. It explicitly reports recorded checkpoint pruning and any
restart-related evidence limitations. Synthetic normal/restart/corruption cases
pass. It does not independently rerun bootstrap interval draws.

`cuda_minparams_postprocess.py` is a separate server-side waiter for the final
report. Once the frozen workflow reports completion, it exports the selected
architecture, refreshes the readable run/architecture report, and runs the final
audit. It never trains or scores a model. The independent systemd unit is
`nas-minparams-postprocess-20261111.service`; its output is recorded in
`postprocessing.log` and `postprocessing.json` under the run root. Check its
actual exit status and audit results; its presence alone is not completion.

Round 4 adds 16 proposals under
`experiments/cuda/minparams-20261111-agent-proposals-round4/`, all below
11,312,256 parameters. They were committed and accepted by the live scheduler
after frozen CUDA schema/count validation, including matched fixed/learned
scalar pairs, FFN capacity redistribution, and shared-layer execution schedules.

---

# Historical MLX handoff (superseded for this branch)

# Handoff: agent architecture search on Alpaca (branch `nas`)

**Branch:** `nas`, an orphan branch with no history from the other branches.
**Worktree:** `/Users/anish/Documents/git-repos/nas/.worktrees/nas`
**Last updated:** 2026-11-10 23:10 PST. Search budget: 12 hours, from 20:15 to about 08:15 on Nov 11.

## Goal

Produce an honest result for this tweet:

> This Saturday, I got bored and wanted to test out agents on their ability to
> optimize architectures. I chose a humble setup for a local MacBook: a standard
> N-layer decoder transformer on the Stanford Alpaca dataset that achieves XXX
> loss after K optimizer steps. I gave an agent the trained model and told it to
> search for a modified architecture constrained to a few modifications [...] and
> it cooked up a model achieving YYY loss after K optimizer steps with ZZ fewer
> parameters.

**Objective order (set by the user):**
1. **Primary: lower loss** than the baseline (YYY < XXX).
2. **Secondary: fewer parameters.** Some reduction is required. Parameter cuts
   are taken only where they cost little or no loss; loss is never traded away
   for a bigger cut.

## The baseline (approved by the user): N = 8, K = 5,000

- GPT-2-style decoder: 8 layers, width 384, 6 heads, GELU MLP 1,536, pre-LayerNorm,
  learned positions, tied 32,768-token BPE word table, dropout 0.1, no linear biases.
  **27,144,960 parameters.**
- Training: K = 5,000 AdamW steps (lr 1e-3, warmup 100, cosine to 10%, wd 0.1,
  betas 0.9/0.95, clip 1.0). Each step is 8 rows of 1,024 packed tokens, about
  10 passes over the data. Conversations are packed with a block-diagonal mask,
  verified to give the same loss as scoring each conversation separately.
- Best **dev loss 3.402** (at step 4,750). Runs at about 0.75 s/step (about 64 min)
  on an M3 Pro with 18 GB, using MLX 0.32.4.
- Depth was chosen by sweep: N=4 gives 3.522, N=6 gives 3.453, N=8 gives 3.402
  (`experiments/baseline-depth/`).

## Data and metric

- Stanford Alpaca (`data/`, with source and hash in `data/source.json`):
  `train.json` has 49,375 records and `validation.json` has 2,599.
- **Dev split:** 1,000 records carved out of `train.json` (seed 1234). All
  selection uses dev. Training uses the remaining 48,375 records.
- **`validation.json` is reserved:** it is scored once at the end for the final
  baseline and the winner (`evaluate.py --split validation`). It has not been used yet.
- **Loss:** mean cross-entropy in nats per token over **assistant answer tokens
  plus the answer's EOS**. Prompts are read but not scored.
  The tokenizer was fit on `train.json` only.

## Allowed architecture modifications (the user's whitelist; nothing else)

| # | Primitive | Spec field |
|---|---|---|
| 1 | Loop a layer r times (optionally re-adding its input each pass, per [Yang et al. 2024](https://arxiv.org/abs/2311.12424)) | `"loop": r, "inject": bool` |
| 2 | Tie two attention blocks (non-adjacent layers) | same `attn` id |
| 3 | Tie two FFNs (non-adjacent layers) | same `ffn` id |
| 4 | Change the activation (elementwise: relu, relu2, silu, gelu_tanh) | `ffns[id].act` |
| 5 | Delete an attention block | `"attn": null` |
| 6 | Delete an FFN | `"ffn": null` |
| 7 | Transpose-tie an FFN (down = up^T) | `ffns[id].transpose_tie` |
| 8 | Narrow an FFN (hidden width < 1,536) | `ffns[id].hidden` |

**Fixed throughout:** width 384, 6 heads, vocabulary and tied word table,
context 1,024, data, K, and the optimizer recipe. A model may use at most 8
layer applications (loops count), and parameters must be <= 27,144,960.
No new learned weights may be added (no per-loop norms, LoRA, etc.).
A shared sublayer shares its pre-LayerNorm. `arch.validate` enforces all of this.

## Code

- `arch.py`: spec-driven model implementing exactly the 8 primitives. The baseline
  spec reproduces `model.GPT` bit-for-bit at the same seed (`test_arch.py`).
- `train.py`: MLX trainer (`--spec`, `--max-params`). Keeps the best-dev checkpoint.
- `search.py`: runs one attempt and writes `experiments/search/<attempt>/`
  (spec, config, log, samples, **loss.png**), `results.jsonl` and **`LOG.md`**.
  Attempts are never overwritten.
- `queue.sh STEPS spec...`: runs attempts sequentially and commits each one
  (`SEED=n` for other seeds).
- `evaluate.py`: final scorer with a paired conversation bootstrap.
- Weights are kept only in `runs/` (gitignored); checkpoint hashes are committed.

## Search protocol

- **Screens use K = 2,000** (about 25 min) against a K = 2,000 baseline (`s00`, dev 3.811).
  The old K = 5,000 curve can't stand in for it, because the learning-rate
  schedule depends on K.
- Promising candidates are retrained at **K = 5,000** against the baseline's 3.402.
- **One job at a time.** A two-job test made the GPU alternate between jobs:
  one stalled for 7+ minutes and swap rose from 8.7 to 13.5 GB, with no throughput
  gain. It is logged as `s00-baseline-k2000-s1-aborted`.
- Caveat: at K = 2,000 the baseline barely overfits, while at K = 5,000 it overfits
  strongly (train 2.53 vs dev 3.40). Regularizing changes such as sharing may look
  worse in screens than they would at full K.

## Results so far (dev loss, K = 2,000; baseline 3.811 with 27.14M parameters)

| Attempt | Change | Params | Dev | Δ |
|---|---|---:|---:|---:|
| s01 | ReLU² everywhere | 27.14M | **3.668** | **−0.143** |
| s02 | SiLU everywhere | 27.14M | 3.975 | +0.164 |
| s03 | Cycle tie ABCDABCD (4 unique layers) | 20.06M | 3.946 | +0.135 |
| s04 | Loop 4 layers ×2, with input injection | 20.06M | 3.897 | +0.086 |
| s05 | Loop 4 layers ×2, no injection | 20.06M | 3.872 | +0.061 |
| s06 | Transpose-tie all FFNs | 22.43M | 4.634 | +0.823 (dead end) |
| s07–s09 | narrow all FFNs to 1,024 / delete 2 attention / delete 2 FFN | | running | |

The full table, with plots for every attempt including failures, is in
`experiments/search/LOG.md`.

**Read so far:** ReLU² is the strongest lever, at no parameter cost. Looping is
the cheapest large cut. The large SiLU-vs-GELU gap suggests seed noise may be
sizeable, which is not yet measured.

## Plan for the rest of the window

1. Finish round 1 (s07–s09), around 00:15.
2. **ReLU² at K = 5,000** against 3.402. This checks whether the biggest gain is real
   at the target K and also serves as the noise check.
3. **Round 2 (K = 2,000):** ReLU² as the base, plus *small, cheap* cuts (loop or tie one
   or two layers, narrow some FFNs, delete one attention block), plus the large-cut
   ReLU² + loop 4×2 for comparison. Pick by lowest loss; a bigger cut breaks ties.
4. Run the best 1–2 at K = 5,000.
5. **Final:** retrain the baseline and the winner with a fresh seed, then score both
   once on `validation.json` with a paired bootstrap. This gives XXX, YYY and ZZ
   for the tweet.

## How to resume

```bash
cd /Users/anish/Documents/git-repos/nas/.worktrees/nas
cat experiments/search/LOG.md                      # everything tried so far
ps aux | grep train.py                             # anything running?
caffeinate -i ./queue.sh 2000 experiments/search/specs/<spec>.json   # new attempt
.venv/bin/python test_arch.py                      # sanity checks
```

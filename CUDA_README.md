# NVIDIA architecture search

This branch adds a PyTorch/CUDA path beside the original MLX experiment. It
starts from `nas` commit `317c3d4`; the original implementation and data files
remain the reference for masking, packing, splits, and training semantics.

## Baseline frame

| Setting | NVIDIA baseline |
| --- | --- |
| Layers | 10 |
| Hidden width | 512 |
| Attention heads | 8 (64 dimensions each) |
| FFN hidden width | 2,048 |
| Vocabulary | 32,768, tied input/output embeddings |
| Context | 1,024, learned positions reset per conversation |
| Parameters | **48,780,288** |
| Optimizer steps | **5,000** |
| Batch | 8 packed rows; 8,192 positions per step |
| Objective | Mean answer-token cross entropy, including answer EOS |
| Dropout | 0.1 |

The user requested a roughly 50M-parameter baseline for this branch. The
architecture search therefore has a 10-application ceiling and a 48,780,288
parameter ceiling. The eight allowed modifications are unchanged. Loss is the
primary objective; the final candidate must also reduce parameters.

The optimizer matches the installed MLX AdamW defaults: betas `(0.9, 0.95)`,
epsilon `1e-8`, weight decay `0.1`, **bias correction disabled**, clipping norm
`1.0`. Learning rate is `1e-3`, with 100 warmup steps and cosine decay to 10%.
The PyTorch adapter adjusts fused AdamW's learning rate, epsilon, and decay
together to reproduce that uncorrected update; logged learning rates are the
original recipe rates. A standard unmodified PyTorch AdamW would differ.

CUDA training uses BF16 autocast and FP32 master parameters, moments, and loss.
Consequently, the baseline and every candidate must be compared on this same
path. The original MLX loss of 3.402 is historical context, not this frame's
baseline. Equal seeds across frameworks do not imply equal random draws.

## Setup and validation

On the Linux GPU host, from the checkout:

```bash
bash prepare_cuda.sh
.venv-cuda/bin/python test_cuda.py
```

`export_mlx_fixture.py` creates deterministic weights, logits, answer losses,
and gradient samples from the original MLX implementation. Numerical parity
is measured in FP32 with dropout disabled. Fixtures live under ignored
`runs/`; they do not contain the held-out validation dataset.

The trainer selects answer positions before the tied vocabulary projection.
Prompts still participate in attention and receive gradients, but their
unscored vocabulary logits are not materialized. Tests compare this optimized
objective and its gradients with the full masked objective.

## Measure speed before the overnight queue

```bash
CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python cuda_train.py \
  --out runs/cuda/benchmark-eager --steps 5000 \
  --benchmark-warmup 20 --benchmark-steps 100

CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python cuda_train.py \
  --out runs/cuda/benchmark-compiled --steps 5000 --compile \
  --benchmark-warmup 20 --benchmark-steps 100
```

Each benchmark keeps the 5,000-step learning-rate schedule. Its `summary.json`
separates measured steady-state step time from the estimated full run and dev
evaluation cost. These are **estimates** until a full run completes. Compilation,
checkpointing, and data preparation are reported separately or explicitly
excluded. Measure simultaneous runs too before assuming four independent runs
have the same throughput as one.

## Run a baseline

```bash
CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python cuda_train.py \
  --out runs/cuda/baseline-k5000-s0 --steps 5000 --seed 0
```

Use another visible GPU and distinct output directory for seed 1. Add
`--compile` only if the measured end-to-end benefit justifies its startup time.
Do not increase batch size to improve throughput: that would change the
number of training tokens at the fixed step budget.

Runs write `config.json`, `log.jsonl`, `summary.json`, best/final checkpoints,
and periodic resumable checkpoints. Configurations record source/data hashes,
software versions, hardware, precision, seed, and architecture. Existing
attempt directories are rejected. Resume only trusted checkpoints with the
same original recipe and output directory.

## Search and final evaluation

`cuda_search.py` manages one process per GPU, distinct attempt directories,
baseline replicates, 2,000-step screens with their own baseline, full-length
finalists, and a fresh-seed confirmation. It never opens `validation.json`.
Short screens can underrate sharing/regularization, so the shortlist includes
a reserved regularization candidate. Rank on loss first; parameter savings
break ties, and final eligible candidates must be smaller than the baseline.

Only after choosing the final baseline/winner pair:

```bash
.venv-cuda/bin/python cuda_evaluate.py \
  runs/cuda/final-baseline runs/cuda/final-winner \
  --split validation --out experiments/cuda/final-validation.json
```

The final scorer produces token-weighted losses and a paired conversation
bootstrap. Validation is never a search or early-stopping signal.

## Cloud access and rental limits

The Lambda credential is in the main checkout's ignored `.env`, mode `0600`.
It is not copied to this worktree's tracked files or to the GPU host. The SSH
key and local rental state are under the main checkout's ignored `.lambda/`.

The user approved up to **$200 and 12 hours**. At the console prices provided
by the user, the selected 4x H100 SXM5 instance costs **$16.36/hour**. Cap that
rental at **10 hours ($163.60 before taxes)** and finish within the overall
12-hour deadline. The hard deadline is the earlier of those two limits.

`lambda_watchdog.py` can copy results periodically and terminate only the
exact task instance IDs recorded in the local state file. It needs working
Lambda API access and a running local host. **Do not treat that cutoff as
armed until its API access and background process have been verified.**
Stopping training or powering down the guest alone does not establish that
cloud billing has ended; confirm termination in Lambda.

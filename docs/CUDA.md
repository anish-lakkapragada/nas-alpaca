# NVIDIA architecture search

## Current package layout

Run commands from the repository root with `python -m nas.<module>`. Model and
training code lives in `nas/`, cloud utilities in `nas/cloud/`, tests in `tests/`,
and setup scripts in `scripts/`. The original MLX implementation is in `nas/mlx/`.

For the completed minimum-parameter study and the current candidate C choice,
see [HANDOFF.md](HANDOFF.md). The original experiment snapshot is preserved in
commit `4823477` and the local artifact archive. The earlier-study measurements,
deadlines, and cloud deployment notes below are historical records.

This branch adds a PyTorch/CUDA path beside the original MLX experiment. It
starts from `nas` commit `317c3d4`; the original implementation and data files
remain the reference for masking, packing, splits, and training semantics.

## Current baseline and approval boundary

On 2026-11-11 the user selected the original **27,144,960-parameter** baseline:
8 layers, width 384, 6 attention heads, and FFN width 1,536. The user authorized
four independent 5,000-step baseline runs, one per H100, with seeds 0–3.
All four completed in 98.1–98.8 seconds per run, concurrently. Best dev losses
were 3.407, 3.418, 3.439, and 3.404; the measured records and full configurations
are in `experiments/cuda/baselines-27m.json`.

The user subsequently **authorized the overnight architecture search** on this
27.14M frame, with four H100s, a total rental budget of $200, and results due
**2026-11-11 at 08:15 PST (16:15 UTC)**. Training stops by 07:45 PST to reserve
time for final evaluation and backups. The model/trainer defaults now use the
original frame. The 50M records below remain historical comparisons.

The exact whitelist is the original eight primitives (loop with optional input
injection, non-adjacent attention sharing, non-adjacent FFN sharing, the four
approved activation alternatives, delete attention, delete FFN, transpose-tie
an FFN, narrow an FFN), plus **fixed or learned scalar residual scaling** and
**reversing attention/FFN order**. Fixed and learned scalar variants start at
the same scalar value. Scalars belong to a layer entry and branch, and are
reused across that entry's loop repetitions. Learned scalars are the sole
authorized new parameters and are included in every reported parameter count.
At most eight layer applications are allowed; shared sublayers share their
pre-LayerNorm. Data, width, heads, context, batch and optimizer recipe stay fixed.
Lower loss comes first; a final winner must also have fewer total parameters.

The search uses full 5,000-step trials, an append-only run ledger, canonical
architecture specifications, train/dev curves, and confirmation with fresh
seeds. The run root is `runs/cuda/search-27m-20261111/`; backups are under the
main checkout's ignored `.lambda/artifacts/<instance-id>/cuda/` directory.
No proposed primitive outside this whitelist is authorized.

## Earlier 50M baseline frame

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

The initial request was a roughly 50M-parameter baseline for this branch.
Those measurements are historical. The active scheduler enforces the original
eight-application, 27,144,960-parameter frame and the ten approved primitives.

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
.venv-cuda/bin/python -m tests.test_cuda
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
CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python -m nas.cuda_train \
  --out runs/cuda/benchmark-eager --steps 5000 \
  --layers 8 --d-model 384 --heads 6 --d-ff 1536 \
  --benchmark-warmup 20 --benchmark-steps 100

CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python -m nas.cuda_train \
  --out runs/cuda/benchmark-compiled --steps 5000 --compile \
  --layers 8 --d-model 384 --heads 6 --d-ff 1536 \
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
CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python -m nas.cuda_train \
  --out runs/cuda/baseline-n8-d384-k5000-s0 --steps 5000 --seed 0 \
  --layers 8 --d-model 384 --heads 6 --d-ff 1536 \
  --max-params 27144960 --batch-rows 8 --compile
```

Use another visible GPU and distinct output directory for each additional seed.
The completed four-seed measurement above used `--compile`.
Do not increase batch size to improve throughput: that would change the
number of training tokens at the fixed step budget.

Runs write `config.json`, `log.jsonl`, `summary.json`, best/final checkpoints,
and periodic resumable checkpoints. Configurations record source/data hashes,
software versions, hardware, precision, seed, and architecture. Existing
attempt directories are rejected. Resume only trusted checkpoints with the
same original recipe and output directory.

## Search and final evaluation

`cuda_search.py` manages one process per GPU and distinct attempt directories.
Every search trial runs for 5,000 steps. It begins with matching CUDA baseline
replicates and a broad set of approved changes, then mutates promising models
and confirms a shortlist with seeds 0, 1 and 2. Lower loss comes first;
parameter savings break ties. It freezes a primary and runner-up before fresh
confirmation seeds 10, 11 and 12. The primary must be smaller than the baseline.
The search never opens `validation.json` or consumes ongoing local MLX results.
Historical imported CUDA baselines are reference records, excluded from ranking.

Only after choosing the final baseline/winner pair:

```bash
.venv-cuda/bin/python -m nas.cuda_evaluate \
  runs/cuda/final-baseline runs/cuda/final-winner \
  --split validation --out experiments/cuda/final-validation.json
```

The autonomous `cuda_overnight.py` controller invokes `cuda_finalize.py` after
successful fresh-seed confirmation. The finalizer checks the frozen selection,
recipe, provenance and checkpoints, then evaluates exactly three baseline and
three primary checkpoints on validation. It reports token-weighted losses and
a paired bootstrap over seeds and conversations. Validation is never a search,
early-stopping or candidate-reselection signal.

Every run has train/dev samples every 250 steps, with a fixed dropout-free
training subset for a comparable generalization curve. Minibatch training loss
is logged separately every 25 steps. `cuda_report.py` renders PNG/SVG curves;
`cuda_inventory.py` indexes historical, completed, failed and queued runs with
their locations, GPUs and seeds. Engineering smoke tests are explicitly
ineligible for ranking. The final report separates registered jobs from
architectures with completed training runs.

## Cloud access and rental limits

The original Lambda credential is in the main checkout's ignored `.env`, mode
`0600`. It is not copied to tracked files or to either GPU host. SSH keys and
local rental state are under the main checkout's ignored `.lambda/`.

The user approved up to **$200 and 12 hours**. The selected 4x H100 SXM5 costs
**$16.36/hour**. The absolute cutoff is **2026-11-11 16:15 UTC**. A dedicated
**$1.29/hour A10 helper** in `us-east-1` was added for durable backups after the
user requested independence from the laptop. Its `nas-backup-20261111`
filesystem remains after compute termination. Both exact task instance IDs
are registered with the local fallback cutoff; projected compute through the
deadline is about $156 before storage and taxes.

`lambda_watchdog.py` copies local fallback results and terminates only the
exact task instance IDs recorded in local state. It requires a running local
host and working API access. The detached training and adaptation already run
entirely on Lambda, using frozen inputs and their own CUDA results.

`lambda_filesystem_backup.py` provides direct server-to-server backups with
SHA-256 verification and completion receipts. Its independent systemd timer
is active on the search host. The first verified snapshot at 09:38 UTC contains
1,183 artifacts (3.49 GB); deployment details are recorded in
`experiments/cuda/cloud-independence.json`. `lambda_restore_backup.py` verifies
the content-addressed snapshot and reconstructs readable files into a new
directory. `lambda_remote_watchdog.py` and
the helper watchdog provide independent API termination once their private
credentials and services are installed. Automatic approval review required
explicit user approval for the proposed eight-hour, workspace-scoped
read/termination credential. **The remote cutoff is not armed while that
approval is pending.** Verify the deployed service state before claiming full
laptop independence. The original account-wide API key stays local.

Stopping training or powering down the guest alone does not establish that
cloud billing has ended; confirm termination in Lambda.

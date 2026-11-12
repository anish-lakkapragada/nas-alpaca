# NAS: NVIDIA architecture search on Stanford Alpaca

The primary checkout is this repository root on branch `main`. It contains the
NVIDIA experiment formerly developed in the `research/nvidia-minparams` worktree.
That redundant worktree has been removed; its branch and the earlier repository
histories are preserved. No GitHub remote is currently configured.

```text
nas/             NVIDIA model, training, search, and evaluation
  cloud/         Lambda access, backups, and watchdog utilities
  mlx/           Original MLX implementation
tests/           Model, scheduler, reporting, and backup tests
scripts/         Environment setup and queue helpers
docs/            CUDA guide and research handoff
experiments/     Architecture specifications and recorded experiments
data/            Fixed tokenizer and dataset splits
```

The user selected **candidate C**, an 11,458,944-parameter transformer, after the
completed search. It has 57.79% fewer parameters than the 27,144,960-parameter
baseline. It uses width 192 and 37 block applications, with shared attention and
FFN modules. Its exact specification is
[`experiments/cuda/candidate-c/spec.json`](experiments/cuda/candidate-c/spec.json).

The three existing search seeds beat their paired baseline on held-out CE. C has
not received five-fresh-seed confirmation. This later user choice does not change
the original experiment outcome: the frozen 10.99M candidate failed its five-seed
held-out comparison. The held-out split was reused from an earlier study.

## Code

| File | Purpose |
|---|---|
| [`cuda_model.py`](nas/cuda_model.py) | Configurable PyTorch transformer and shared modules |
| [`cuda_train.py`](nas/cuda_train.py) | Fixed training recipe, checkpoints, and train/dev curves |
| [`data.py`](nas/data.py) | Fixed data splits, tokenization, and conversation packing |
| [`cuda_minparams_search.py`](nas/cuda_minparams_search.py) | Architecture proposals, scheduling, and run logging |
| [`cuda_evaluate.py`](nas/cuda_evaluate.py) | Checkpoint loading and paired evaluation |
| [`experiments/cuda/candidate-c/metadata.json`](experiments/cuda/candidate-c/metadata.json) | C's dimensions, seeds, results, and checkpoint identities |

Run commands from the repository root using Python modules:

```bash
# Configure a Linux CUDA host:
bash scripts/prepare_cuda.sh

# Inspect the training and search interfaces:
.venv-cuda/bin/python -m nas.cuda_train --help
.venv-cuda/bin/python -m nas.cuda_minparams_search --help

# Run CPU verification in an environment with the CUDA requirements and pytest:
python -m pytest -q
```

The older MLX implementation lives in `nas.mlx` and requires MLX; its optional
suite can be run with `python -m pytest -o addopts= tests/test_arch.py`. Cloud
commands use the same module convention, for example
`python -m nas.cloud.lambda_restore_backup --help`.

Every completed run trained from scratch for 5,000 steps using the same recipe.
The split contains 48,375 training, 1,000 dev, and 2,599 held-out conversations.
The completed study trained 368 distinct architectures across 616 runs; see
[`experiments/cuda/minparams-search-volume.json`](experiments/cuda/minparams-search-volume.json)
for counts and the distinction between Python source and architecture specifications.

## Local experiment artifacts

The run logs, architecture registry, curves, reports, and backed-up checkpoints
are under the ignored directory:

```text
.lambda/artifacts/287cdb48d8e64da3b849dceaaf9b3bc2/cuda/minparams-20261111/
```

`CANDIDATE_C_LOCAL_BACKUP.json` lists C's three checkpoints and the three paired
baseline checkpoints, with verified hashes. These weights, plus the ten original
fresh-confirmation checkpoints, are stored locally. Credentials and artifacts
remain outside Git tracking.

The original experiment checkout is preserved in Git commit `4823477` and in
`frozen-checkout-4823477.tar.gz` in the artifact directory above. Use that snapshot
when reproducing the original source-hash audit. The package reorganization
changes imports and paths; it does not change the recorded experiment results.

See [`HANDOFF.md`](docs/HANDOFF.md) for the current state and complete research history,
and [`CUDA_README.md`](docs/CUDA.md) for the CUDA training workflow.
